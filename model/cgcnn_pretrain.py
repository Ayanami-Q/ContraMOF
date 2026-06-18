from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs.
    Supports optional periodic offset features on edges.
    """
    def __init__(self, atom_fea_len, nbr_fea_len, use_offset=False):
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.use_offset = use_offset

        total_nbr_len = 2*self.atom_fea_len + self.nbr_fea_len
        self.fc_full = nn.Linear(total_nbr_len, 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed.to(atom_in_fea.dtype))
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Crystal graph convolutional neural network with:
    - Main graph encoder for crystal-level embeddings
    - Atom type MLM head for predicting masked atom species
    - Coordinate denoising head for predicting coordinate noise
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 num_atom_types=100, drop_ratio=0):
        super(CrystalGraphConvNet, self).__init__()
        self.drop_ratio = drop_ratio
        self.atom_fea_len = atom_fea_len
        self.num_atom_types = num_atom_types
        self.orig_atom_fea_len = orig_atom_fea_len
        self.base_fea_len = orig_atom_fea_len - 3

        # Main embedding: maps (orig_atom_fea_len + 3 frac_coords) -> atom_fea_len
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Learned mask embedding for atom type masking (matches base feature dim)
        self.atom_mask_embed = nn.Parameter(torch.randn(1, self.base_fea_len) * 0.02)

        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        # Graph-level projection head for contrastive learning
        self.fc_head = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len),
            nn.Softplus(),
            nn.Linear(h_fea_len, h_fea_len)
        )

        # --- Atom Type MLM Head ---
        # Predicts atom type from convolved atom features
        self.atom_type_head = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(atom_fea_len * 2, num_atom_types)
        )

        # --- Coordinate Denoising Head ---
        # Predicts 3D noise vector added to fractional coordinates
        self.coord_denoise_head = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.SiLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(atom_fea_len, 3)
        )

    def mask_atom_types(self, atom_fea, atom_types, mask_prob=0.15):
        """
        Randomly mask atom type features and return masked features + labels.

        Args:
            atom_fea: (N, orig_atom_fea_len) original atom features
            atom_types: (N,) atom type indices
            mask_prob: probability of masking each atom

        Returns:
            masked_atom_fea: (N, orig_atom_fea_len) features with masked positions
            atom_mlm_labels: (N,) atom types with -1 for non-masked positions
        """
        N = atom_fea.shape[0]
        device = atom_fea.device

        # Randomly select atoms to mask
        mask = torch.rand(N, device=device) < mask_prob

        masked_atom_fea = atom_fea.clone()
        masked_atom_fea[mask] = self.atom_mask_embed.to(device=device, dtype=atom_fea.dtype)

        # Labels: original type for masked, -1 for others
        atom_mlm_labels = torch.full((N,), -1, dtype=torch.long, device=device)
        atom_mlm_labels[mask] = atom_types[mask].long()

        return masked_atom_fea, atom_mlm_labels

    def add_coord_noise(self, frac_coords, noise_std=0.05):
        """
        Add Gaussian noise to fractional coordinates for denoising task.
        Wrap around using periodic boundary conditions (mod 1).

        Args:
            frac_coords: (N, 3) fractional coordinates in [0, 1]
            noise_std: standard deviation of Gaussian noise

        Returns:
            noisy_frac_coords: (N, 3) noise-corrupted fractional coordinates
            coord_noise: (N, 3) the actual noise added (prediction target)
        """
        device = frac_coords.device
        noise = torch.randn_like(frac_coords) * noise_std
        noisy_frac_coords = torch.fmod(frac_coords + noise + 1.0, 1.0)
        # Store the effective noise (after wrapping)
        # For simplicity, we use the raw noise as target - model learns to denoise
        return noisy_frac_coords, noise

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx,
                atom_types=None, frac_coords=None, nbr_offset=None,
                mask_atom_prob=0.0, coord_noise_std=0.0):
        """
        Forward pass with optional atom masking and coordinate denoising.

        Args:
            atom_fea: (N, orig_atom_fea_len + 3) atom features with frac_coords
            nbr_fea: (N, M, nbr_fea_len) bond features (Gaussian distance expansion)
            nbr_fea_idx: (N, M) neighbor indices
            crystal_atom_idx: list of LongTensors mapping crystal to atom idx
            atom_types: (N,) optional atom type indices for MLM
            frac_coords: (N, 3) optional fractional coordinates for denoising
            nbr_offset: (N, M, 6) optional periodic image offset encoding
            mask_atom_prob: float, probability for atom type masking (0 = disabled)
            coord_noise_std: float, std for coordinate noise (0 = disabled)

        Returns:
            crys_fea: (N0, h_fea_len) crystal-level embeddings
            atom_fea_conved: (N, atom_fea_len) post-convolution atom features (pre-pooling)
            atom_mlm_logits: (N, num_atom_types) or None
            atom_mlm_labels: (N,) or None
            coord_noise_pred: (N, 3) or None
            coord_noise_target: (N, 3) or None
        """
        # Separate base features from fractional coordinates
        base_fea_len = atom_fea.shape[1] - 3
        base_atom_fea = atom_fea[:, :base_fea_len]
        frac_coords_in = atom_fea[:, base_fea_len:]

        # --- Atom type masking ---
        atom_mlm_logits = None
        atom_mlm_labels = None
        if mask_atom_prob > 0 and atom_types is not None:
            base_atom_fea, atom_mlm_labels = self.mask_atom_types(
                base_atom_fea, atom_types, mask_atom_prob
            )

        # --- Coordinate denoising ---
        coord_noise_pred = None
        coord_noise_target = None
        if coord_noise_std > 0 and frac_coords is not None:
            frac_coords_in, coord_noise_target = self.add_coord_noise(
                frac_coords, coord_noise_std
            )

        # Reassemble atom features
        atom_fea_combined = torch.cat([base_atom_fea, frac_coords_in], dim=1)

        # Incorporate periodic offset into neighbor features
        if nbr_offset is not None:
            nbr_fea_aug = torch.cat([nbr_fea, nbr_offset], dim=2)
        else:
            nbr_fea_aug = nbr_fea

        # Embed atom features
        atom_fea_emb = self.embedding(atom_fea_combined)

        # Graph convolutions
        for conv_func in self.convs:
            atom_fea_emb = conv_func(atom_fea_emb, nbr_fea_aug, nbr_fea_idx)

        # --- Atom type prediction ---
        if mask_atom_prob > 0 and atom_types is not None:
            atom_mlm_logits = self.atom_type_head(atom_fea_emb)

        # --- Coordinate denoising prediction ---
        if coord_noise_std > 0 and frac_coords is not None:
            coord_noise_pred = self.coord_denoise_head(atom_fea_emb)

        # --- Crystal-level pooling ---
        crys_fea = self.pooling(atom_fea_emb, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.fc_head(crys_fea)

        return crys_fea, atom_fea_emb, atom_mlm_logits, atom_mlm_labels, coord_noise_pred, coord_noise_target

    def pooling(self, atom_fea, crystal_atom_idx):
        """Mean-pooling atom features to crystal features."""
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
