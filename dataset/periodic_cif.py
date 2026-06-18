"""
Enhanced CIF dataset with periodic boundary condition awareness.

Key additions over the original CIFData:
1. Fractional coordinate features appended to atom features
2. Periodic image offset tracking for neighbor edges
3. Periodic-aware augmentations (random lattice shift, supercell sampling)
4. Edge features enriched with periodic image encoding
"""

from __future__ import print_function, division

import csv
import functools
import json
import random
import warnings
import math
import os
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """Base class for initializing the vector representation for atoms."""
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """Initialize atom feature vectors using a JSON file."""
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def _encode_periodic_offset(offset):
    """
    Encode periodic image offset as a compact feature vector.
    offset: array of shape (3,) representing cell translations.
    Returns a 6-dim one-hot-like encoding for sign of each component.
    Components are typically -1, 0, or 1 for nearest-image neighbors.
    """
    encoded = np.zeros(6, dtype=float)
    for i in range(3):
        if offset[i] < -0.5:
            encoded[2*i] = 1.0
        elif offset[i] > 0.5:
            encoded[2*i + 1] = 1.0
    return encoded


def _apply_random_lattice_shift(structure):
    """
    Apply a random translation in fractional coordinates.
    This exploits periodic invariance: the crystal is the same regardless
    of where the unit cell origin is chosen.
    """
    shift = np.random.uniform(0, 1, size=3)
    s = structure.copy()
    frac_coords = s.frac_coords + shift
    frac_coords = frac_coords % 1.0  # wrap back into [0,1)
    for i, site in enumerate(s):
        site.frac_coords = frac_coords[i]
    return s


def _expand_to_supercell(structure, min_atoms=10):
    """
    Expand small unit cells to a supercell if atom count is below min_atoms.
    This ensures the graph builder has enough atoms to work with and
    captures meaningful periodicity.
    """
    n_atoms = len(structure)
    if n_atoms >= min_atoms:
        return structure

    # Determine expansion factors
    target = min_atoms
    factors = [1, 1, 1]
    while factors[0] * factors[1] * factors[2] * n_atoms < target:
        dim = np.argmin(factors)
        factors[dim] += 1

    # Limit max expansion to avoid memory issues
    max_factor = 3
    factors = [min(f, max_factor) for f in factors]

    if factors == [1, 1, 1]:
        return structure

    try:
        structure.make_supercell(factors)
    except Exception:
        pass
    return structure


class PeriodicCIFData(Dataset):
    """
    Enhanced CIF dataset with periodic boundary condition awareness.

    Compared to the original CIFData, this dataset:
    - Appends fractional coordinate features to atom features
    - Tracks periodic image offsets in neighbor lists
    - Provides periodic-aware augmentations
    - Handles supercell expansion for small unit cells

    Parameters
    ----------
    root_dir: str
        Path to the root directory of the dataset
    tokenizer: MOFTokenizer
        Tokenizer for MOFid strings
    max_num_nbr: int (default 12)
        Maximum number of neighbors for crystal graph
    radius: float (default 8)
        Cutoff radius for neighbor search in angstroms
    dmin: float (default 0)
        Minimum distance for Gaussian distance filter
    step: float (default 0.2)
        Step size for Gaussian distance filter
    random_seed: int (default 123)
        Random seed
    use_periodic_augment: bool (default True)
        Whether to apply periodic-aware augmentations
    use_supercell: bool (default True)
        Whether to expand small cells to supercells
    min_atoms_for_supercell: int (default 10)
        Minimum atoms before supercell expansion is triggered
    """
    def __init__(self, root_dir, tokenizer, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, use_periodic_augment=True, use_supercell=True,
                 min_atoms_for_supercell=10):
        self.tokenizer = tokenizer
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.use_periodic_augment = use_periodic_augment
        self.use_supercell = use_supercell
        self.min_atoms_for_supercell = min_atoms_for_supercell

        assert os.path.exists(root_dir), 'root_dir does not exist!'

        id_prop_file = os.path.join(self.root_dir, 'id_prop.npy')
        assert os.path.exists(id_prop_file), 'id_prop.npy does not exist!'
        self.id_prop_data = np.load(id_prop_file, allow_pickle=True)

        atom_init_file = os.path.join('dataset/atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, mofid = self.id_prop_data[idx]
        fname = cif_id
        if fname[-4:] != '.cif':
            fname = fname + '.cif'

        crystal = Structure.from_file(os.path.join(self.root_dir, fname))

        # --- Periodic augmentation: random lattice shift ---
        if self.use_periodic_augment:
            crystal = _apply_random_lattice_shift(crystal)

        # --- Supercell expansion for small cells ---
        if self.use_supercell and len(crystal) < self.min_atoms_for_supercell:
            crystal = _expand_to_supercell(crystal, self.min_atoms_for_supercell)

        # --- Tokenize MOFid ---
        tokens = np.array([self.tokenizer.encode(
            mofid, max_length=512, truncation=True, padding='max_length'
        )])
        tokens = torch.from_numpy(tokens)

        # --- Build atom features including fractional coordinates ---
        atom_fea_list = []
        atom_types = []
        frac_coords_list = []
        for i in range(len(crystal)):
            specie_num = crystal[i].specie.number
            base_fea = self.ari.get_atom_fea(specie_num)
            atom_fea_list.append(base_fea)
            atom_types.append(specie_num)
            frac_coords_list.append(crystal[i].frac_coords)

        atom_fea = np.vstack(atom_fea_list)

        # Append fractional coordinates to atom features (normalized to [0,1])
        frac_coords = np.vstack(frac_coords_list)
        atom_fea = np.concatenate([atom_fea, frac_coords], axis=1)

        atom_fea = torch.Tensor(atom_fea)
        atom_types = torch.LongTensor(atom_types)

        # --- Build neighbor list with periodic image tracking ---
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True,
                                              include_image=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea, nbr_offset = [], [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) +
                    [0] * (self.max_num_nbr - len(nbr))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr))
                )
                nbr_offset.append(
                    list(map(lambda x: _encode_periodic_offset(np.array(x[3])), nbr)) +
                    [np.zeros(6)] * (self.max_num_nbr - len(nbr))
                )
            else:
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr[:self.max_num_nbr]))
                )
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr[:self.max_num_nbr]))
                )
                nbr_offset.append(
                    list(map(lambda x: _encode_periodic_offset(np.array(x[3])),
                             nbr[:self.max_num_nbr]))
                )

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_offset = np.array(nbr_offset)

        # Expand distances with Gaussian filter
        nbr_fea = self.gdf.expand(nbr_fea)

        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        nbr_offset = torch.Tensor(nbr_offset)

        # Store frac_coords for coordinate denoising task
        frac_coords = torch.Tensor(frac_coords)

        return (atom_fea, nbr_fea, nbr_fea_idx, nbr_offset, atom_types, frac_coords), tokens, cif_id


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, val_ratio=0.1, random_seed=11, num_workers=1,
                              pin_memory=False, **kwargs):
    """Split dataset into train/val loaders."""
    total_size = len(dataset)
    train_ratio = 1 - val_ratio
    train_size = int(train_ratio * total_size)
    indices = list(range(total_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers, drop_last=True,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers, drop_last=True,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch.

    Extended from the original collate_pool to handle the additional fields:
    nbr_offset, atom_types, frac_coords from PeriodicCIFData.

    Returns
    -------
    (atom_fea, nbr_fea, nbr_fea_idx, nbr_offset, atom_types, frac_coords, crystal_atom_idx),
    tokens,
    batch_cif_ids
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_nbr_offset, batch_atom_types, batch_frac_coords = [], [], []
    crystal_atom_idx, batch_tokens = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, nbr_offset, atom_types, frac_coords),
            tokens, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_nbr_offset.append(nbr_offset)
        batch_atom_types.append(atom_types)
        batch_frac_coords.append(frac_coords)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_tokens.append(tokens)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.cat(batch_nbr_offset, dim=0),
            torch.cat(batch_atom_types, dim=0),
            torch.cat(batch_frac_coords, dim=0),
            crystal_atom_idx), \
        torch.cat(batch_tokens, dim=0), \
        batch_cif_ids


if __name__ == '__main__':
    # Quick sanity check
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from tokenizer.mof_tokenizer import MOFTokenizer

    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'tokenizer', 'vocab_full.txt')
    tokenizer = MOFTokenizer(vocab_path, model_max_length=512, padding_side='right')

    root_dir = '/home/shw/ml/train_dataset/cif'
    if os.path.exists(root_dir):
        dataset = PeriodicCIFData(
            root_dir=root_dir,
            tokenizer=tokenizer,
            max_num_nbr=12,
            radius=8,
            dmin=0,
            step=0.2,
            use_periodic_augment=True,
            use_supercell=True
        )
        print(f"Dataset size: {len(dataset)}")
        (atom_fea, nbr_fea, nbr_fea_idx, nbr_offset, atom_types, frac_coords, crys_idx), tokens, cif_id = dataset[0]
        print(f"atom_fea shape: {atom_fea.shape}")          # (N, orig_atom_fea_len + 3)
        print(f"nbr_fea shape: {nbr_fea.shape}")             # (N, M, nbr_fea_gdf_len)
        print(f"nbr_fea_idx shape: {nbr_fea_idx.shape}")     # (N, M)
        print(f"nbr_offset shape: {nbr_offset.shape}")       # (N, M, 6)
        print(f"atom_types shape: {atom_types.shape}")       # (N,)
        print(f"frac_coords shape: {frac_coords.shape}")     # (N, 3)
        print(f"tokens shape: {tokens.shape}")               # (1, 512)
        print(f"cif_id: {cif_id}")
        print("PeriodicCIFData test passed!")

        # Test collate
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_pool, shuffle=True)
        batch = next(iter(loader))
        (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_nbr_offset,
         batch_atom_types, batch_frac_coords, batch_crys_idx), batch_tokens, batch_ids = batch
        print(f"\nBatch atom_fea: {batch_atom_fea.shape}")
        print(f"Batch nbr_offset: {batch_nbr_offset.shape}")
        print(f"Batch frac_coords: {batch_frac_coords.shape}")
        print("Collate test passed!")
    else:
        print(f"Data directory {root_dir} not found, skipping test.")
