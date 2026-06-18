"""
Multi-task pretraining losses for MOFormer.

Includes:
- NT-Xent (InfoNCE) contrastive loss with learnable temperature
- BYOL-style asymmetric predictor loss (cosine similarity)
- VICReg variance-covariance regularization (prevents embedding collapse)
- Uncertainty weighting for dynamic multi-task balance
- Atom type MLM loss (cross-entropy)
- Coordinate denoising loss (SmoothL1)
- Node-to-token alignment loss (optional, OT-based)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# NT-Xent (InfoNCE) contrastive loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (InfoNCE / SimCLR).

    Symmetric: treats (z1_i, z2_i) as positive pairs across the two views.
    Uses a learnable temperature parameter.
    """
    def __init__(self, temperature_init: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.tau = nn.Parameter(torch.tensor(temperature_init))
        else:
            self.register_buffer('tau', torch.tensor(temperature_init))

    def forward(self, z1, z2):
        """
        Args:
            z1: (N, D) first view embeddings (already L2-normalized recommended)
            z2: (N, D) second view embeddings

        Returns:
            scalar contrastive loss
        """
        N = z1.shape[0]
        device = z1.device

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)                        # [2N, D]
        sim = torch.mm(z, z.t()) / self.tau                   # [2N, 2N]

        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)], dim=0).to(device)

        mask = torch.eye(2*N, device=device).bool()
        sim = sim.masked_fill(mask, float('-inf'))

        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# BYOL-style asymmetric predictor loss
# ---------------------------------------------------------------------------

class BYOLPredictorLoss(nn.Module):
    """
    Asymmetric cosine-similarity loss with stop-gradient on the target.

    pred(z_query) → stopgrad(z_target)
    Loss = 2 - 2 * cosine_sim(pred, target)

    This is the BYOL/SimSiam style loss that prevents collapse via asymmetry
    between the query+prediction and target branches.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_query, z_target):
        """
        Args:
            pred_query: (N, D) predicted embeddings from query branch
            z_target: (N, D) target embeddings (stop-grad applied BEFORE calling)

        Returns:
            scalar loss in [0, 4]
        """
        pred_q = F.normalize(pred_query, dim=1)
        z_t = F.normalize(z_target, dim=1)
        return 2 - 2 * (pred_q * z_t).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# VICReg — variance + covariance regularization (collapse prevention)
# ---------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization.

    Prevents embedding collapse by:
    - Variance loss: pushes per-dimension std away from zero
    - Covariance loss: decorrelates embedding dimensions

    IMPORTANT: Must be applied to CENTERED but NOT L2-normalized embeddings.
    If applied after L2 normalization, variance is trivially 1 and the
    variance term becomes useless.
    """
    def __init__(self, var_gamma: float = 1.0, cov_lambda: float = 0.005):
        super().__init__()
        self.var_gamma = var_gamma
        self.cov_lambda = cov_lambda
        self.eps = 1e-4

    def forward(self, z):
        """
        Args:
            z: (N, D) raw projection embeddings (NOT L2-normalized)

        Returns:
            total_loss: scalar
            var_loss: scalar (for logging)
            cov_loss: scalar (for logging)
        """
        N, D = z.shape

        # Center the embeddings
        z_centered = z - z.mean(dim=0)                        # (N, D)

        # Variance loss: encourage std ≥ gamma
        std = torch.sqrt(z_centered.var(dim=0) + self.eps)    # (D,)
        var_loss = F.relu(self.var_gamma - std).mean()

        # Covariance loss: penalize off-diagonal elements
        cov = (z_centered.t() @ z_centered) / (N - 1)         # (D, D)
        diag = torch.diag(cov)
        # Zero out diagonal, sum squared off-diagonals
        off_diag = cov - torch.diag(diag)
        n_off = D * (D - 1)
        cov_loss = (off_diag ** 2).sum() / max(n_off, 1)

        total = var_loss + self.cov_lambda * cov_loss
        return total, var_loss, cov_loss


# ---------------------------------------------------------------------------
# Uncertainty weighting (Kendall et al.)
# ---------------------------------------------------------------------------

class UncertaintyWeighting(nn.Module):
    """
    Homoscedastic uncertainty weighting for multi-task learning.

    Each task loss L_i is weighted by a learned log-variance σ_i²:
        L_weighted = L_i / (2 * σ_i²) + log(σ_i)

    The log(σ) term prevents σ from growing to infinity.
    """
    def __init__(self, n_tasks: int, init_log_sigma: float = 0.0):
        """
        Args:
            n_tasks: number of task losses to weight
            init_log_sigma: initial value for log-σ²
                - 0.0 → σ ≈ 1.0 (reasonable initial uncertainty)
                - Positive → more uncertain
        """
        super().__init__()
        self.log_sigma2 = nn.Parameter(
            torch.full((n_tasks,), init_log_sigma)
        )

    def forward(self, losses):
        """
        Args:
            losses: (n_tasks,) tensor of raw task losses

        Returns:
            weighted_sum: scalar
            weights: (n_tasks,) tensor of per-task weights (for logging)
            sigma_vals: (n_tasks,) tensor of σ values (for logging)
        """
        sigma2 = torch.exp(self.log_sigma2) + 1e-6
        sigma = torch.sqrt(sigma2)
        precision = 1.0 / (2.0 * sigma2)
        weighted = precision * losses + 0.5 * self.log_sigma2
        return weighted.sum(), precision.detach(), sigma.detach()


# ---------------------------------------------------------------------------
# Task-specific losses
# ---------------------------------------------------------------------------

class AtomMLMLoss(nn.Module):
    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        if labels is None or logits is None:
            return None
        n_masked = (labels != self.ignore_index).sum()
        if n_masked == 0:
            return None
        return self.loss_fn(logits, labels)


class CoordDenoiseLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, noise_pred, noise_target):
        if noise_pred is None or noise_target is None:
            return None
        return self.loss_fn(noise_pred, noise_target)


class NodeTokenAlignLoss(nn.Module):
    """
    Optimal Transport based alignment (optional, off by default).
    """
    def __init__(self, epsilon: float = 0.1, max_iter: int = 50, tol: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol

    def sinkhorn(self, cost, mu, nu):
        N, M = cost.shape
        device = cost.device
        K = torch.exp(-cost / self.epsilon)
        u = torch.ones(N, device=device) / N
        v = torch.ones(M, device=device) / M
        for _ in range(self.max_iter):
            u0, v0 = u.clone(), v.clone()
            u = mu / (K @ v + 1e-10)
            v = nu / (K.t() @ u + 1e-10)
            if (u - u0).abs().max() < self.tol and (v - v0).abs().max() < self.tol:
                break
        return torch.diag(u) @ K @ torch.diag(v)

    def forward(self, atom_embeds, token_embeds, crystal_atom_idx=None):
        B = token_embeds.shape[0]
        if crystal_atom_idx is None:
            return None
        total_loss = 0.0
        for b in range(B):
            if b >= len(crystal_atom_idx):
                continue
            idx = crystal_atom_idx[b]
            atom_b = atom_embeds[idx]
            token_b = token_embeds[b]
            token_mask = token_b.abs().sum(dim=1) > 1e-8
            token_b = token_b[token_mask]
            if len(token_b) == 0 or len(atom_b) == 0:
                continue
            atom_b = F.normalize(atom_b, dim=1)
            token_b = F.normalize(token_b, dim=1)
            cost = 1 - torch.mm(atom_b, token_b.t())
            mu = torch.ones(len(atom_b), device=atom_embeds.device) / max(len(atom_b), 1)
            nu = torch.ones(len(token_b), device=atom_embeds.device) / max(len(token_b), 1)
            T = self.sinkhorn(cost, mu, nu)
            total_loss += (T * cost).sum()
        return total_loss / max(B, 1)


# ---------------------------------------------------------------------------
# Combined multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task pretraining loss with:
    - NT-Xent contrastive (projection space)
    - BYOL asymmetric predictor (cross-modal prediction)
    - VICReg collapse prevention (variance + covariance)
    - Uncertainty-weighted task losses
    - Atom MLM + Coordinate denoise auxiliary tasks

    L_total = Σ uncertainty_weight(L_task_i)                         [task losses]
            + lambda_vicreg * L_vicreg                                [regularizer]
            + lambda_byol  * (L_byol_g2t + L_byol_t2g)              [alignment]
    """
    def __init__(self, config: dict):
        super().__init__()

        # Contrastive loss
        self.nt_xent = NTXentLoss(
            temperature_init=config.get('temperature', 0.07),
            learnable_temp=config.get('learnable_temp', True)
        )

        # BYOL predictor loss
        self.byol = BYOLPredictorLoss()

        # VICReg collapse prevention
        self.vicreg = VICRegLoss(
            var_gamma=config.get('vicreg_var_gamma', 1.0),
            cov_lambda=config.get('vicreg_cov_lambda', 0.005)
        )

        # Task-specific losses
        self.atom_mlm = AtomMLMLoss()
        self.coord_denoise = CoordDenoiseLoss(beta=config.get('coord_beta', 0.1))
        self.node_align = NodeTokenAlignLoss(
            epsilon=config.get('ot_epsilon', 0.1),
            max_iter=config.get('ot_max_iter', 50)
        )

        # Uncertainty weighting (3 tasks: ntxent, atom_mlm, coord_denoise)
        self.use_uncertainty = config.get('use_uncertainty_weighting', True)
        if self.use_uncertainty:
            self.uncertainty = UncertaintyWeighting(
                n_tasks=3,
                init_log_sigma=config.get('uncertainty_init_log_sigma', 0.0)
            )

        # Fixed weights
        self.lambda_atom_mlm = config.get('lambda_atom_mlm', 1.0)
        self.lambda_coord = config.get('lambda_coord', 0.1)
        self.lambda_node = config.get('lambda_node', 0.0)
        self.lambda_vicreg = config.get('lambda_vicreg', 0.1)
        self.lambda_byol = config.get('lambda_byol', 1.0)

    def forward(self, z_graph, z_text,
                pred_g2t=None, pred_t2g=None,
                atom_mlm_logits=None, atom_mlm_labels=None,
                coord_noise_pred=None, coord_noise_target=None,
                atom_embeds=None, token_embeds=None, crystal_atom_idx=None):
        """
        Compute total multi-task loss.

        Args:
            z_graph: (B, D) raw graph projections (NOT L2-normalized)
            z_text:  (B, D) raw text projections  (NOT L2-normalized)
            pred_g2t: (B, D) predictor output: graph → text
            pred_t2g: (B, D) predictor output: text → graph
            atom_mlm_logits: (N, num_types) or None
            atom_mlm_labels: (N,) or None
            coord_noise_pred: (N, 3) or None
            coord_noise_target: (N, 3) or None
            atom_embeds: (N, D) convolved atom embeddings or None
            token_embeds: (B, L, D) token embeddings or None
            crystal_atom_idx: list of LongTensors or None

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        device = z_graph.device

        # ------ 1. NT-Xent contrastive loss ------
        loss_ntxent = self.nt_xent(z_graph, z_text)

        # ------ 2. BYOL asymmetric predictor losses ------
        loss_byol = torch.tensor(0.0, device=device)
        loss_byol_g2t = torch.tensor(0.0, device=device)
        loss_byol_t2g = torch.tensor(0.0, device=device)
        if pred_g2t is not None and self.lambda_byol > 0:
            loss_byol_g2t = self.byol(pred_g2t, z_text.detach())
            loss_byol_t2g = self.byol(pred_t2g, z_graph.detach())
            loss_byol = loss_byol_g2t + loss_byol_t2g

        # ------ 3. VICReg collapse prevention ------
        loss_vicreg_graph, var_g, cov_g = self.vicreg(z_graph)
        loss_vicreg_text, var_t, cov_t = self.vicreg(z_text)
        loss_vicreg = loss_vicreg_graph + loss_vicreg_text

        # ------ 4. Task losses ------
        loss_atom_mlm = self.atom_mlm(atom_mlm_logits, atom_mlm_labels)
        loss_coord = self.coord_denoise(coord_noise_pred, coord_noise_target)
        loss_node = torch.tensor(0.0, device=device)
        if atom_embeds is not None and token_embeds is not None and self.lambda_node > 0:
            loss_node = self.node_align(atom_embeds, token_embeds, crystal_atom_idx)
            if loss_node is None:
                loss_node = torch.tensor(0.0, device=device)

        # ------ 5. Uncertainty-weighted combination (only task losses) ------
        if self.use_uncertainty:
            # Collect task losses
            task_losses = [loss_ntxent]
            if loss_atom_mlm is not None:
                task_losses.append(loss_atom_mlm * self.lambda_atom_mlm)
            else:
                task_losses.append(torch.tensor(0.0, device=device))
            if loss_coord is not None:
                task_losses.append(loss_coord * self.lambda_coord)
            else:
                task_losses.append(torch.tensor(0.0, device=device))
            task_losses = torch.stack(task_losses)

            loss_weighted, precision_vals, sigma_vals = self.uncertainty(task_losses)

            total_loss = (loss_weighted
                          + self.lambda_byol * loss_byol
                          + self.lambda_vicreg * loss_vicreg
                          + self.lambda_node * loss_node)
        else:
            # Fixed weights
            loss_mlm_w = loss_atom_mlm * self.lambda_atom_mlm if loss_atom_mlm is not None else torch.tensor(0.0, device=device)
            loss_coord_w = loss_coord * self.lambda_coord if loss_coord is not None else torch.tensor(0.0, device=device)

            total_loss = (loss_ntxent
                          + loss_mlm_w
                          + loss_coord_w
                          + self.lambda_byol * loss_byol
                          + self.lambda_vicreg * loss_vicreg
                          + self.lambda_node * loss_node)
            precision_vals = torch.ones(3, device=device)
            sigma_vals = torch.ones(3, device=device)

        # ------ 6. Build loss dict for logging ------
        loss_dict = {
            'loss_ntxent': loss_ntxent.item(),
            'loss_byol': loss_byol.item() if isinstance(loss_byol, torch.Tensor) else loss_byol,
            'loss_byol_g2t': loss_byol_g2t.item() if isinstance(loss_byol_g2t, torch.Tensor) else loss_byol_g2t,
            'loss_byol_t2g': loss_byol_t2g.item() if isinstance(loss_byol_t2g, torch.Tensor) else loss_byol_t2g,
            'loss_vicreg': loss_vicreg.item() if isinstance(loss_vicreg, torch.Tensor) else loss_vicreg,
            'loss_vicreg_var_g': var_g.item(),
            'loss_vicreg_var_t': var_t.item(),
            'loss_vicreg_cov_g': cov_g.item(),
            'loss_vicreg_cov_t': cov_t.item(),
            'loss_atom_mlm': loss_atom_mlm.item() if loss_atom_mlm is not None and isinstance(loss_atom_mlm, torch.Tensor) else (loss_atom_mlm if loss_atom_mlm is not None else 0),
            'loss_coord': loss_coord.item() if loss_coord is not None and isinstance(loss_coord, torch.Tensor) else (loss_coord if loss_coord is not None else 0),
            'loss_node': loss_node.item() if isinstance(loss_node, torch.Tensor) else (loss_node if loss_node is not None else 0),
            'total_loss': total_loss.item(),
            'temperature': self.nt_xent.tau.item() if hasattr(self.nt_xent.tau, 'item') else self.nt_xent.tau,
            'sigma_ntxent': sigma_vals[0].item() if len(sigma_vals) > 0 else 1.0,
            'sigma_atom_mlm': sigma_vals[1].item() if len(sigma_vals) > 1 else 1.0,
            'sigma_coord': sigma_vals[2].item() if len(sigma_vals) > 2 else 1.0,
        }

        return total_loss, loss_dict

    def freeze_uncertainty(self):
        """Freeze uncertainty parameters for warmup period."""
        if self.use_uncertainty:
            self.uncertainty.log_sigma2.requires_grad = False

    def unfreeze_uncertainty(self):
        """Unfreeze uncertainty parameters after warmup."""
        if self.use_uncertainty:
            self.uncertainty.log_sigma2.requires_grad = True
