"""
Multi-task pretraining for MOFormer with:
1. Periodicity-aware CIF processing
2. NT-Xent contrastive + BYOL asymmetric predictor alignment
3. VICReg variance-covariance collapse prevention
4. Uncertainty-weighted multi-task balancing
5. Atom type masking + coordinate denoising auxiliary tasks
6. Momentum encoder (EMA) with full projection chain

Acceleration: AMP, fused AdamW, gradient accumulation, torch.compile, cuDNN tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import shutil
import yaml
from datetime import datetime, timedelta
from time import time
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerPretrain
from model.utils import AverageMeter
from model.cgcnn_pretrain import CrystalGraphConvNet
from dataset.periodic_cif import PeriodicCIFData, collate_pool, get_train_val_test_loader
from loss.multitask_loss import MultiTaskLoss

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_config_file(model_checkpoints_folder, config):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    config_save = {k: v for k, v in config.items() if not k.startswith('_')}
    with open(os.path.join(model_checkpoints_folder, 'config.yaml'), 'w') as f:
        yaml.dump(config_save, f, default_flow_style=False)


@torch.no_grad()
def momentum_update_models(models_q: list, models_k: list, momentum=0.999):
    """Update momentum encoder parameters for all model pairs."""
    for model_q, model_k in zip(models_q, models_k):
        for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
            param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data


def get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr=0):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _maybe_compile(model, config):
    if config.get('use_compile', False) and hasattr(torch, 'compile'):
        try:
            return torch.compile(model, mode=config.get('compile_mode', 'reduce-overhead'))
        except Exception as e:
            print(f"torch.compile failed ({e}), falling back to eager mode.")
    return model


def _make_predictor(in_dim, hidden_dim, out_dim, dropout=0.0):
    """BYOL-style predictor: Linear → BN → ReLU → Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim)
    )


def _log_grad_norms(writer, step, named_params, prefix='grad'):
    """Log gradient norms per parameter group to TensorBoard."""
    for name, param in named_params:
        if param.grad is not None:
            norm = param.grad.norm().item()
            writer.add_scalar(f'{prefix}/{name}', norm, step)


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class MultiTaskPretrain(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self._setup_cudnn()

        # TensorBoard
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs_multiview', current_time + '_multitask')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

        # Tokenizer
        self.vocab_path = config['vocab_path']
        self.tokenizer = MOFTokenizer(
            self.vocab_path, model_max_length=512, padding_side='right'
        )

        # Gradient accumulation
        self.grad_accum_steps = config.get('grad_accum_steps', 1)

        # AMP
        self.use_amp = config.get('use_amp', True) and self.device != 'cpu'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) enabled.")

        # Dataset
        self.dataset = PeriodicCIFData(
            root_dir=config['graph_dataset']['root_dir'],
            tokenizer=self.tokenizer,
            max_num_nbr=config['graph_dataset']['max_num_nbr'],
            radius=config['graph_dataset']['radius'],
            dmin=config['graph_dataset']['dmin'],
            step=config['graph_dataset']['step'],
            random_seed=config['graph_dataset']['random_seed'],
            use_periodic_augment=config.get('use_periodic_augment', True),
            use_supercell=config.get('use_supercell', True),
            min_atoms_for_supercell=config.get('min_atoms_for_supercell', 10),
        )
        self.train_loader, self.valid_loader = self._build_loaders()

        self._init_models()
        self._init_loss()      # must be BEFORE _init_optimizer (uses criterion params)
        self._init_optimizer()

        # Uncertainty warmup tracking
        self.uncertainty_warmup_epochs = config.get('uncertainty_warmup_epochs', 5)
        if self.criterion.use_uncertainty and self.uncertainty_warmup_epochs > 0:
            self.criterion.freeze_uncertainty()
            print(f"Uncertainty weights frozen for first {self.uncertainty_warmup_epochs} epochs.")

    # ----- Device & cuDNN ---------------------------------------------------

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print(f"Running on: {device}")
        return device

    def _setup_cudnn(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends, 'cuda_matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True

    def _build_loaders(self):
        cfg = self.config
        num_workers = cfg['dataloader'].get('num_workers', 4)
        prefetch = cfg.get('prefetch_factor', 2)
        persistent = cfg.get('persistent_workers', True) and num_workers > 0

        train_loader, valid_loader = get_train_val_test_loader(
            dataset=self.dataset, collate_fn=collate_pool,
            pin_memory=cfg['gpu'] != 'cpu',
            batch_size=cfg['batch_size'],
            **cfg['dataloader']
        )

        if num_workers > 0 and persistent:
            train_loader = DataLoader(
                self.dataset, batch_size=train_loader.batch_size,
                sampler=train_loader.sampler, num_workers=num_workers,
                drop_last=True, collate_fn=collate_pool,
                pin_memory=cfg['gpu'] != 'cpu',
                prefetch_factor=prefetch, persistent_workers=persistent
            )
        return train_loader, valid_loader

    # ----- Model initialization ---------------------------------------------

    def _init_models(self):
        cfg = self.config
        proj_dim = cfg['proj_dim']
        h_fea_len = cfg['model_cgcnn']['h_fea_len']
        atom_fea_len = cfg['model_cgcnn']['atom_fea_len']

        # Text branch
        self.model_t = TransformerPretrain(
            ntoken=cfg['Transformer']['ntoken'],
            d_model=cfg['Transformer']['d_model'],
            nhead=cfg['Transformer']['nhead'],
            d_hid=cfg['Transformer']['d_hid'],
            nlayers=cfg['Transformer']['nlayers'],
            dropout=cfg['Transformer']['dropout'],
            proj_dim=proj_dim,
            pad_token_id=0
        ).to(self.device)

        # Graph branch
        sample = self.dataset[0]
        structures = sample[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1] + 6  # +6 for periodic offset

        self.model_g = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            num_atom_types=cfg.get('num_atom_types', 100),
            **cfg['model_cgcnn']
        ).to(self.device)

        # ---- Projection heads (graph → contrastive space) ----
        self.graph_proj = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Linear(h_fea_len, proj_dim)
        ).to(self.device)

        self.atom_proj = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len // 2),
            nn.ReLU(),
            nn.Linear(atom_fea_len // 2, proj_dim)
        ).to(self.device)

        # ---- Predictor heads (BYOL-style, on top of projections) ----
        predictor_hidden = cfg.get('predictor_hidden', proj_dim // 2)
        self.pred_g = _make_predictor(proj_dim, predictor_hidden, proj_dim).to(self.device)
        self.pred_t = _make_predictor(proj_dim, predictor_hidden, proj_dim).to(self.device)

        # ---- Momentum encoders (EMA copies) ----
        self.model_g_m = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            num_atom_types=cfg.get('num_atom_types', 100),
            **cfg['model_cgcnn']
        ).to(self.device)
        self.graph_proj_m = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Linear(h_fea_len, proj_dim)
        ).to(self.device)
        self.atom_proj_m = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len // 2),
            nn.ReLU(),
            nn.Linear(atom_fea_len // 2, proj_dim)
        ).to(self.device)

        # Copy weights and freeze momentum models
        self._copy_to_momentum()

        # torch.compile (PT 2.0+)
        if cfg.get('use_compile', False):
            self.model_g = _maybe_compile(self.model_g, cfg)
            self.model_t = _maybe_compile(self.model_t, cfg)

        self._load_pretrained_weights()

    def _copy_to_momentum(self):
        """Initialize momentum models with online weights and freeze."""
        for online, momentum in [
            (self.model_g, self.model_g_m),
            (self.graph_proj, self.graph_proj_m),
            (self.atom_proj, self.atom_proj_m),
        ]:
            for p_o, p_m in zip(online.parameters(), momentum.parameters()):
                p_m.data.copy_(p_o.data)
                p_m.requires_grad = False

    # ----- Optimizer --------------------------------------------------------

    def _init_optimizer(self):
        cfg = self.config
        params = (
            list(self.model_t.parameters())
            + list(self.model_g.parameters())
            + list(self.graph_proj.parameters())
            + list(self.atom_proj.parameters())
            + list(self.pred_g.parameters())
            + list(self.pred_t.parameters())
            + list(self.criterion.parameters())   # temperature, uncertainty σ²
        )
        lr = cfg['optim']['init_lr']
        wd = float(cfg['optim']['weight_decay'])

        if cfg.get('use_fused_optim', True) and self.device != 'cpu':
            try:
                self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
                print("Using fused AdamW.")
            except (RuntimeError, TypeError):
                self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

        total_steps = len(self.train_loader) * cfg['epochs'] // self.grad_accum_steps
        warmup_epochs = cfg.get('warmup_epochs', 5)
        warmup_steps = len(self.train_loader) * warmup_epochs // self.grad_accum_steps
        self.scheduler = get_linear_warmup_cosine_scheduler(
            self.optimizer, warmup_steps=warmup_steps,
            total_steps=total_steps, min_lr=cfg.get('min_lr', 0)
        )
        print(f"LR: {warmup_epochs}ep warmup + cosine → min_lr={cfg.get('min_lr', 0)}")

    def _init_loss(self):
        self.criterion = MultiTaskLoss(self.config.get('loss', {})).to(self.device)

    def _load_pretrained_weights(self):
        cfg = self.config
        if cfg.get('fine_tune_from') is None:
            print("Training from scratch.")
            return

        # Resolve checkpoint directory from fine_tune_from (which includes
        # the full relative path, e.g. "runs_multiview/May13_13-02-44_multitask/checkpoints")
        ft_from = cfg['fine_tune_from']
        if os.path.isabs(ft_from):
            ckpt = ft_from
        elif ft_from.startswith('./'):
            ckpt = ft_from
        else:
            ckpt = os.path.join('.', ft_from)

        load = lambda f: torch.load(os.path.join(ckpt, f), map_location=self.device)

        try:
            # Core encoders
            self.model_t.load_state_dict(load('model_t_best.pth'), strict=False)
            self.model_g.load_state_dict(load('model_g_best.pth'), strict=False)

            # Projection heads
            self.graph_proj.load_state_dict(load('graph_proj_best.pth'), strict=False)
            self.atom_proj.load_state_dict(load('atom_proj_best.pth'), strict=False)

            # BYOL predictors
            self.pred_g.load_state_dict(load('pred_g_best.pth'), strict=False)
            self.pred_t.load_state_dict(load('pred_t_best.pth'), strict=False)

            # Momentum models (load independently, do NOT overwrite from online)
            self.model_g_m.load_state_dict(load('model_g_m_best.pth'), strict=False)
            self.graph_proj_m.load_state_dict(load('graph_proj_m_best.pth'), strict=False)
            # atom_proj_m has no separate checkpoint → init from atom_proj
            for p_o, p_m in zip(self.atom_proj.parameters(),
                                self.atom_proj_m.parameters()):
                p_m.data.copy_(p_o.data)

            print(f"Loaded pre-trained model from {ckpt} with success.")
        except FileNotFoundError as e:
            print(f"Pre-trained weights not found at {ckpt}: {e}")
            print("Training from scratch.")

    # ----- Data preparation -------------------------------------------------

    def _prepare_graph_input(self, structures, device):
        (atom_fea, nbr_fea, nbr_fea_idx, nbr_offset,
         atom_types, frac_coords, crystal_atom_idx) = structures
        return (
            atom_fea.to(device, non_blocking=True),
            nbr_fea.to(device, non_blocking=True),
            nbr_fea_idx.to(device, non_blocking=True),
            nbr_offset.to(device, non_blocking=True),
            atom_types.to(device, non_blocking=True),
            frac_coords.to(device, non_blocking=True),
            [crys_idx.to(device, non_blocking=True) for crys_idx in crystal_atom_idx],
        )

    # ----- Core step --------------------------------------------------------

    def _forward_all(self, atom_fea, nbr_fea, nbr_fea_idx, nbr_offset,
                     atom_types, frac_coords, crys_idx, text_inputs, train=True):
        """Single AMP context for all forward passes (graph + text + projections + predictors)."""
        mask_prob = self.config.get('mask_atom_prob', 0.15) if train else 0.0
        noise_std = self.config.get('coord_noise_std', 0.05) if train else 0.0

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Graph encoder
            crys_fea, atom_fea_conved, atom_mlm_logits, atom_mlm_labels, \
                coord_noise_pred, coord_noise_target = self.model_g(
                    atom_fea, nbr_fea, nbr_fea_idx, crys_idx,
                    atom_types=atom_types, frac_coords=frac_coords,
                    nbr_offset=nbr_offset,
                    mask_atom_prob=mask_prob, coord_noise_std=noise_std,
                )

            # Graph projections + predictor
            z_graph = self.graph_proj(crys_fea)
            pred_g2t = self.pred_g(z_graph)
            atom_embeds = self.atom_proj(atom_fea_conved)

            # Text encoder + predictor
            z_text, token_embeds = self.model_t(text_inputs, return_tokens=True)
            pred_t2g = self.pred_t(z_text)

        return (z_graph, pred_g2t, atom_embeds,
                z_text, pred_t2g, token_embeds,
                atom_mlm_logits, atom_mlm_labels,
                coord_noise_pred, coord_noise_target)

    def _step(self, graph_inputs, text_inputs, train=True):
        """
        Single step with all forward passes under one AMP context.
        """
        structures, _, _ = graph_inputs
        atom_fea, nbr_fea, nbr_fea_idx, nbr_offset, atom_types, frac_coords, crys_idx = \
            self._prepare_graph_input(structures, self.device)

        # ---- All forward passes (single AMP context) ----
        (z_graph, pred_g2t, atom_embeds,
         z_text, pred_t2g, token_embeds,
         atom_mlm_logits, atom_mlm_labels,
         coord_noise_pred, coord_noise_target) = \
            self._forward_all(atom_fea, nbr_fea, nbr_fea_idx, nbr_offset,
                             atom_types, frac_coords, crys_idx, text_inputs, train=train)

        # ---- Momentum forward (no grad, full precision for stability) ----
        with torch.no_grad():
            if train:
                crys_fea_m, _, _, _, _, _ = self.model_g_m(
                    atom_fea, nbr_fea, nbr_fea_idx, crys_idx,
                    atom_types=None, frac_coords=None, nbr_offset=nbr_offset,
                    mask_atom_prob=0.0, coord_noise_std=0.0,
                )
                z_graph_m = self.graph_proj_m(crys_fea_m)
            else:
                z_graph_m = z_graph

        # ---- Compute losses (cast to float32 for numerical stability) ----
        total_loss, loss_dict = self.criterion(
            z_graph.float(), z_text.float(),
            pred_g2t=pred_g2t.float(),
            pred_t2g=pred_t2g.float(),
            atom_mlm_logits=atom_mlm_logits.float() if atom_mlm_logits is not None else None,
            atom_mlm_labels=atom_mlm_labels,
            coord_noise_pred=coord_noise_pred.float() if coord_noise_pred is not None else None,
            coord_noise_target=coord_noise_target,
            atom_embeds=atom_embeds.float(),
            token_embeds=token_embeds.float(),
            crystal_atom_idx=crys_idx,
        )

        # Additional NT-Xent momentum loss
        if train:
            loss_momentum = self.criterion.nt_xent(z_graph.float(), z_graph_m.float().detach())
            total_loss = total_loss + 0.5 * loss_momentum
            loss_dict['loss_momentum'] = loss_momentum.item()

        if train and self.grad_accum_steps > 1:
            total_loss = total_loss / self.grad_accum_steps

        return total_loss, loss_dict

    # ----- Training loop ----------------------------------------------------

    def train(self):
        cfg = self.config
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = float('inf')
        grad_accum_steps = self.grad_accum_steps

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder, cfg)

        epoch_pbar = tqdm(range(cfg['epochs']), desc='Epochs', unit='epoch',
                          dynamic_ncols=True, position=0)

        for epoch_counter in epoch_pbar:
            # Unfreeze uncertainty after warmup
            if (self.criterion.use_uncertainty
                    and epoch_counter == self.uncertainty_warmup_epochs):
                self.criterion.unfreeze_uncertainty()
                tqdm.write(f'  >> Uncertainty weights unfrozen at epoch {epoch_counter+1}')

            self.model_t.train()
            self.model_g.train()
            self.graph_proj.train()
            self.atom_proj.train()
            self.pred_g.train()
            self.pred_t.train()
            self.model_g_m.eval()
            self.graph_proj_m.eval()
            self.atom_proj_m.eval()

            epoch_losses = AverageMeter()
            self.optimizer.zero_grad(set_to_none=True)

            batch_pbar = tqdm(self.train_loader, desc=f'E {epoch_counter+1}/{cfg["epochs"]}',
                              unit='b', leave=False, dynamic_ncols=True, position=1)

            for bn, (input_graph, input_text, _) in enumerate(batch_pbar):
                if cfg.get('cuda', True):
                    input_text = input_text.to(self.device, non_blocking=True)

                loss, loss_dict = self._step(
                    (input_graph, input_text, None), input_text, train=True
                )

                # Backward with AMP
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step
                if (bn + 1) % grad_accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        self._clip_gradients()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self._clip_gradients()
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    # Update ALL momentum models
                    momentum_update_models(
                        [self.model_g, self.graph_proj, self.atom_proj],
                        [self.model_g_m, self.graph_proj_m, self.atom_proj_m],
                        momentum=cfg.get('momentum', 0.999)
                    )

                epoch_losses.update(loss.item() * grad_accum_steps)

                # Update tqdm
                lr = self.scheduler.get_last_lr()[0]
                batch_pbar.set_postfix({
                    'loss': f'{loss.item() * grad_accum_steps:.3f}',
                    'ntx': f'{loss_dict.get("loss_ntxent", 0):.2f}',
                    'byol': f'{loss_dict.get("loss_byol", 0):.2f}',
                    'mlm': f'{loss_dict.get("loss_atom_mlm", 0):.2f}',
                    'lr': f'{lr:.1e}',
                })

                # TensorBoard
                if n_iter % cfg['log_every_n_steps'] == 0:
                    self._log_to_tb(loss_dict, lr, n_iter)

                    # Log gradient norms every 2x log steps
                    if n_iter % (cfg['log_every_n_steps'] * 2) == 0:
                        _log_grad_norms(self.writer, n_iter,
                                        self.model_g.named_parameters(), 'grad/g')
                        _log_grad_norms(self.writer, n_iter,
                                        self.model_t.named_parameters(), 'grad/t')
                        _log_grad_norms(self.writer, n_iter,
                                        self.pred_g.named_parameters(), 'grad/pred_g')
                        _log_grad_norms(self.writer, n_iter,
                                        self.pred_t.named_parameters(), 'grad/pred_t')

                n_iter += 1

            batch_pbar.close()
            torch.cuda.empty_cache()

            # Validation
            if epoch_counter % cfg['eval_every_n_epochs'] == 0:
                valid_loss = self._validate()
                epoch_pbar.set_postfix({
                    'train': f'{epoch_losses.avg:.4f}',
                    'valid': f'{valid_loss:.4f}',
                    'best': f'{best_valid_loss:.4f}',
                })
                self.writer.add_scalar('valid/total_loss', valid_loss, valid_n_iter)
                valid_n_iter += 1

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self._save_checkpoint(model_checkpoints_folder, 'best')
                    tqdm.write(f'  >> Best model saved (valid: {valid_loss:.4f})')

                tqdm.write(f'  E{epoch_counter+1}/{cfg["epochs"]} | '
                           f'Train {epoch_losses.avg:.4f} | Valid {valid_loss:.4f} | Best {best_valid_loss:.4f}')

            if epoch_counter > 0 and epoch_counter % cfg.get('save_every_n_epochs', 5) == 0:
                self._save_checkpoint(model_checkpoints_folder, str(epoch_counter))

            self.writer.add_scalar('train/epoch_loss', epoch_losses.avg, epoch_counter)

        epoch_pbar.close()
        tqdm.write(f'Training completed. Best validation loss: {best_valid_loss:.4f}')

    def _clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(
            list(self.model_t.parameters())
            + list(self.model_g.parameters())
            + list(self.graph_proj.parameters())
            + list(self.atom_proj.parameters())
            + list(self.pred_g.parameters())
            + list(self.pred_t.parameters()),
            max_norm=self.config.get('grad_clip', 5.0)
        )

    def _log_to_tb(self, loss_dict, lr, step):
        w = self.writer
        w.add_scalar('train/total_loss', loss_dict['total_loss'], step)
        w.add_scalar('train/ntxent', loss_dict.get('loss_ntxent', 0), step)
        w.add_scalar('train/byol', loss_dict.get('loss_byol', 0), step)
        w.add_scalar('train/byol_g2t', loss_dict.get('loss_byol_g2t', 0), step)
        w.add_scalar('train/byol_t2g', loss_dict.get('loss_byol_t2g', 0), step)
        w.add_scalar('train/vicreg', loss_dict.get('loss_vicreg', 0), step)
        w.add_scalar('train/vicreg_var_g', loss_dict.get('loss_vicreg_var_g', 0), step)
        w.add_scalar('train/vicreg_var_t', loss_dict.get('loss_vicreg_var_t', 0), step)
        w.add_scalar('train/vicreg_cov_g', loss_dict.get('loss_vicreg_cov_g', 0), step)
        w.add_scalar('train/vicreg_cov_t', loss_dict.get('loss_vicreg_cov_t', 0), step)
        w.add_scalar('train/atom_mlm', loss_dict.get('loss_atom_mlm', 0), step)
        w.add_scalar('train/coord', loss_dict.get('loss_coord', 0), step)
        w.add_scalar('train/momentum', loss_dict.get('loss_momentum', 0), step)
        w.add_scalar('train/sigma_ntxent', loss_dict.get('sigma_ntxent', 1), step)
        w.add_scalar('train/sigma_atom_mlm', loss_dict.get('sigma_atom_mlm', 1), step)
        w.add_scalar('train/sigma_coord', loss_dict.get('sigma_coord', 1), step)
        w.add_scalar('train/temperature', loss_dict.get('temperature', 0.07), step)
        w.add_scalar('train/lr', lr, step)

    # ----- Checkpointing ----------------------------------------------------

    def _save_checkpoint(self, folder, tag):
        torch.save(self.model_t.state_dict(), os.path.join(folder, f'model_t_{tag}.pth'))
        torch.save(self.model_g.state_dict(), os.path.join(folder, f'model_g_{tag}.pth'))
        torch.save(self.graph_proj.state_dict(), os.path.join(folder, f'graph_proj_{tag}.pth'))
        torch.save(self.atom_proj.state_dict(), os.path.join(folder, f'atom_proj_{tag}.pth'))
        torch.save(self.pred_g.state_dict(), os.path.join(folder, f'pred_g_{tag}.pth'))
        torch.save(self.pred_t.state_dict(), os.path.join(folder, f'pred_t_{tag}.pth'))
        torch.save(self.model_g_m.state_dict(), os.path.join(folder, f'model_g_m_{tag}.pth'))
        torch.save(self.graph_proj_m.state_dict(), os.path.join(folder, f'graph_proj_m_{tag}.pth'))
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(folder, f'training_state_{tag}.pth'))

    # ----- Validation -------------------------------------------------------

    def _validate(self):
        self.model_t.eval()
        self.model_g.eval()
        self.graph_proj.eval()
        self.atom_proj.eval()
        self.pred_g.eval()
        self.pred_t.eval()

        loss_total, total_num = 0.0, 0
        val_pbar = tqdm(self.valid_loader, desc='Validation', unit='b',
                        leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for input_graph, input_text, batch_cif_ids in val_pbar:
                if self.config.get('cuda', True):
                    input_text = input_text.to(self.device, non_blocking=True)
                loss, loss_dict = self._step(
                    (input_graph, input_text, batch_cif_ids), input_text, train=False
                )
                loss_total += loss.item() * len(batch_cif_ids)
                total_num += len(batch_cif_ids)
                val_pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        val_pbar.close()

        self.model_t.train()
        self.model_g.train()
        self.graph_proj.train()
        self.atom_proj.train()
        self.pred_g.train()
        self.pred_t.train()

        return loss_total / max(total_num, 1)


if __name__ == "__main__":
    config = yaml.load(open("config_multitask.yaml", "r"), Loader=yaml.FullLoader)
    print("Configuration:")
    for k, v in config.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")

    trainer = MultiTaskPretrain(config)
    trainer.train()
