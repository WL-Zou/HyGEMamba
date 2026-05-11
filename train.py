#!/usr/bin/env python3
"""
HyGE-Mamba training script v3.
===================================
Supported training modes (--target_mode):
  curve  : four growth curves (Maize.AboveGround.Wt, LAI, LeafWt, StemWt)
  scalar : three scalar traits (yield, plant_height_cm, ear_height_cm)
  all    : all seven targets (three scalar traits plus four growth curves)

Usage:
    python train.py --h5 data/hyge_mamba_dataset.h5 --split M1 --target_mode all
    python train.py --h5 data/hyge_mamba_dataset.h5 --split M1 --target_mode curve
    python train.py --h5 data/hyge_mamba_dataset.h5 --split M1 --target_mode scalar
        --primary_task yield --patience 30
"""

import os, sys, argparse, json, time, math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.models.hyge_mamba import HyGEMamba, HyGEMambaConfig, HyGEMambaOutput


# Constants
CURVE_NAMES  = ['Maize.AboveGround.Wt', 'LAI', 'LeafWt', 'StemWt']
SCALAR_NAMES = ['yield', 'plant_height_cm', 'ear_height_cm']

TASK_CN = {
    'Maize.AboveGround.Wt': 'Above-ground biomass',
    'LAI':                   'Leaf area index',
    'LeafWt':                'Leaf weight',
    'StemWt':                'Stem weight',
    'yield':                 'Yield',
    'plant_height_cm':       'Plant height (cm)',
    'ear_height_cm':         'Ear height (cm)',
}


# Dataset
class MyH5Dataset(Dataset):
    """
    HDF5 dataset supporting curve and scalar targets.

    targets (curves): (N, 4, T) or (N, T, 4)
    scalar_targets:   (N, 3) = [yield, plant_height_cm, ear_height_cm]
    """

    def __init__(self, h5_path, split='M1', partition='train',
                 target_mode='all', preload=True):
        """
        target_mode: 'curve', 'scalar', 'all'
        """
        self.h5_path = h5_path
        self._h5_file = None
        self.target_mode = target_mode
        self.has_curves = target_mode in ('curve', 'all')
        self.has_scalars = target_mode in ('scalar', 'all')

        with h5py.File(h5_path, 'r') as f:
            self.indices = f[f'splits/{split}/{partition}_indices'][:]
            self.num_snps = int(f.attrs['num_snps'])
            self.num_days = int(f.attrs['num_days'])
            self.num_wf = int(f.attrs['num_weather_features'])
            sg = f[f'splits/{split}']
            self.use_geno = bool(sg.attrs.get('use_genotype', True))
            self.use_env  = bool(sg.attrs.get('use_environment', True))
            self.has_apsim = 'apsim_params' in f

            is_shared = f.attrs.get('is_shared_genomic_coords', False)
            if is_shared or f['chromosomes'].ndim == 1:
                self._shared_chromosomes = torch.from_numpy(
                    f['chromosomes'][:].astype(np.int64))
                self._shared_positions = torch.from_numpy(
                    f['positions'][:].astype(np.int64))
            else:
                self._shared_chromosomes = torch.from_numpy(
                    f['chromosomes'][0].astype(np.int64))
                self._shared_positions = torch.from_numpy(
                    f['positions'][0].astype(np.int64))

            #  targets 
            self.targets_layout = None
            self.num_timesteps = 0
            if self.has_curves and 'targets' in f:
                tgt_shape = f['targets'].shape
                if len(tgt_shape) == 3:
                    if tgt_shape[1] == len(CURVE_NAMES):
                        self.targets_layout = 'NCT'
                        self.num_timesteps = tgt_shape[2]
                    elif tgt_shape[2] == len(CURVE_NAMES):
                        self.targets_layout = 'NTC'
                        self.num_timesteps = tgt_shape[1]
                    else:
                        self.targets_layout = 'NCT'
                        self.num_timesteps = tgt_shape[2]

            #  targets 
            self.has_scalar_data = 'scalar_targets' in f
            if self.has_scalars and not self.has_scalar_data:
                # :  yield / plant_height_cm / ear_height_cm 
                self.has_scalar_data = all(
                    k in f for k in ['yield', 'plant_height_cm', 'ear_height_cm'])
                self._scalar_from_separate = self.has_scalar_data
            else:
                self._scalar_from_separate = False

        self.preloaded = None
        if preload:
            with h5py.File(h5_path, 'r') as f:
                d = {
                    'genotypes_a': f['genotypes_a'][:].astype(np.int8),
                    'genotypes_b': f['genotypes_b'][:].astype(np.int8),
                    'weather_data': f['weather_data'][:].astype(np.float32),
                }
                if self.has_apsim:
                    d['apsim_params'] = f['apsim_params'][:].astype(np.float32)
                if self.has_curves and 'targets' in f:
                    d['targets'] = f['targets'][:].astype(np.float32)
                if self.has_scalars:
                    if 'scalar_targets' in f:
                        d['scalar_targets'] = f['scalar_targets'][:].astype(np.float32)
                    elif self._scalar_from_separate:
                        y = f['yield'][:].astype(np.float32)
                        h = f['plant_height_cm'][:].astype(np.float32)
                        e = f['ear_height_cm'][:].astype(np.float32)
                        d['scalar_targets'] = np.column_stack([y, h, e])
                self.preloaded = d

    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5_file

    def __del__(self):
        if self._h5_file is not None:
            try: self._h5_file.close()
            except: pass

    def __len__(self):
        return len(self.indices)

    def _to_nct(self, arr):
        """ targets  (4, T)"""
        if self.targets_layout == 'NTC':
            return arr.T  # (T, 4)  (4, T)
        return arr

    def __getitem__(self, idx):
        ri = self.indices[idx]
        s = {}

        if self.preloaded is not None:
            d = self.preloaded
            if self.use_geno:
                s['genotypes_a'] = torch.from_numpy(d['genotypes_a'][ri].astype(np.int64))
                s['genotypes_b'] = torch.from_numpy(d['genotypes_b'][ri].astype(np.int64))
                s['chromosomes'] = self._shared_chromosomes
            else:
                s['genotypes_a'] = torch.zeros(self.num_snps, dtype=torch.long)
                s['genotypes_b'] = torch.zeros(self.num_snps, dtype=torch.long)
                s['chromosomes'] = torch.zeros(self.num_snps, dtype=torch.long)
            s['positions'] = self._shared_positions
            s['weather'] = torch.from_numpy(
                d['weather_data'][ri] if self.use_env
                else np.zeros((self.num_days, self.num_wf), dtype=np.float32))
            if self.has_apsim and 'apsim_params' in d:
                s['apsim_params'] = torch.from_numpy(d['apsim_params'][ri])
            else:
                s['apsim_params'] = torch.zeros(6)
            #  targets
            if self.has_curves and 'targets' in d:
                tgt = self._to_nct(d['targets'][ri])
                s['curve_targets'] = torch.from_numpy(tgt)
            #  targets
            if self.has_scalars and 'scalar_targets' in d:
                s['scalar_targets'] = torch.from_numpy(d['scalar_targets'][ri])
        else:
            f = self._get_h5()
            if self.use_geno:
                s['genotypes_a'] = torch.from_numpy(f['genotypes_a'][ri].astype(np.int64))
                s['genotypes_b'] = torch.from_numpy(f['genotypes_b'][ri].astype(np.int64))
                s['chromosomes'] = self._shared_chromosomes
            else:
                s['genotypes_a'] = torch.zeros(self.num_snps, dtype=torch.long)
                s['genotypes_b'] = torch.zeros(self.num_snps, dtype=torch.long)
                s['chromosomes'] = torch.zeros(self.num_snps, dtype=torch.long)
            s['positions'] = self._shared_positions
            s['weather'] = torch.from_numpy(
                f['weather_data'][ri].astype(np.float32) if self.use_env
                else np.zeros((self.num_days, self.num_wf), dtype=np.float32))
            if self.has_apsim:
                s['apsim_params'] = torch.from_numpy(f['apsim_params'][ri].astype(np.float32))
            else:
                s['apsim_params'] = torch.zeros(6)
            if self.has_curves:
                tgt = self._to_nct(f['targets'][ri].astype(np.float32))
                s['curve_targets'] = torch.from_numpy(tgt)
            if self.has_scalars:
                if 'scalar_targets' in f:
                    s['scalar_targets'] = torch.from_numpy(
                        f['scalar_targets'][ri].astype(np.float32))
                elif self._scalar_from_separate:
                    y = float(f['yield'][ri])
                    h = float(f['plant_height_cm'][ri])
                    e = float(f['ear_height_cm'][ri])
                    s['scalar_targets'] = torch.tensor([y, h, e], dtype=torch.float32)

        return s


def get_dataloaders(h5_path, split='M1', batch_size=32,
                    target_mode='all', num_workers=0, preload=True):
    loaders = []
    for part in ['train', 'val', 'test']:
        ds = MyH5Dataset(h5_path, split, part, target_mode=target_mode,
                         preload=preload)
        loaders.append(DataLoader(
            ds, batch_size=batch_size,
            shuffle=(part == 'train'),
            num_workers=num_workers,
            drop_last=(part == 'train'),
        ))
    return loaders[0], loaders[1], loaders[2]


# 2. 
class MultiCurveHead(nn.Module):
    """ + 4 ,  T """

    def __init__(self, input_dim, num_curves=4, num_timesteps=180,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_curves = num_curves
        self.num_timesteps = num_timesteps

        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.residual_proj = nn.Linear(input_dim, hidden_dim) \
            if input_dim != hidden_dim else nn.Identity()

        self.curve_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_timesteps),
            ) for _ in range(num_curves)
        ])
        for head in self.curve_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        h = self.shared_backbone(x) + self.residual_proj(x)
        preds = [head(h) for head in self.curve_heads]
        return torch.stack(preds, dim=1)  # (B, 4, T)


class ScalarHead(nn.Module):
    """3  (yield, plant_height_cm, ear_height_cm)"""

    def __init__(self, input_dim, num_scalars=3, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scalars),
        )
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)  # (B, num_scalars)


# 3. 
def create_model_config(h5_path, use_genotype=True, use_environment=True):
    with h5py.File(h5_path, 'r') as f:
        num_snps = int(f.attrs['num_snps'])
        num_days = int(f.attrs['num_days'])
        num_wf   = int(f.attrs['num_weather_features'])

    trait_configs = {}  # / Head , 

    seg = min(20, max(5, num_snps // 30))
    config = HyGEMambaConfig(
        snp_embedding_dim=16, num_allele_values=10, num_chromosomes=10,
        mamba_d_model=32, mamba_d_state=8, mamba_num_layers=1,
        num_segments=seg,
        weather_input_features=num_wf, use_hourly_weather=False,
        inter_day_dim=32, max_days=num_days,
        fusion_hidden_dim=64, fusion_output_dim=64,
        fusion_type="gated_cross_attention",
        use_moe=True, moe_hidden_dim=128, num_experts=2, top_k=2,
        num_shared_experts=1,
        use_apsim_head=True, use_curve_decoder=False,
        num_timesteps=num_days,
        trait_configs=trait_configs, dropout=0.1,
    )

    config.use_genotype = use_genotype
    config.use_environment = use_environment
    if not use_genotype:
        config.use_apsim_head = False
    return config


# 4. Per-Task Tracker
class TaskTracker:
    """ early stopping"""

    def __init__(self, task_name, metric_name, mode='max', patience=20):
        self.task_name = task_name
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.counter = 0
        self.is_best = False

    def update(self, value):
        if math.isnan(value):
            self.is_best = False
            self.counter += 1
            return False
        improved = (value > self.best_value) if self.mode == 'max' \
            else (value < self.best_value)
        if improved:
            self.best_value = value
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
            self.is_best = False
        return improved

    @property
    def should_stop(self):
        return self.counter >= self.patience


# 5.  Loss ( + )
class MultiTaskLoss(nn.Module):
    """
    : 

     loss = 0.5*MSE + 0.5*Huber + derivative_weight*
     loss = 0.5*MSE + 0.5*Huber
    """

    def __init__(self, curve_names=None, scalar_names=None,
                 apsim_weight=0.3, moe_weight=0.01, derivative_weight=0.2):
        super().__init__()
        self.curve_names = curve_names or []
        self.scalar_names = scalar_names or []
        self.apsim_weight = apsim_weight
        self.moe_weight = moe_weight
        self.derivative_weight = derivative_weight

    def _curve_loss(self, pred, gt):
        """ loss: pred/gt shape (B, T)"""
        mse = F.mse_loss(pred, gt)
        huber = F.huber_loss(pred, gt, delta=1.0)
        value_loss = 0.5 * mse + 0.5 * huber
        if self.derivative_weight > 0 and pred.shape[1] > 1:
            deriv_loss = F.mse_loss(pred[:, 1:] - pred[:, :-1],
                                     gt[:, 1:] - gt[:, :-1])
        else:
            deriv_loss = torch.tensor(0.0, device=pred.device)
        return value_loss + self.derivative_weight * deriv_loss

    def _scalar_loss(self, pred, gt):
        """ loss: pred/gt shape (B,)"""
        mse = F.mse_loss(pred, gt)
        huber = F.huber_loss(pred, gt, delta=1.0)
        return 0.5 * mse + 0.5 * huber

    def compute_per_task_losses(self, curve_preds, scalar_preds,
                                batch, model_output=None):
        """
        Returns:
            task_losses: {name: loss_tensor}   loss
            loss_dict:   {str: float}  
            aux_loss:     loss (APSIM + MoE)
        """
        device = next(iter(
            [curve_preds.device] if curve_preds is not None
            else [scalar_preds.device]))
        task_losses = {}
        loss_dict = {}
        aux_loss = torch.tensor(0.0, device=device)

        # Target metadata
        if curve_preds is not None and 'curve_targets' in batch:
            targets = batch['curve_targets']  # (B, 4, T)
            for i, name in enumerate(self.curve_names):
                loss = self._curve_loss(curve_preds[:, i, :], targets[:, i, :])
                task_losses[name] = loss
                loss_dict[f'loss_{name}'] = loss.item()

        # Save metrics
        if scalar_preds is not None and 'scalar_targets' in batch:
            targets = batch['scalar_targets']  # (B, 3)
            for i, name in enumerate(self.scalar_names):
                loss = self._scalar_loss(scalar_preds[:, i], targets[:, i])
                task_losses[name] = loss
                loss_dict[f'loss_{name}'] = loss.item()

        # APSIM  loss
        if model_output is not None and model_output.apsim_params is not None \
                and 'apsim_params' in batch:
            apsim_gt = batch['apsim_params']
            if isinstance(model_output.apsim_params, dict):
                apsim_pred = torch.stack(
                    list(model_output.apsim_params.values()), dim=-1)
            else:
                apsim_pred = model_output.apsim_params
            mask = apsim_gt.abs().sum(dim=-1) > 0
            if mask.any():
                g, p = apsim_gt[mask], apsim_pred[mask]
                sc = g.abs().mean(dim=0).clamp(min=1.0)
                apsim_loss = F.mse_loss(p / sc, g / sc)
                aux_loss = aux_loss + self.apsim_weight * apsim_loss
                loss_dict['loss_apsim'] = apsim_loss.item()

        # MoE auxiliary loss
        if model_output is not None and model_output.moe_auxiliary_loss is not None:
            aux_loss = aux_loss + self.moe_weight * model_output.moe_auxiliary_loss
            loss_dict['loss_moe'] = model_output.moe_auxiliary_loss.item()

        return task_losses, loss_dict, aux_loss


# 6. 

class FAMOWeighter:
    """Fast Adaptive Multi-Objective (FAMO) weighting"""

    def __init__(self, task_names, device, w_lr=0.025, gamma=0.01):
        self.task_names = task_names
        self.n = len(task_names)
        self.device = device
        self.w_lr = w_lr
        self.gamma = gamma
        self.log_weights = torch.zeros(self.n, device=device, requires_grad=False)
        self.prev_losses = None

    def get_task_weights(self):
        w = F.softmax(self.log_weights, dim=0)
        return {n: w[i].item() for i, n in enumerate(self.task_names)}

    def compute_weighted_loss(self, task_losses, aux_loss):
        w = F.softmax(self.log_weights, dim=0)
        total = sum(w[i] * task_losses[n] for i, n in enumerate(self.task_names)
                    if n in task_losses)
        total = total + aux_loss
        return total, {f'w_{n}': w[i].item() for i, n in enumerate(self.task_names)}

    def update_weights(self, task_losses):
        curr = torch.tensor(
            [task_losses[n].item() if n in task_losses else 0
             for n in self.task_names],
            device=self.device)
        if self.prev_losses is not None:
            delta = curr - self.prev_losses
            self.log_weights = self.log_weights + self.w_lr * delta
            self.log_weights = self.log_weights - self.gamma * (
                self.log_weights - self.log_weights.mean())
        self.prev_losses = curr.detach()

    def state_dict(self):
        return {'log_weights': self.log_weights.cpu(), 'prev_losses': self.prev_losses}

    def load_state_dict(self, d):
        if 'log_weights' in d:
            self.log_weights = d['log_weights'].to(self.device)


class UncertaintyWeighter(nn.Module):
    """Kendall et al. CVPR 2018  learned uncertainty per task"""

    def __init__(self, task_names, device):
        super().__init__()
        self.task_names = task_names
        self.n = len(task_names)
        self.log_sigma = nn.Parameter(torch.zeros(self.n, device=device))
        self.device = device

    def get_task_weights(self):
        w = torch.exp(-2 * self.log_sigma)
        return {n: w[i].item() for i, n in enumerate(self.task_names)}

    def compute_weighted_loss(self, task_losses, aux_loss):
        total = torch.tensor(0.0, device=self.device)
        info = {}
        for i, n in enumerate(self.task_names):
            if n in task_losses:
                w = torch.exp(-2 * self.log_sigma[i])
                total = total + w * task_losses[n] + self.log_sigma[i]
                info[f'w_{n}'] = w.item()
                info[f'_{n}'] = torch.exp(self.log_sigma[i]).item()
        return total + aux_loss, info

    def update_weights(self, task_losses):
        pass  # updated via gradient


class DWAWeighter:
    """Dynamic Weight Average (Liu et al. CVPR 2019)"""

    def __init__(self, task_names, device, temperature=2.0):
        self.task_names = task_names
        self.n = len(task_names)
        self.device = device
        self.T = temperature
        self.weights = torch.ones(self.n, device=device) / self.n
        self.prev_losses = None

    def get_task_weights(self):
        return {n: self.weights[i].item() for i, n in enumerate(self.task_names)}

    def compute_weighted_loss(self, task_losses, aux_loss):
        total = sum(self.weights[i] * task_losses[n]
                    for i, n in enumerate(self.task_names)
                    if n in task_losses)
        return total + aux_loss, self.get_task_weights()

    def update_epoch(self, epoch_avg_losses):
        curr = torch.tensor(
            [epoch_avg_losses.get(n, 1.0) for n in self.task_names],
            device=self.device)
        if self.prev_losses is not None:
            ratio = curr / self.prev_losses.clamp(min=1e-8)
            self.weights = F.softmax(ratio / self.T, dim=0) * self.n
        self.prev_losses = curr.detach()

    def update_weights(self, task_losses):
        pass

    def state_dict(self):
        return {'weights': self.weights.cpu(), 'prev_losses': self.prev_losses}

    def load_state_dict(self, d):
        if 'weights' in d and d['weights'] is not None:
            self.weights = d['weights'].to(self.device)


# 7. 

def compute_curve_metrics(pred, gt, prefix=''):
    """R, RMSE, MAE, NRMSE for curve data (N, T)"""
    diff = pred - gt
    mse = (diff ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = diff.abs().mean().item()
    gt_range = gt.max().item() - gt.min().item()
    nrmse = rmse / max(gt_range, 1e-6)
    ss_res = (diff ** 2).sum().item()
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return {f'{prefix}_r2': r2, f'{prefix}_rmse': rmse,
            f'{prefix}_mae': mae, f'{prefix}_nrmse': nrmse}


def compute_scalar_metrics(pred, gt, prefix=''):
    """R, RMSE, MAE, NRMSE for scalar data (N,)"""
    diff = pred - gt
    mse = (diff ** 2).mean().item()
    rmse = math.sqrt(mse)
    mae = diff.abs().mean().item()
    gt_range = gt.max().item() - gt.min().item()
    nrmse = rmse / max(gt_range, 1e-6)
    ss_res = (diff ** 2).sum().item()
    ss_tot = ((gt - gt.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return {f'{prefix}_r2': r2, f'{prefix}_rmse': rmse,
            f'{prefix}_mae': mae, f'{prefix}_nrmse': nrmse}


# 8. 
class HyGEMambaTrainer:
    """
    HyGEMamba multi-target trainer.

    target_mode:
      'curve'   4 
      'scalar'  3 
      'all'     7  (3  + 4 )
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_mode = args.target_mode
        self.has_curves  = self.target_mode in ('curve', 'all')
        self.has_scalars = self.target_mode in ('scalar', 'all')

        self.curve_names  = list(CURVE_NAMES) if self.has_curves else []
        self.scalar_names = list(SCALAR_NAMES) if self.has_scalars else []
        self.target_names = self.scalar_names + self.curve_names

        self._setup()

    # Data setup
    def _setup(self):
        args = self.args
        mode_label = {
            'curve': 'four-curve regression',
            'scalar': 'three-scalar regression',
            'all': 'all targets (3 scalar + 4 curve)',
        }[self.target_mode]
        print(f"\n{'='*65}")
        print(f"  HyGEMamba trainer ({mode_label})")
        print(f"  Targets: {', '.join(self.target_names)}")
        print(f"{'='*65}")

        # Dataset information
        with h5py.File(args.h5, 'r') as f:
            print(f"\n  Dataset: {args.h5}")
            print(f"  Samples: {f.attrs['num_samples']}, SNPs: {f.attrs['num_snps']}, "
                  f"days: {f.attrs['num_days']}, weather features: {f.attrs['num_weather_features']}")

            self.num_timesteps = 0
            if self.has_curves:
                tgt_shape = f['targets'].shape
                print(f"  Curve target shape: {tgt_shape}")
                if len(tgt_shape) == 3:
                    if tgt_shape[1] == len(CURVE_NAMES):
                        self.num_timesteps = tgt_shape[2]
                    elif tgt_shape[2] == len(CURVE_NAMES):
                        self.num_timesteps = tgt_shape[1]
                    else:
                        self.num_timesteps = tgt_shape[2]

            if self.has_scalars:
                if 'scalar_targets' in f:
                    print(f"  Scalar target shape: {f['scalar_targets'].shape}")
                else:
                    print("  Scalar targets: loaded from yield/height/ear arrays")

            sg = f[f'splits/{args.split}']
            tr = len(sg['train_indices'])
            va = len(sg['val_indices'])
            te = len(sg['test_indices'])
            print(f"  Split: {args.split} ({sg.attrs['description']})")
            print(f"  train={tr}, val={va}, test={te}")

            self.use_genotype = bool(sg.attrs.get('use_genotype', True))
            self.use_environment = bool(sg.attrs.get('use_environment', True))

        # DataLoaders
        print("\n  Loading data...")
        loaders = get_dataloaders(
            args.h5, split=args.split, batch_size=args.batch_size,
            target_mode=self.target_mode,
            num_workers=args.num_workers, preload=not args.lazy_load,
        )
        self.train_loader, self.val_loader, self.test_loader = loaders
        print(f"  Batches - train: {len(self.train_loader)}, "
              f"val: {len(self.val_loader)}, test: {len(self.test_loader)}")

        # Normalization
        self._compute_norm_stats()

        # Model
        print("\n  Building model (small)...")
        config = create_model_config(
            args.h5,
            use_genotype=self.use_genotype,
            use_environment=self.use_environment,
        )
        self.model = HyGEMamba(config).to(self.device)

        # Output heads
        downstream_dim = config.fusion_output_dim + (6 if config.use_apsim_head else 0)

        self.curve_head = None
        self.scalar_head = None
        head_params = 0

        if self.has_curves:
            curve_hidden = max(128, downstream_dim)
            self.curve_head = MultiCurveHead(
                input_dim=downstream_dim,
                num_curves=len(CURVE_NAMES),
                num_timesteps=self.num_timesteps,
                hidden_dim=curve_hidden,
                dropout=config.dropout,
            ).to(self.device)
            cp = sum(p.numel() for p in self.curve_head.parameters())
            head_params += cp
            print(f"  MultiCurveHead: {cp:,} params")

        if self.has_scalars:
            scalar_hidden = max(128, downstream_dim)
            self.scalar_head = ScalarHead(
                input_dim=downstream_dim,
                num_scalars=len(SCALAR_NAMES),
                hidden_dim=scalar_hidden,
                dropout=config.dropout,
            ).to(self.device)
            sp = sum(p.numel() for p in self.scalar_head.parameters())
            head_params += sp
            print(f"  ScalarHead: {sp:,} params")

        # Parameter statistics
        model_p = sum(p.numel() for p in self.model.parameters())
        total = model_p + head_params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in self._all_head_params() if p.requires_grad)
        print(f"  Total parameters: {total:,} (main model: {model_p:,}, heads: {head_params:,})")
        print(f"  Trainable parameters: {trainable:,}")

        counts = self.model.get_num_parameters()
        for name, cnt in counts.items():
            if name != 'total':
                print(f"    {name:20s}: {cnt:>10,}")

        # Loss function
        self.loss_fn = MultiTaskLoss(
            curve_names=self.curve_names,
            scalar_names=self.scalar_names,
            apsim_weight=args.apsim_weight,
            derivative_weight=getattr(args, 'derivative_weight', 0.2),
        ).to(self.device)

        # Multi-task weighting strategy
        strategy = getattr(args, 'loss_strategy', 'famo')
        print(f"\n  Multi-task weighting strategy: {strategy.upper()}")

        if strategy == 'famo':
            self.weighter = FAMOWeighter(
                task_names=self.target_names, device=self.device,
                w_lr=getattr(args, 'famo_w_lr', 0.025),
                gamma=getattr(args, 'famo_gamma', 0.01),
            )
        elif strategy == 'uncertainty':
            self.weighter = UncertaintyWeighter(
                task_names=self.target_names, device=self.device)
        elif strategy == 'dwa':
            self.weighter = DWAWeighter(
                task_names=self.target_names, device=self.device,
                temperature=getattr(args, 'dwa_temperature', 2.0),
            )
        else:
            raise ValueError(f"Unknown loss_strategy: {strategy}")
        self.loss_strategy = strategy

        # Optimizer
        model_params = list(self.model.parameters()) + list(self._all_head_params())
        if isinstance(self.weighter, UncertaintyWeighter):
            model_params += list(self.weighter.parameters())
        self.optimizer = torch.optim.AdamW(
            model_params, lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )

        # Learning-rate schedule
        total_steps = args.epochs * len(self.train_loader)
        warmup_steps = args.warmup_epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return max(step / max(warmup_steps, 1), 1e-2)
            prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.5 * (1 + math.cos(math.pi * prog)), 1e-3)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Trackers
        self.trackers = {}
        for name in self.target_names:
            self.trackers[name] = TaskTracker(
                name, f'val_{name}_r2', mode='max', patience=args.patience)
        self.primary_task = args.primary_task
        self.best_overall_loss = float('inf')
        self.best_overall_epoch = -1

        # Resume
        self.start_epoch = 0
        if args.resume:
            self._load_checkpoint(args.resume)

        # Save configuration
        cfg = vars(args).copy()
        cfg['total_params'] = total
        cfg['target_mode'] = self.target_mode
        cfg['target_names'] = self.target_names
        if self.has_curves:
            cfg['num_timesteps'] = self.num_timesteps
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=2, default=str)

        print(f"\n  Device: {self.device}")
        print(f"  Target mode: {self.target_mode} ({len(self.target_names)} tasks)")
        print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
        print(f"  Primary task (early stopping): {self.primary_task}")
        print(f"  Output: {self.output_dir}")

    def _all_head_params(self):
        """Return parameters from all output heads."""
        params = []
        if self.curve_head is not None:
            params.extend(self.curve_head.parameters())
        if self.scalar_head is not None:
            params.extend(self.scalar_head.parameters())
        return params

    def _compute_norm_stats(self):
        """"""
        args = self.args
        with h5py.File(args.h5, 'r') as f:
            train_idx = f[f'splits/{args.split}/train_indices'][:]

            # Weather: per-feature z-score
            weather_all = f['weather_data'][:].astype(np.float32)
            weather_train = weather_all[train_idx]
            N, D, Fw = weather_train.shape
            weather_flat = weather_train.reshape(-1, Fw)
            w_mean = weather_flat.mean(axis=0)
            w_std  = np.clip(weather_flat.std(axis=0), 1e-6, None)
            self.weather_mean = torch.tensor(w_mean, dtype=torch.float32).to(self.device)
            self.weather_std  = torch.tensor(w_std,  dtype=torch.float32).to(self.device)

            # Curve targets: per-curve z-score (4,)
            if self.has_curves:
                targets_all = f['targets'][:].astype(np.float32)
                targets_train = targets_all[train_idx]
                tgt_shape = targets_train.shape
                if len(tgt_shape) == 3:
                    if tgt_shape[2] == len(CURVE_NAMES) and tgt_shape[1] != len(CURVE_NAMES):
                        targets_train = targets_train.transpose(0, 2, 1)
                    num_curves = targets_train.shape[1]
                    ct_mean = np.zeros(num_curves, dtype=np.float32)
                    ct_std  = np.zeros(num_curves, dtype=np.float32)
                    for c in range(num_curves):
                        vals = targets_train[:, c, :].flatten()
                        ct_mean[c] = vals.mean()
                        ct_std[c]  = max(vals.std(), 1e-6)
                self.curve_mean = torch.tensor(ct_mean, dtype=torch.float32).to(self.device)
                self.curve_std  = torch.tensor(ct_std,  dtype=torch.float32).to(self.device)
            else:
                self.curve_mean = None
                self.curve_std  = None

            # Scalar targets: per-scalar z-score (3,)
            if self.has_scalars:
                if 'scalar_targets' in f:
                    st_all = f['scalar_targets'][:].astype(np.float32)
                elif all(k in f for k in ['yield', 'plant_height_cm', 'ear_height_cm']):
                    y = f['yield'][:].astype(np.float32)
                    h = f['plant_height_cm'][:].astype(np.float32)
                    e = f['ear_height_cm'][:].astype(np.float32)
                    st_all = np.column_stack([y, h, e])
                else:
                    st_all = np.zeros((len(targets_all), 3), dtype=np.float32)
                st_train = st_all[train_idx]
                st_mean = st_train.mean(axis=0)
                st_std  = np.clip(st_train.std(axis=0), 1e-6, None)
                self.scalar_mean = torch.tensor(st_mean, dtype=torch.float32).to(self.device)
                self.scalar_std  = torch.tensor(st_std,  dtype=torch.float32).to(self.device)
            else:
                self.scalar_mean = None
                self.scalar_std  = None

        print(f"\n  Normalization statistics ({len(train_idx)} training samples):")
        if self.has_curves:
            for i, name in enumerate(CURVE_NAMES):
                print(f"    {name:25s}: mean={self.curve_mean[i].item():.4f}, "
                      f"std={self.curve_std[i].item():.4f}")
        if self.has_scalars:
            for i, name in enumerate(SCALAR_NAMES):
                print(f"    {name:25s}: mean={self.scalar_mean[i].item():.4f}, "
                      f"std={self.scalar_std[i].item():.4f}")

    def _to_device(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        # z-score weather
        if 'weather' in batch:
            batch['weather'] = (batch['weather'] - self.weather_mean) / self.weather_std
        # z-score curve targets: (B, 4, T)
        if 'curve_targets' in batch and self.curve_mean is not None:
            m = self.curve_mean.view(1, -1, 1)
            s = self.curve_std.view(1, -1, 1)
            batch['curve_targets'] = (batch['curve_targets'] - m) / s
        # z-score scalar targets: (B, 3)
        if 'scalar_targets' in batch and self.scalar_mean is not None:
            batch['scalar_targets'] = (
                (batch['scalar_targets'] - self.scalar_mean) / self.scalar_std)
        return batch

    def _denorm_curve(self, preds, gts, curve_index):
        m = self.curve_mean[curve_index].cpu()
        s = self.curve_std[curve_index].cpu()
        return preds * s + m, gts * s + m

    def _denorm_scalar(self, preds, gts, scalar_index):
        m = self.scalar_mean[scalar_index].cpu()
        s = self.scalar_std[scalar_index].cpu()
        return preds * s + m, gts * s + m

    def _forward_all(self, batch):
        """ + """
        output = self.model(
            genotypes_a=batch['genotypes_a'],
            genotypes_b=batch['genotypes_b'],
            chromosomes=batch['chromosomes'],
            positions=batch['positions'],
            weather=batch['weather'],
            return_intermediates=True,
        )
        # downstream representation
        gxe_repr = output.gxe_repr
        if output.apsim_params is not None:
            apsim_vec = torch.stack(list(output.apsim_params.values()), dim=-1)
            downstream_repr = torch.cat([gxe_repr, apsim_vec], dim=-1)
        else:
            downstream_repr = gxe_repr

        curve_preds = self.curve_head(downstream_repr) if self.curve_head else None
        scalar_preds = self.scalar_head(downstream_repr) if self.scalar_head else None

        return output, curve_preds, scalar_preds

    def train_epoch(self, epoch):
        self.model.train()
        if self.curve_head: self.curve_head.train()
        if self.scalar_head: self.scalar_head.train()
        total_loss = 0.0
        loss_accum = {}
        n_batches = len(self.train_loader)

        all_preds = {n: [] for n in self.target_names}
        all_gts   = {n: [] for n in self.target_names}

        pbar = tqdm(self.train_loader, desc=f"  Train Ep{epoch+1}",
                    leave=True, ncols=120,
                    bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}] {postfix}')

        for batch in pbar:
            batch = self._to_device(batch)
            output, curve_preds, scalar_preds = self._forward_all(batch)
            task_losses, ld, aux_loss = self.loss_fn.compute_per_task_losses(
                curve_preds, scalar_preds, batch, output)

            if self.loss_strategy == 'famo' and task_losses:
                self.weighter.update_weights(task_losses)

            weighted_loss, w_info = self.weighter.compute_weighted_loss(
                task_losses, aux_loss)
            ld.update(w_info)

            self.optimizer.zero_grad()
            weighted_loss.backward()
            all_params = list(self.model.parameters()) + list(self._all_head_params())
            torch.nn.utils.clip_grad_norm_(all_params, self.args.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += weighted_loss.item()
            for k, v in ld.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0) + v

            with torch.no_grad():
                if curve_preds is not None and 'curve_targets' in batch:
                    targets = batch['curve_targets']
                    for i, name in enumerate(self.curve_names):
                        all_preds[name].append(curve_preds[:, i, :].cpu())
                        all_gts[name].append(targets[:, i, :].cpu())
                if scalar_preds is not None and 'scalar_targets' in batch:
                    targets = batch['scalar_targets']
                    for i, name in enumerate(self.scalar_names):
                        all_preds[name].append(scalar_preds[:, i].cpu())
                        all_gts[name].append(targets[:, i].cpu())

            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix_str(f"loss={weighted_loss.item():.4f} lr={lr:.1e}")

        pbar.close()
        avg_loss = total_loss / max(n_batches, 1)

        # Epoch  ()
        metrics = {'train_loss': avg_loss}
        metrics['gate_mean'] = getattr(self.model.HGIE, '_last_gate_mean', -1)

        for i, name in enumerate(self.curve_names):
            if all_preds[name]:
                p = torch.cat(all_preds[name])
                g = torch.cat(all_gts[name])
                p, g = self._denorm_curve(p, g, i)
                metrics.update(compute_curve_metrics(p, g, prefix=f'train_{name}'))
        for i, name in enumerate(self.scalar_names):
            if all_preds[name]:
                p = torch.cat(all_preds[name])
                g = torch.cat(all_gts[name])
                p, g = self._denorm_scalar(p, g, i)
                metrics.update(compute_scalar_metrics(p, g, prefix=f'train_{name}'))

        return avg_loss, loss_accum, metrics

    @torch.no_grad()
    def evaluate(self, loader, prefix='val'):
        self.model.eval()
        if self.curve_head: self.curve_head.eval()
        if self.scalar_head: self.scalar_head.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = {n: [] for n in self.target_names}
        all_gts   = {n: [] for n in self.target_names}

        pbar = tqdm(loader, desc=f"  {prefix.capitalize():5s}",
                    leave=True, ncols=120,
                    bar_format='{l_bar}{bar:20}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}] {postfix}')

        for batch in pbar:
            batch = self._to_device(batch)
            output, curve_preds, scalar_preds = self._forward_all(batch)
            task_losses, _, aux_loss = self.loss_fn.compute_per_task_losses(
                curve_preds, scalar_preds, batch, output)
            loss, _ = self.weighter.compute_weighted_loss(task_losses, aux_loss)
            total_loss += loss.item()
            n_batches += 1

            if curve_preds is not None and 'curve_targets' in batch:
                targets = batch['curve_targets']
                for i, name in enumerate(self.curve_names):
                    all_preds[name].append(curve_preds[:, i, :].cpu())
                    all_gts[name].append(targets[:, i, :].cpu())
            if scalar_preds is not None and 'scalar_targets' in batch:
                targets = batch['scalar_targets']
                for i, name in enumerate(self.scalar_names):
                    all_preds[name].append(scalar_preds[:, i].cpu())
                    all_gts[name].append(targets[:, i].cpu())

            pbar.set_postfix_str(f"loss={loss.item():.4f}")
        pbar.close()

        avg_loss = total_loss / max(n_batches, 1)
        metrics = {f'{prefix}_loss': avg_loss}

        for i, name in enumerate(self.curve_names):
            if all_preds[name]:
                p = torch.cat(all_preds[name])
                g = torch.cat(all_gts[name])
                p, g = self._denorm_curve(p, g, i)
                metrics.update(compute_curve_metrics(p, g, prefix=f'{prefix}_{name}'))
        for i, name in enumerate(self.scalar_names):
            if all_preds[name]:
                p = torch.cat(all_preds[name])
                g = torch.cat(all_gts[name])
                p, g = self._denorm_scalar(p, g, i)
                metrics.update(compute_scalar_metrics(p, g, prefix=f'{prefix}_{name}'))

        return metrics

    #  / 
    def _base_state(self, epoch, metrics, task_name=None):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'loss_strategy': self.loss_strategy,
            'weighter_state_dict': (self.weighter.state_dict()
                                    if hasattr(self.weighter, 'state_dict') else {}),
            'metrics': metrics,
            'task': task_name or 'overall',
            'target_mode': self.target_mode,
            'target_names': self.target_names,
            'best_overall_loss': self.best_overall_loss,
            'trackers': {k: {'best': t.best_value, 'counter': t.counter}
                         for k, t in self.trackers.items()},
            'weather_mean': self.weather_mean.cpu(),
            'weather_std': self.weather_std.cpu(),
        }
        if self.curve_head:
            state['curve_head_state_dict'] = self.curve_head.state_dict()
            state['curve_mean'] = self.curve_mean.cpu()
            state['curve_std'] = self.curve_std.cpu()
            state['num_timesteps'] = self.num_timesteps
        if self.scalar_head:
            state['scalar_head_state_dict'] = self.scalar_head.state_dict()
            state['scalar_mean'] = self.scalar_mean.cpu()
            state['scalar_std'] = self.scalar_std.cpu()
        return state

    def _save_task_checkpoint(self, task_name, epoch, metrics):
        state = self._base_state(epoch, metrics, task_name)
        torch.save(state, self.output_dir / f'best_{task_name}.pt')

    def _save_overall_best(self, epoch, metrics):
        torch.save(self._base_state(epoch, metrics),
                   self.output_dir / 'best_overall.pt')

    def _save_latest(self, epoch, metrics):
        torch.save(self._base_state(epoch, metrics),
                   self.output_dir / 'latest.pt')

    def _load_checkpoint(self, path):
        print(f"  : {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if self.curve_head and 'curve_head_state_dict' in ckpt:
            self.curve_head.load_state_dict(ckpt['curve_head_state_dict'])
        if self.scalar_head and 'scalar_head_state_dict' in ckpt:
            self.scalar_head.load_state_dict(ckpt['scalar_head_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt.get('epoch', 0) + 1
        if 'best_overall_loss' in ckpt:
            self.best_overall_loss = ckpt['best_overall_loss']
        if 'trackers' in ckpt:
            for k, v in ckpt['trackers'].items():
                if k in self.trackers:
                    self.trackers[k].best_value = v['best']
                    self.trackers[k].counter = v['counter']
        print(f"   epoch {self.start_epoch} ")

    def _load_task_checkpoint(self, task_name):
        path = self.output_dir / f'best_{task_name}.pt'
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if self.curve_head and 'curve_head_state_dict' in ckpt:
                self.curve_head.load_state_dict(ckpt['curve_head_state_dict'])
            if self.scalar_head and 'scalar_head_state_dict' in ckpt:
                self.scalar_head.load_state_dict(ckpt['scalar_head_state_dict'])
            return True
        return False

    @staticmethod
    def _r2_rating(r2):
        if math.isnan(r2): return '--'
        if r2 >= 0.8:  return 'excellent'
        if r2 >= 0.5:  return 'good'
        if r2 >= 0.2:  return 'fair'
        if r2 >= 0.0:  return 'weak'
        return 'poor'

    def _print_epoch_table(self, train_m, val_m, epoch_time):
        """Print an aligned metric table for one epoch."""
        import unicodedata

        def _dw(s):
            return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
                       for ch in s)

        def _pad(s, width, align='left'):
            pad_n = max(0, width - _dw(s))
            if align == 'right': return ' ' * pad_n + s
            if align == 'center': return ' ' * (pad_n // 2) + s + ' ' * (pad_n - pad_n // 2)
            return s + ' ' * pad_n

        COL1, COL2 = 28, 32

        # Scalar tasks
        if self.has_scalars:
            print("\n  ---- Scalar tasks ---------------------------------------------")
            print(f"  {_pad('Task', COL1)}  {'Train R2':>9s} {'RMSE':>9s} {'NRMSE':>10s}  "
                  f"{'Val R2':>9s} {'RMSE':>9s} {'NRMSE':>10s}  Rating  Status")
            for name in self.scalar_names:
                t = self.trackers[name]
                cn = TASK_CN.get(name, name)
                t_r2 = train_m.get(f'train_{name}_r2', float('nan'))
                t_rmse = train_m.get(f'train_{name}_rmse', float('nan'))
                t_nrmse = train_m.get(f'train_{name}_nrmse', float('nan'))
                v_r2 = val_m.get(f'val_{name}_r2', float('nan'))
                v_rmse = val_m.get(f'val_{name}_rmse', float('nan'))
                v_nrmse = val_m.get(f'val_{name}_nrmse', float('nan'))
                rating = self._r2_rating(v_r2).strip()
                best_str = 'new best' if t.is_best else f'wait {t.counter}/{t.patience}'
                print(f"  {_pad(cn, COL1)}  {t_r2:>9.4f} {t_rmse:>9.3f} {t_nrmse:>10.4f}  "
                      f"{v_r2:>9.4f} {v_rmse:>9.3f} {v_nrmse:>10.4f}  {rating:5s}  {best_str}")

        # Curve tasks
        if self.has_curves:
            print("\n  ---- Curve tasks ----------------------------------------------")
            print(f"  {_pad('Task', COL1)}  {'Train R2':>9s} {'RMSE':>9s} {'NRMSE':>10s}  "
                  f"{'Val R2':>9s} {'RMSE':>9s} {'NRMSE':>10s}  Rating  Status")
            for name in self.curve_names:
                t = self.trackers[name]
                cn = TASK_CN.get(name, name)
                t_r2 = train_m.get(f'train_{name}_r2', float('nan'))
                t_rmse = train_m.get(f'train_{name}_rmse', float('nan'))
                t_nrmse = train_m.get(f'train_{name}_nrmse', float('nan'))
                v_r2 = val_m.get(f'val_{name}_r2', float('nan'))
                v_rmse = val_m.get(f'val_{name}_rmse', float('nan'))
                v_nrmse = val_m.get(f'val_{name}_nrmse', float('nan'))
                rating = self._r2_rating(v_r2).strip()
                best_str = 'new best' if t.is_best else f'wait {t.counter}/{t.patience}'
                print(f"  {_pad(cn, COL1)}  {t_r2:>9.4f} {t_rmse:>9.3f} {t_nrmse:>10.4f}  "
                      f"{v_r2:>9.4f} {v_rmse:>9.3f} {v_nrmse:>10.4f}  {rating:5s}  {best_str}")

        # Total loss
        tl = train_m.get('train_loss', float('nan'))
        vl = val_m.get('val_loss', float('nan'))
        print(f"\n  Total loss: train={tl:.4f}  val={vl:.4f}")

        lr = self.optimizer.param_groups[0]['lr']
        print(f"  Time: {epoch_time:.1f}s  LR={lr:.2e}")

        if hasattr(self, 'weighter'):
            ws = self.weighter.get_task_weights()
            if isinstance(ws, dict):
                parts = [f"{TASK_CN.get(k,k)[:6]}={v if isinstance(v,float) else v:.3f}"
                         for k, v in ws.items()]
                print(f"  {self.loss_strategy.upper()}: {', '.join(parts)}")

    def train(self):
        print(f"\n{'='*65}")
        print("  Start training")
        print(f"{'='*65}")

        history = []
        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            print(f"\n{'-'*65}")
            print(f"  Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'-'*65}")

            train_loss, train_losses, train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, 'val')
            epoch_time = time.time() - t0

            # Per-task tracker
            for name in self.target_names:
                val = val_metrics.get(f'val_{name}_r2', float('nan'))
                improved = self.trackers[name].update(val)
                if improved:
                    self._save_task_checkpoint(name, epoch, val_metrics)

            # Overall best
            val_loss = val_metrics.get('val_loss', float('inf'))
            if val_loss < self.best_overall_loss:
                self.best_overall_loss = val_loss
                self.best_overall_epoch = epoch
                self._save_overall_best(epoch, val_metrics)
                overall_status = 'new best'
            else:
                overall_status = f'(best: Ep{self.best_overall_epoch+1})'

            # DWA epoch update
            if self.loss_strategy == 'dwa':
                epoch_task_losses = {n: train_metrics.get(f'train_{n}_rmse', 1.0)
                                     for n in self.target_names}
                self.weighter.update_epoch(epoch_task_losses)

            self._print_epoch_table(train_metrics, val_metrics, epoch_time)
            print(f"  Best overall: {overall_status}")

            self._save_latest(epoch, val_metrics)
            history.append({'epoch': epoch + 1, 'time': epoch_time,
                            **train_metrics, **val_metrics})

            # Early stopping
            primary_tracker = self.trackers[self.primary_task]
            if primary_tracker.should_stop:
                print(f"\n   Early stopping: '{self.primary_task}' "
                      f"{primary_tracker.counter} epochs without improvement")
                all_stop = all(t.should_stop for t in self.trackers.values())
                if all_stop:
                    print("  All tasks have plateaued; stopping training")
                    break
                elif primary_tracker.counter > primary_tracker.patience + 10:
                    print("  Primary task exceeded patience+10; forcing stop")
                    break
                else:
                    active = [k for k, t in self.trackers.items() if not t.should_stop]
                    print(f"  Active tasks remain: {active}; continuing training")

        print(f"\n{'='*65}")
        print("  Final test-set evaluation")
        print(f"{'='*65}")

        overall_path = self.output_dir / 'best_overall.pt'
        if overall_path.exists():
            ckpt = torch.load(overall_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if self.curve_head and 'curve_head_state_dict' in ckpt:
                self.curve_head.load_state_dict(ckpt['curve_head_state_dict'])
            if self.scalar_head and 'scalar_head_state_dict' in ckpt:
                self.scalar_head.load_state_dict(ckpt['scalar_head_state_dict'])
            test_overall = self.evaluate(self.test_loader, 'test')
            ov_ep = ckpt.get('epoch', 'legacy')
            if isinstance(ov_ep, int): ov_ep += 1

            print(f"\n  Best overall model (Epoch {ov_ep})")
            for name in self.target_names:
                cn = TASK_CN.get(name, name)
                r2 = test_overall.get(f'test_{name}_r2', float('nan'))
                rmse = test_overall.get(f'test_{name}_rmse', float('nan'))
                print(f"    {cn:25s}: R2={r2:.4f}  RMSE={rmse:.4f} {self._r2_rating(r2)}")
        else:
            test_overall = None

        final_results = {}
        for task_name in self.target_names:
            cn = TASK_CN.get(task_name, task_name)
            self._load_task_checkpoint(task_name)
            test_m = self.evaluate(self.test_loader, 'test')
            final_results[task_name] = test_m
            r2 = test_m.get(f'test_{task_name}_r2', float('nan'))
            rmse = test_m.get(f'test_{task_name}_rmse', float('nan'))
            print(f"\n  [{cn}] R2={r2:.4f}  RMSE={rmse:.4f} {self._r2_rating(r2)}")

        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        summary = {}
        if test_overall:
            summary['overall'] = {k: v for k, v in test_overall.items()
                                  if isinstance(v, (int, float)) and not math.isnan(v)}
        for task, m in final_results.items():
            summary[task] = {k: v for k, v in m.items()
                             if isinstance(v, (int, float)) and not math.isnan(v)}
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*65}")
        print(f"  Training complete. Output: {self.output_dir}/")
        print(f"{'='*65}")
        return final_results


# Main entry point
def main():
    p = argparse.ArgumentParser(description='Train HyGEMamba with train.py')

    # Data
    p.add_argument('--h5', required=True, help='Path to the HDF5 dataset')
    p.add_argument('--split', default='M1', choices=['M1', 'M2', 'M3', 'M4', 'M5'])

    # Target mode
    p.add_argument('--target_mode', default='all',
                   choices=['curve', 'scalar', 'all'],
                   help='Training target mode: curve=4 growth curves, scalar=3 scalar traits, all=7 targets')

    # Training
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=5)

    # Task management
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--primary_task', default=None,
                   help='Primary task for early stopping. Defaults: curve=Maize.AboveGround.Wt, scalar/all=yield')
    p.add_argument('--apsim_weight', type=float, default=0.3)
    p.add_argument('--derivative_weight', type=float, default=0.2)

    # Multi-task weighting
    p.add_argument('--loss_strategy', default='famo',
                   choices=['famo', 'uncertainty', 'dwa'])
    p.add_argument('--famo_w_lr', type=float, default=0.025)
    p.add_argument('--famo_gamma', type=float, default=0.01)
    p.add_argument('--dwa_temperature', type=float, default=2.0)

    # Hardware
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--lazy_load', action='store_true')

    # Output
    p.add_argument('--output_dir', default=None)
    p.add_argument('--resume', default=None)

    args = p.parse_args()

    # Default primary task
    if args.primary_task is None:
        if args.target_mode == 'curve':
            args.primary_task = 'Maize.AboveGround.Wt'
        else:
            args.primary_task = 'yield'

    if args.output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'outputs/{args.split}_small_{args.target_mode}_{ts}'

    trainer = HyGEMambaTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()


# Backward-compatible alias for older experiment scripts.
TrainerV3 = HyGEMambaTrainer
