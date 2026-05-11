"""
Curve Loss module for HyGE-Mamba.

This module provides loss functions for time-resolved growth curve
prediction, handling temporal structure and dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CurveLoss(nn.Module):
    """
    Loss function for growth curve prediction.

    Combines multiple loss components:
    1. Point-wise MSE: Standard reconstruction loss
    2. Temporal smoothness: Penalize jerky predictions
    3. Shape matching: Match curve derivatives

    Args:
        mse_weight: Weight for point-wise MSE.
        smoothness_weight: Weight for temporal smoothness.
        derivative_weight: Weight for derivative matching.
        use_masked: If True, handle variable-length curves.
    """

    def __init__(
            self,
            mse_weight: float = 1.0,
            smoothness_weight: float = 0.1,
            derivative_weight: float = 0.1,
            use_masked: bool = False,
    ) -> None:
        super().__init__()

        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.derivative_weight = derivative_weight
        self.use_masked = use_masked

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute curve loss.

        Args:
            pred: Predicted curves, shape (batch, num_curves, timesteps).
            target: Target curves, shape (batch, num_curves, timesteps).
            mask: Optional temporal mask, shape (batch, timesteps).

        Returns:
            Scalar loss value.
        """
        # Point-wise MSE
        mse_loss = F.mse_loss(pred, target, reduction="none")

        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, timesteps)
            mse_loss = (mse_loss * mask).sum() / mask.sum()
        else:
            mse_loss = mse_loss.mean()

        total_loss = self.mse_weight * mse_loss

        # Temporal smoothness (second derivative penalty on predictions)
        if self.smoothness_weight > 0:
            # Second derivative (curvature)
            d2_pred = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
            smoothness_loss = (d2_pred ** 2).mean()
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        # Derivative matching (first derivative)
        if self.derivative_weight > 0:
            d1_pred = pred[:, :, 1:] - pred[:, :, :-1]
            d1_target = target[:, :, 1:] - target[:, :, :-1]
            derivative_loss = F.mse_loss(d1_pred, d1_target)
            total_loss = total_loss + self.derivative_weight * derivative_loss

        return total_loss


class DTWLoss(nn.Module):
    """
    Soft Dynamic Time Warping loss for curve matching.

    DTW allows for temporal alignment flexibility, useful when
    exact timing of growth events may vary.

    This is a differentiable soft-DTW implementation.

    Args:
        gamma: Softmax smoothing parameter (lower = harder DTW).
        normalize: If True, normalize by path length.
    """

    def __init__(
            self,
            gamma: float = 1.0,
            normalize: bool = True,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.normalize = normalize

    def _pairwise_distances(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise squared distances."""
        # x: (batch, n, d), y: (batch, m, d)
        # Returns: (batch, n, m)
        n = x.shape[1]
        m = y.shape[1]

        x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # (batch, n, 1)
        y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # (batch, m, 1)

        dist = x_sq + y_sq.transpose(1, 2) - 2 * torch.bmm(x, y.transpose(1, 2))

        return dist.clamp(min=0)

    def _soft_dtw(
            self,
            distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft-DTW distance.

        Args:
            distances: Pairwise distances, shape (batch, n, m).

        Returns:
            Soft-DTW distance, shape (batch,).
        """
        batch_size, n, m = distances.shape
        device = distances.device

        # Initialize cost matrix
        R = torch.full((batch_size, n + 1, m + 1), float("inf"), device=device)
        R[:, 0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Soft minimum of three possible predecessors
                r0 = R[:, i - 1, j - 1]
                r1 = R[:, i - 1, j]
                r2 = R[:, i, j - 1]

                r_stack = torch.stack([r0, r1, r2], dim=-1)
                soft_min = -self.gamma * torch.logsumexp(-r_stack / self.gamma, dim=-1)

                R[:, i, j] = distances[:, i - 1, j - 1] + soft_min

        dtw_dist = R[:, n, m]

        if self.normalize:
            dtw_dist = dtw_dist / (n + m)

        return dtw_dist

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute soft-DTW loss.

        Args:
            pred: Predicted curves, shape (batch, num_curves, timesteps).
            target: Target curves, shape (batch, num_curves, timesteps).

        Returns:
            Scalar loss value.
        """
        batch_size, num_curves, timesteps = pred.shape

        total_loss = torch.tensor(0.0, device=pred.device)

        # Compute DTW for each curve type
        for c in range(num_curves):
            # Reshape for DTW: (batch, timesteps, 1)
            pred_c = pred[:, c, :].unsqueeze(-1)
            target_c = target[:, c, :].unsqueeze(-1)

            # Compute pairwise distances
            distances = self._pairwise_distances(pred_c, target_c)

            # Compute soft-DTW
            dtw_dist = self._soft_dtw(distances)

            total_loss = total_loss + dtw_dist.mean()

        return total_loss / num_curves


class CombinedCurveLoss(nn.Module):
    """
    Combined loss for curve prediction.

    Combines MSE-based loss with optional DTW for flexible matching.

    Args:
        mse_weight: Weight for MSE-based loss.
        dtw_weight: Weight for DTW loss.
        dtw_gamma: Soft-DTW smoothing parameter.
    """

    def __init__(
            self,
            mse_weight: float = 1.0,
            dtw_weight: float = 0.1,
            dtw_gamma: float = 1.0,
    ) -> None:
        super().__init__()

        self.mse_loss = CurveLoss(
            mse_weight=1.0,
            smoothness_weight=0.1,
            derivative_weight=0.1,
        )
        self.dtw_loss = DTWLoss(gamma=dtw_gamma)

        self.mse_weight = mse_weight
        self.dtw_weight = dtw_weight

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined curve loss.

        Args:
            pred: Predicted curves.
            target: Target curves.
            mask: Optional temporal mask.

        Returns:
            Scalar loss value.
        """
        mse = self.mse_loss(pred, target, mask)

        if self.dtw_weight > 0:
            dtw = self.dtw_loss(pred, target)
            return self.mse_weight * mse + self.dtw_weight * dtw

        return mse


class PhenologyCurveLoss(nn.Module):
    """
    Phenology-aware curve loss.

    Applies different weights to different growth stages, focusing
    more on critical periods like flowering and grain fill.

    Args:
        stage_weights: Weights for different growth stages.
        num_stages: Number of growth stages.
    """

    def __init__(
            self,
            stage_weights: Optional[torch.Tensor] = None,
            num_stages: int = 10,
    ) -> None:
        super().__init__()

        if stage_weights is None:
            # Default: higher weight for flowering (stage 4-5) and grain fill (5-8)
            weights = torch.ones(num_stages)
            weights[4:6] = 2.0  # Flowering
            weights[5:8] = 1.5  # Grain fill
            stage_weights = weights

        self.register_buffer("stage_weights", stage_weights)

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            growth_stages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phenology-weighted loss.

        Args:
            pred: Predicted curves, shape (batch, num_curves, timesteps).
            target: Target curves, shape (batch, num_curves, timesteps).
            growth_stages: Growth stage per timestep, shape (batch, timesteps).

        Returns:
            Scalar loss value.
        """
        batch_size, num_curves, timesteps = pred.shape

        # Get weights for each timestep
        weights = self.stage_weights[growth_stages]  # (batch, timesteps)
        weights = weights.unsqueeze(1)  # (batch, 1, timesteps)

        # Compute weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = (mse * weights).sum() / weights.sum()

        return weighted_mse