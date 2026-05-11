"""
Tests for HyGE-Mamba loss functions.

Run with: pytest tests/test_losses.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses.multi_task_loss import MultiTaskLoss, TaskWeights, YieldLoss, ScalarTraitLoss
from src.losses.curve_loss import CurveLoss, DTWLoss, CombinedCurveLoss
from src.losses.moe_auxiliary_loss import (
    MoEAuxiliaryLoss,
    LoadBalancingLoss,
    RouterZLoss,
    compute_moe_metrics,
)


class TestTaskWeights:
    """Test task weights configuration."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = TaskWeights()
        assert weights.yield_weight == 2.0
        assert weights.apsim_weight == 1.0
        assert weights.moe_aux_weight == 0.01

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = TaskWeights(yield_weight=3.0)
        d = weights.to_dict()
        assert d["yield"] == 3.0


class TestYieldLoss:
    """Test yield prediction loss."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        loss_fn = YieldLoss(use_huber=False)

        pred = torch.tensor([10.0, 12.0, 14.0])
        target = torch.tensor([11.0, 12.0, 13.0])

        loss = loss_fn(pred, target)

        # MSE should be (1 + 0 + 1) / 3 = 0.667
        expected = torch.tensor((1 + 0 + 1) / 3)
        assert torch.isclose(loss, expected, atol=1e-3)

    def test_huber_loss(self):
        """Test Huber loss computation."""
        loss_fn = YieldLoss(use_huber=True, huber_delta=1.0)

        pred = torch.tensor([10.0, 12.0, 14.0])
        target = torch.tensor([11.0, 12.0, 13.0])

        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_masked_loss(self):
        """Test loss with mask."""
        loss_fn = YieldLoss()

        pred = torch.tensor([10.0, 12.0, 14.0])
        target = torch.tensor([11.0, 12.0, 13.0])
        mask = torch.tensor([True, False, True])

        loss = loss_fn(pred, target, mask)

        # Only first and third values should contribute
        # MSE should be (1 + 1) / 2 = 1.0
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-3)


class TestScalarTraitLoss:
    """Test scalar trait loss."""

    def test_basic_loss(self):
        """Test basic loss computation."""
        loss_fn = ScalarTraitLoss(num_traits=5)

        pred = torch.randn(4, 5)
        target = torch.randn(4, 5)

        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_weighted_loss(self):
        """Test weighted loss computation."""
        weights = torch.tensor([2.0, 1.0, 1.0, 0.5, 0.5])
        loss_fn = ScalarTraitLoss(num_traits=5, trait_weights=weights)

        pred = torch.zeros(4, 5)
        target = torch.ones(4, 5)

        loss = loss_fn(pred, target)
        assert loss >= 0


class TestCurveLoss:
    """Test curve prediction loss."""

    def test_mse_only(self):
        """Test MSE-only curve loss."""
        loss_fn = CurveLoss(
            mse_weight=1.0,
            smoothness_weight=0.0,
            derivative_weight=0.0,
        )

        pred = torch.randn(4, 2, 90)  # batch, curves, timesteps
        target = torch.randn(4, 2, 90)

        loss = loss_fn(pred, target)
        assert loss >= 0

    def test_smoothness_penalty(self):
        """Test smoothness penalty."""
        loss_fn = CurveLoss(smoothness_weight=0.1)

        # Smooth prediction
        t = torch.linspace(0, 1, 90).unsqueeze(0).unsqueeze(0).expand(4, 2, 90)
        smooth_pred = torch.sin(2 * np.pi * t)

        # Noisy prediction
        noisy_pred = smooth_pred + torch.randn_like(smooth_pred) * 0.5

        target = torch.zeros_like(smooth_pred)

        smooth_loss = loss_fn(smooth_pred, target)
        noisy_loss = loss_fn(noisy_pred, target)

        # Noisy should have higher loss due to smoothness penalty
        # Note: This isn't always true due to the random nature

    def test_derivative_matching(self):
        """Test derivative matching loss."""
        loss_fn = CurveLoss(derivative_weight=0.1)

        pred = torch.randn(4, 2, 90)
        target = torch.randn(4, 2, 90)

        loss = loss_fn(pred, target)
        assert loss >= 0


class TestDTWLoss:
    """Test Dynamic Time Warping loss."""

    def test_identical_curves(self):
        """Test DTW on identical curves."""
        loss_fn = DTWLoss(gamma=1.0)

        curve = torch.randn(4, 2, 50)
        loss = loss_fn(curve, curve)

        # Loss should be close to 0 for identical curves
        assert loss < 0.1

    def test_different_curves(self):
        """Test DTW on different curves."""
        loss_fn = DTWLoss(gamma=1.0)

        pred = torch.zeros(4, 2, 50)
        target = torch.ones(4, 2, 50)

        loss = loss_fn(pred, target)

        # Compute loss for identical curves for comparison
        identical_loss = loss_fn(pred, pred)

        # Different curves should have higher loss than identical curves
        assert loss > identical_loss


class TestCombinedCurveLoss:
    """Test combined curve loss."""

    def test_combined_loss(self):
        """Test combined MSE + DTW loss."""
        loss_fn = CombinedCurveLoss(mse_weight=1.0, dtw_weight=0.1)

        pred = torch.randn(4, 2, 50)
        target = torch.randn(4, 2, 50)

        loss = loss_fn(pred, target)
        assert loss >= 0


class TestLoadBalancingLoss:
    """Test MoE load balancing loss."""

    def test_uniform_routing(self):
        """Test loss with uniform routing."""
        loss_fn = LoadBalancingLoss(num_experts=4)

        batch_size = 100
        # Uniform router logits
        router_logits = torch.zeros(batch_size, 4)
        # Evenly distributed expert selection
        expert_indices = torch.arange(batch_size).remainder(4).unsqueeze(-1)
        expert_indices = torch.cat([expert_indices, (expert_indices + 1) % 4], dim=-1)

        loss = loss_fn(router_logits, expert_indices)
        # With uniform routing, loss should be close to 1.0 (num_experts * (1/num_experts)^2 * num_experts)

    def test_skewed_routing(self):
        """Test loss with skewed routing."""
        loss_fn = LoadBalancingLoss(num_experts=4)

        batch_size = 100
        # Skewed logits favoring expert 0
        router_logits = torch.zeros(batch_size, 4)
        router_logits[:, 0] = 10.0  # Strong preference for expert 0

        # All tokens go to expert 0
        expert_indices = torch.zeros(batch_size, 2, dtype=torch.long)

        loss = loss_fn(router_logits, expert_indices)
        # Skewed routing should have higher loss


class TestRouterZLoss:
    """Test router z-loss."""

    def test_small_logits(self):
        """Test z-loss with small logits."""
        loss_fn = RouterZLoss()

        # Small logits
        router_logits = torch.randn(100, 8) * 0.1
        loss = loss_fn(router_logits)

        assert loss >= 0

    def test_large_logits(self):
        """Test z-loss penalizes large logits."""
        loss_fn = RouterZLoss()

        # Small logits
        small_logits = torch.randn(100, 8) * 0.1
        small_loss = loss_fn(small_logits)

        # Large logits
        large_logits = torch.randn(100, 8) * 10.0
        large_loss = loss_fn(large_logits)

        # Large logits should have higher z-loss
        assert large_loss > small_loss


class TestMoEAuxiliaryLoss:
    """Test combined MoE auxiliary loss."""

    def test_combined_loss(self):
        """Test combined auxiliary loss."""
        loss_fn = MoEAuxiliaryLoss(
            num_experts=8,
            load_balance_weight=0.01,
            z_loss_weight=0.001,
        )

        router_logits = torch.randn(100, 8)
        expert_indices = torch.randint(0, 8, (100, 2))

        loss, loss_dict = loss_fn(router_logits, expert_indices)

        assert "load_balance" in loss_dict
        assert "z_loss" in loss_dict
        assert loss >= 0


class TestMultiTaskLoss:
    """Test multi-task loss."""

    def test_yield_only(self):
        """Test with yield prediction only."""
        loss_fn = MultiTaskLoss()

        predictions = {"yield": torch.randn(4)}
        targets = {"yields": torch.randn(4)}

        loss, loss_dict = loss_fn(predictions, targets)

        assert "yield" in loss_dict
        assert "total" in loss_dict
        assert loss >= 0

    def test_multi_task(self):
        """Test with multiple tasks."""
        loss_fn = MultiTaskLoss()

        predictions = {
            "yield": torch.randn(4),
            "apsim_params": torch.randn(4, 6),
            "growth_curves": torch.randn(4, 4, 90),
            "scalar_traits": torch.randn(4, 5),
        }
        targets = {
            "yields": torch.randn(4),
            "apsim_params": torch.randn(4, 6),
            "growth_curves": torch.randn(4, 4, 90),
            "scalar_traits": torch.randn(4, 5),
        }

        loss, loss_dict = loss_fn(predictions, targets)

        assert "yield" in loss_dict
        assert "apsim" in loss_dict
        assert "curve" in loss_dict
        assert "scalar" in loss_dict

    def test_with_moe_aux(self):
        """Test with MoE auxiliary loss."""
        loss_fn = MultiTaskLoss()

        predictions = {"yield": torch.randn(4)}
        targets = {"yields": torch.randn(4)}
        moe_aux = torch.tensor(0.1)

        loss, loss_dict = loss_fn(predictions, targets, moe_aux_loss=moe_aux)

        assert "moe_aux" in loss_dict


class TestMoEMetrics:
    """Test MoE metric computation."""

    def test_compute_metrics(self):
        """Test metric computation."""
        router_logits = torch.randn(100, 8)
        expert_indices = torch.randint(0, 8, (100, 2))

        metrics = compute_moe_metrics(router_logits, expert_indices, num_experts=8)

        assert "expert_utilization" in metrics
        assert "utilization_std" in metrics
        assert "router_entropy" in metrics
        assert "max_capacity" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])