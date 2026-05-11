"""
MoE Auxiliary Loss module for HyGE-Mamba.

This module provides auxiliary loss functions for training sparse
Mixture of Experts layers, including load balancing and router
regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss for MoE training.

    Encourages uniform distribution of tokens across experts to
    prevent expert collapse and ensure efficient utilization.

    Loss = num_experts * sum_i(f_i * P_i)
    where:
        f_i = fraction of tokens routed to expert i
        P_i = average routing probability to expert i

    Args:
        num_experts: Number of experts in the MoE layer.
    """

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts

    def forward(
            self,
            router_logits: torch.Tensor,
            expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss.

        Args:
            router_logits: Raw router logits, shape (..., num_experts).
            expert_indices: Selected expert indices, shape (..., top_k).

        Returns:
            Scalar load balancing loss.
        """
        # Flatten batch dimensions
        router_logits = router_logits.reshape(-1, self.num_experts)
        expert_indices = expert_indices.reshape(-1, expert_indices.shape[-1])

        num_tokens = router_logits.shape[0]
        top_k = expert_indices.shape[-1]

        # Compute fraction of tokens going to each expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # Sum across top_k
        f_i = expert_mask.sum(dim=0) / (num_tokens * top_k)

        # Compute average routing probability
        router_probs = F.softmax(router_logits, dim=-1)
        P_i = router_probs.mean(dim=0)

        # Load balancing loss
        loss = self.num_experts * (f_i * P_i).sum()

        return loss


class RouterZLoss(nn.Module):
    """
    Router z-loss for training stability.

    Penalizes large router logits to prevent numerical instability
    and encourage exploration.

    Loss = mean(log(sum_i(exp(z_i)))^2)

    This encourages the router to maintain moderate logit magnitudes.
    """

    def forward(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router z-loss.

        Args:
            router_logits: Raw router logits, shape (..., num_experts).

        Returns:
            Scalar z-loss.
        """
        # Flatten batch dimensions
        logits_flat = router_logits.reshape(-1, router_logits.shape[-1])

        # Compute log-sum-exp
        log_z = torch.logsumexp(logits_flat, dim=-1)

        # Z-loss: mean squared log-sum-exp
        z_loss = (log_z ** 2).mean()

        return z_loss


class ExpertDiversityLoss(nn.Module):
    """
    Expert diversity loss.

    Encourages experts to learn different representations by
    penalizing similarity between expert outputs.

    Args:
        temperature: Temperature for similarity computation.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            expert_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expert diversity loss.

        Args:
            expert_outputs: Outputs from each expert,
                           shape (batch, num_experts, dim).

        Returns:
            Scalar diversity loss.
        """
        batch_size, num_experts, dim = expert_outputs.shape

        # Normalize outputs
        outputs_norm = F.normalize(expert_outputs, dim=-1)

        # Compute pairwise similarity
        similarity = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))
        similarity = similarity / self.temperature

        # Mask diagonal (self-similarity)
        mask = torch.eye(num_experts, device=expert_outputs.device).bool()
        similarity = similarity.masked_fill(mask.unsqueeze(0), 0)

        # Loss is mean similarity (want to minimize)
        diversity_loss = similarity.abs().mean()

        return diversity_loss


class MoEAuxiliaryLoss(nn.Module):
    """
    Combined MoE auxiliary loss.

    Combines multiple auxiliary losses for training sparse MoE:
    1. Load balancing: Uniform expert utilization
    2. Router z-loss: Training stability
    3. Expert diversity: Different expert specializations

    Args:
        num_experts: Number of experts.
        load_balance_weight: Weight for load balancing loss.
        z_loss_weight: Weight for router z-loss.
        diversity_weight: Weight for expert diversity loss.
    """

    def __init__(
            self,
            num_experts: int,
            load_balance_weight: float = 0.01,
            z_loss_weight: float = 0.001,
            diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()

        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight
        self.diversity_weight = diversity_weight

        self.load_balance_loss = LoadBalancingLoss(num_experts)
        self.z_loss = RouterZLoss()

        if diversity_weight > 0:
            self.diversity_loss = ExpertDiversityLoss()
        else:
            self.diversity_loss = None

    def forward(
            self,
            router_logits: torch.Tensor,
            expert_indices: torch.Tensor,
            expert_outputs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined MoE auxiliary loss.

        Args:
            router_logits: Raw router logits, shape (..., num_experts).
            expert_indices: Selected expert indices, shape (..., top_k).
            expert_outputs: Optional expert outputs for diversity loss,
                           shape (batch, num_experts, dim).

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=router_logits.device)

        # Load balancing loss
        if self.load_balance_weight > 0:
            lb_loss = self.load_balance_loss(router_logits, expert_indices)
            loss_dict["load_balance"] = lb_loss
            total_loss = total_loss + self.load_balance_weight * lb_loss

        # Router z-loss
        if self.z_loss_weight > 0:
            z_loss = self.z_loss(router_logits)
            loss_dict["z_loss"] = z_loss
            total_loss = total_loss + self.z_loss_weight * z_loss

        # Expert diversity loss
        if self.diversity_weight > 0 and expert_outputs is not None:
            div_loss = self.diversity_loss(expert_outputs)
            loss_dict["diversity"] = div_loss
            total_loss = total_loss + self.diversity_weight * div_loss

        return total_loss, loss_dict


def compute_moe_metrics(
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int,
) -> dict:
    """
    Compute MoE routing metrics for monitoring.

    Args:
        router_logits: Raw router logits.
        expert_indices: Selected expert indices.
        num_experts: Number of experts.

    Returns:
        Dictionary of metrics.
    """
    # Flatten
    router_logits = router_logits.reshape(-1, num_experts)
    expert_indices = expert_indices.reshape(-1, expert_indices.shape[-1])

    num_tokens = router_logits.shape[0]

    # Expert utilization
    expert_counts = torch.zeros(num_experts, device=router_logits.device)
    for k in range(expert_indices.shape[-1]):
        expert_counts.scatter_add_(
            0,
            expert_indices[:, k],
            torch.ones(num_tokens, device=router_logits.device),
        )
    utilization = expert_counts / expert_counts.sum()

    # Router entropy (higher = more uniform)
    router_probs = F.softmax(router_logits, dim=-1)
    entropy = -(router_probs * router_probs.log().clamp(min=-100)).sum(dim=-1).mean()

    # Expert capacity (fraction of tokens each expert receives)
    capacity = expert_counts / num_tokens

    return {
        "expert_utilization": utilization,
        "utilization_std": utilization.std().item(),
        "router_entropy": entropy.item(),
        "max_capacity": capacity.max().item(),
        "min_capacity": capacity.min().item(),
    }