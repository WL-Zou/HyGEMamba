"""
Router module for Sparse Mixture of Experts (MoE).

This module implements various routing strategies for selecting which
experts process each input token. Key considerations:
- Load balancing: Ensure experts are utilized evenly
- Sparsity: Only route to top-k experts for efficiency
- Differentiability: Allow gradient flow through soft routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Router(nn.Module):
    """
    Base router class for MoE.
    
    Computes routing weights for assigning inputs to experts.
    
    Args:
        input_dim: Input dimension.
        num_experts: Number of experts to route to.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Routing linear layer
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize with small values for stable initial routing
        nn.init.normal_(self.gate.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute routing logits.
        
        Args:
            x: Input tensor, shape (..., input_dim).
        
        Returns:
            Routing logits, shape (..., num_experts).
        """
        return self.gate(x)


class TopKRouter(Router):
    """
    Top-K router that selects the k highest-scoring experts.
    
    Implements sparse routing where only top-k experts are activated
    for each input, enabling efficient inference and training.
    
    Args:
        input_dim: Input dimension.
        num_experts: Number of experts.
        top_k: Number of experts to select for each input.
        normalize: If True, normalize routing weights to sum to 1.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        normalize: bool = True,
    ) -> None:
        super().__init__(input_dim, num_experts)
        
        self.top_k = top_k
        self.normalize = normalize
        
        assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route inputs to top-k experts.
        
        Args:
            x: Input tensor, shape (batch, seq_len, input_dim) or (batch, input_dim).
        
        Returns:
            Tuple of:
                - Routing weights, shape (..., top_k)
                - Expert indices, shape (..., top_k)
                - Full router logits, shape (..., num_experts)
        """
        # Compute router logits
        router_logits = self.gate(x)  # (..., num_experts)
        
        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        
        # Normalize weights to sum to 1
        if self.normalize:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class NoisyTopKRouter(Router):
    """
    Noisy Top-K router with Gaussian noise for exploration.
    
    Adds learnable noise to routing logits during training to encourage
    exploration and prevent expert collapse.
    
    Args:
        input_dim: Input dimension.
        num_experts: Number of experts.
        top_k: Number of experts to select.
        noise_std: Standard deviation of Gaussian noise (if noise_type='gaussian').
        noise_type: Type of noise ('gaussian', 'gumbel', 'none').
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        noise_type: str = "gaussian",
    ) -> None:
        super().__init__(input_dim, num_experts)
        
        self.top_k = top_k
        self.noise_std = noise_std
        self.noise_type = noise_type
        
        # Learnable noise scale
        if noise_type == "gaussian":
            self.noise_gate = nn.Linear(input_dim, num_experts, bias=False)
            nn.init.zeros_(self.noise_gate.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        add_noise: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route inputs to top-k experts with optional noise.
        
        Args:
            x: Input tensor, shape (..., input_dim).
            add_noise: Override training mode for noise. If None, use self.training.
        
        Returns:
            Tuple of (routing_weights, expert_indices, router_logits).
        """
        # Compute router logits
        router_logits = self.gate(x)
        
        # Add noise during training
        if add_noise is None:
            add_noise = self.training
        
        if add_noise and self.noise_type != "none":
            if self.noise_type == "gaussian":
                noise_scale = F.softplus(self.noise_gate(x))
                noise = torch.randn_like(router_logits) * noise_scale * self.noise_std
                router_logits = router_logits + noise
            elif self.noise_type == "gumbel":
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(router_logits) + 1e-10) + 1e-10)
                router_logits = router_logits + gumbel_noise * self.noise_std
        
        # Apply softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return top_k_weights, top_k_indices, router_logits


class ExpertChoiceRouter(nn.Module):
    """
    Expert-choice routing where experts select tokens.
    
    Instead of tokens selecting experts, each expert selects which
    tokens to process. This can lead to better load balancing.
    
    Args:
        input_dim: Input dimension.
        num_experts: Number of experts.
        capacity_factor: How many tokens each expert can process (relative to uniform).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
        # Expert affinity scoring
        self.expert_affinity = nn.Linear(input_dim, num_experts, bias=False)
        nn.init.normal_(self.expert_affinity.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing.
        
        Args:
            x: Input tensor, shape (batch, seq_len, input_dim).
        
        Returns:
            Tuple of:
                - Dispatch mask: (batch, num_experts, capacity, seq_len)
                - Combine weights: (batch, num_experts, capacity)
                - Router logits: (batch, seq_len, num_experts)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute affinity scores
        scores = self.expert_affinity(x)  # (batch, seq_len, num_experts)
        scores = scores.transpose(1, 2)  # (batch, num_experts, seq_len)
        
        # Compute capacity per expert
        capacity = int(seq_len * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)
        
        # Each expert selects top-capacity tokens
        top_scores, top_indices = torch.topk(scores, capacity, dim=-1)
        
        # Create dispatch mask
        dispatch_mask = torch.zeros(
            batch_size, self.num_experts, capacity, seq_len,
            device=x.device, dtype=x.dtype
        )
        
        # Fill dispatch mask
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        expert_idx = torch.arange(self.num_experts, device=x.device).view(1, -1, 1)
        capacity_idx = torch.arange(capacity, device=x.device).view(1, 1, -1)
        
        dispatch_mask[batch_idx, expert_idx, capacity_idx, top_indices] = F.softmax(top_scores, dim=-1)
        
        # Combine weights
        combine_weights = F.softmax(top_scores, dim=-1)
        
        return dispatch_mask, combine_weights, scores.transpose(1, 2)


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Compute load balancing auxiliary loss.
    
    Encourages even distribution of tokens across experts to prevent
    expert collapse and ensure efficient utilization.
    
    Loss = num_experts * sum_i(f_i * P_i)
    where f_i = fraction of tokens assigned to expert i
          P_i = mean routing probability for expert i
    
    Args:
        router_logits: Raw router logits, shape (batch, seq_len, num_experts)
                      or (batch, num_experts).
        expert_indices: Selected expert indices, shape (..., top_k).
        num_experts: Total number of experts.
    
    Returns:
        Load balancing loss scalar.
    """
    # Handle different input shapes
    if router_logits.dim() == 2:
        router_logits = router_logits.unsqueeze(1)
    if expert_indices.dim() == 2:
        expert_indices = expert_indices.unsqueeze(1)
    
    batch_size, seq_len, _ = router_logits.shape
    
    # Compute routing probabilities
    router_probs = F.softmax(router_logits, dim=-1)
    
    # f_i: fraction of tokens routed to each expert
    # Create one-hot encoding of expert assignments
    expert_mask = F.one_hot(expert_indices, num_classes=num_experts).float()
    # Sum over top_k dimension
    expert_mask = expert_mask.sum(dim=-2)
    # Normalize to get fraction
    tokens_per_expert = expert_mask.sum(dim=[0, 1])  # (num_experts,)
    f = tokens_per_expert / (batch_size * seq_len)
    
    # P_i: mean routing probability for each expert
    P = router_probs.mean(dim=[0, 1])  # (num_experts,)
    
    # Load balancing loss
    loss = num_experts * (f * P).sum()
    
    return loss


def compute_router_z_loss(
    router_logits: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute router z-loss for stability.
    
    Encourages router logits to not become too large, which helps
    with training stability.
    
    Loss = mean(log(sum(exp(z))))^2
    
    Args:
        router_logits: Router logits, shape (..., num_experts).
        eps: Small constant for numerical stability.
    
    Returns:
        Z-loss scalar.
    """
    # log-sum-exp of router logits
    log_z = torch.logsumexp(router_logits, dim=-1)
    
    # Squared mean
    z_loss = (log_z ** 2).mean()
    
    return z_loss
