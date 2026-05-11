"""
Sparse Mixture of Experts (MoE) Layer for HyGE-Mamba.

This module implements the complete sparse MoE layer combining:
- Top-K routing
- Shared + routed experts
- Load balancing auxiliary loss

The MoE layer allows the model to specialize different experts for
different types of GE interactions, such as:
- Drought-responsive genetics
- Heat stress interactions
- Optimal condition performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .router import TopKRouter, NoisyTopKRouter, compute_load_balancing_loss, compute_router_z_loss
from .experts import Expert, SharedExpert, ExpertGroup
from ..common.layer_norm import RMSNorm


@dataclass
class MoEOutput:
    """Container for MoE layer output."""
    
    output: torch.Tensor  # Main output
    router_logits: torch.Tensor  # For auxiliary loss computation
    expert_indices: torch.Tensor  # Selected experts
    expert_weights: torch.Tensor  # Routing weights
    auxiliary_loss: Optional[torch.Tensor] = None  # Load balancing loss


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.
    
    Combines shared experts (always active) with routed experts
    (selectively activated based on input). This architecture provides:
    - Baseline computation via shared experts
    - Specialized computation via routed experts
    - Efficient inference with sparse routing
    
    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for experts.
        output_dim: Output dimension.
        num_experts: Total number of routed experts.
        top_k: Number of experts to route to per input.
        num_shared_experts: Number of shared experts (always active).
        use_noisy_gating: If True, use noisy top-k routing.
        load_balance_coef: Coefficient for load balancing loss.
        z_loss_coef: Coefficient for router z-loss.
        dropout: Dropout probability.
        expert_activation: Activation for expert FFNs.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 2,
        use_noisy_gating: bool = True,
        load_balance_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        dropout: float = 0.1,
        expert_activation: str = "swiglu",
    ) -> None:
        super().__init__()
        
        output_dim = output_dim or input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        
        # Routed experts
        if use_noisy_gating:
            self.router = NoisyTopKRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
                noise_std=0.1,
            )
        else:
            self.router = TopKRouter(
                input_dim=input_dim,
                num_experts=num_experts,
                top_k=top_k,
            )
        
        self.expert_group = ExpertGroup(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            activation=expert_activation,
            dropout=dropout,
        )
        
        # Shared experts
        if num_shared_experts > 0:
            self.shared_experts = SharedExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_shared=num_shared_experts,
                dropout=dropout,
            )
        else:
            self.shared_experts = None
        
        # Combine shared and routed outputs
        if num_shared_experts > 0:
            self.output_gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid(),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary_loss: bool = True,
    ) -> MoEOutput:
        """
        Forward pass through sparse MoE.
        
        Args:
            x: Input tensor, shape (batch, seq_len, input_dim) or (batch, input_dim).
            return_auxiliary_loss: If True, compute and return auxiliary losses.
        
        Returns:
            MoEOutput containing output tensor and optional losses.
        """
        # Route inputs to experts
        expert_weights, expert_indices, router_logits = self.router(x)
        
        # Routed expert output
        routed_output = self.expert_group(x, expert_weights, expert_indices)
        
        # Shared expert output
        if self.shared_experts is not None:
            shared_output = self.shared_experts(x)
            
            # Combine shared and routed outputs
            combined = torch.cat([shared_output, routed_output], dim=-1)
            gate = self.output_gate(combined)
            output = gate * shared_output + (1 - gate) * routed_output
        else:
            output = routed_output
        
        # Compute auxiliary losses
        auxiliary_loss = None
        if return_auxiliary_loss and self.training:
            # Load balancing loss
            lb_loss = compute_load_balancing_loss(
                router_logits, expert_indices, self.num_experts
            )
            
            # Z-loss for stability
            z_loss = compute_router_z_loss(router_logits)
            
            auxiliary_loss = self.load_balance_coef * lb_loss + self.z_loss_coef * z_loss
        
        return MoEOutput(
            output=output,
            router_logits=router_logits,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            auxiliary_loss=auxiliary_loss,
        )


class SparseMoELayer(nn.Module):
    """
    Complete MoE layer with pre-norm and residual connection.
    
    Follows the standard Transformer layer pattern:
        output = x + MoE(Norm(x))
    
    Args:
        input_dim: Input/output dimension.
        hidden_dim: Expert hidden dimension.
        num_experts: Number of routed experts.
        top_k: Number of experts per input.
        num_shared_experts: Number of shared experts.
        dropout: Dropout probability.
        use_noisy_gating: If True, use noisy routing.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 2,
        dropout: float = 0.1,
        use_noisy_gating: bool = True,
    ) -> None:
        super().__init__()
        
        self.norm = RMSNorm(input_dim)
        self.moe = SparseMoE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_shared_experts=num_shared_experts,
            use_noisy_gating=use_noisy_gating,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor.
            return_auxiliary_loss: If True, return auxiliary loss.
        
        Returns:
            Tuple of (output, auxiliary_loss).
        """
        # Pre-norm
        normed = self.norm(x)
        
        # MoE
        moe_output = self.moe(normed, return_auxiliary_loss)
        
        # Residual connection
        output = x + self.dropout(moe_output.output)
        
        return output, moe_output.auxiliary_loss


class MoEStack(nn.Module):
    """
    Stack of MoE layers.
    
    Multiple MoE layers with intermediate processing.
    
    Args:
        input_dim: Input/output dimension.
        hidden_dim: Expert hidden dimension.
        num_layers: Number of MoE layers.
        num_experts: Number of experts per layer.
        top_k: Number of experts per input.
        num_shared_experts: Number of shared experts.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            SparseMoELayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_auxiliary_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE stack.
        
        Args:
            x: Input tensor.
            return_auxiliary_loss: If True, return sum of auxiliary losses.
        
        Returns:
            Tuple of (output, total_auxiliary_loss).
        """
        total_aux_loss = 0.0 if return_auxiliary_loss else None
        
        for layer in self.layers:
            x, aux_loss = layer(x, return_auxiliary_loss)
            if return_auxiliary_loss and aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
        
        x = self.final_norm(x)
        
        return x, total_aux_loss


def get_expert_utilization(
    expert_indices: torch.Tensor,
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute expert utilization statistics.
    
    Useful for monitoring load balancing during training.
    
    Args:
        expert_indices: Selected expert indices, shape (..., top_k).
        num_experts: Total number of experts.
    
    Returns:
        Dictionary containing:
            - counts: Token counts per expert
            - utilization: Fraction of tokens per expert
            - max_utilization: Maximum expert utilization
            - min_utilization: Minimum expert utilization
            - balance_ratio: max/min utilization ratio
    """
    # Flatten expert indices
    flat_indices = expert_indices.flatten()
    
    # Count tokens per expert
    counts = torch.bincount(flat_indices, minlength=num_experts).float()
    total = counts.sum()
    
    # Compute utilization
    utilization = counts / total
    
    return {
        "counts": counts,
        "utilization": utilization,
        "max_utilization": utilization.max(),
        "min_utilization": utilization.min(),
        "balance_ratio": utilization.max() / (utilization.min() + 1e-8),
    }
