"""
Expert modules for Sparse Mixture of Experts (MoE).

This module implements the expert networks used in MoE layers.
Experts are typically feed-forward networks (FFN) that specialize
in processing different types of inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from ..common.glu_ffn import ExpertFFN, SwiGLUFFN
from ..common.layer_norm import RMSNorm


class Expert(nn.Module):
    """
    Single expert network (feed-forward network).
    
    A standard FFN with configurable activation and hidden dimension.
    
    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden (intermediate) dimension.
        output_dim: Output dimension.
        activation: Activation function ('relu', 'gelu', 'swiglu').
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        activation: str = "swiglu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        output_dim = output_dim or input_dim
        
        if activation == "swiglu":
            self.ffn = SwiGLUFFN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
        else:
            self.ffn = ExpertFFN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert.
        
        Args:
            x: Input tensor, shape (..., input_dim).
        
        Returns:
            Output tensor, shape (..., output_dim).
        """
        return self.ffn(x)


class SharedExpert(nn.Module):
    """
    Shared expert that processes all inputs.
    
    Unlike routed experts, the shared expert sees every input,
    providing a baseline transformation that all inputs receive.
    This helps with training stability and ensures important
    common computations are always applied.
    
    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
        num_shared: Number of shared expert networks.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_shared: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        output_dim = output_dim or input_dim
        self.num_shared = num_shared
        
        # Multiple shared experts
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                activation="swiglu",
                dropout=dropout,
            )
            for _ in range(num_shared)
        ])
        
        # Combine shared expert outputs
        if num_shared > 1:
            self.combine = nn.Linear(output_dim * num_shared, output_dim)
        else:
            self.combine = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared experts.
        
        Args:
            x: Input tensor, shape (..., input_dim).
        
        Returns:
            Combined shared expert output, shape (..., output_dim).
        """
        expert_outputs = [expert(x) for expert in self.experts]
        
        if self.num_shared > 1:
            combined = torch.cat(expert_outputs, dim=-1)
            output = self.combine(combined)
        else:
            output = expert_outputs[0]
        
        return output


class ExpertGroup(nn.Module):
    """
    Group of routed experts.
    
    Contains multiple expert networks and handles batched computation
    for efficiency when experts process different subsets of inputs.
    
    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension for each expert.
        output_dim: Output dimension.
        num_experts: Number of experts in the group.
        activation: Activation function.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_experts: int = 8,
        activation: str = "swiglu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        output_dim = output_dim or input_dim
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Individual expert networks
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through selected experts.
        
        Args:
            x: Input tensor, shape (batch, seq_len, input_dim) or (batch, input_dim).
            expert_weights: Routing weights, shape (..., top_k).
            expert_indices: Selected expert indices, shape (..., top_k).
        
        Returns:
            Weighted expert outputs, shape same as x but with output_dim.
        """
        # Handle 2D input
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(1)
            expert_weights = expert_weights.unsqueeze(1)
            expert_indices = expert_indices.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        
        # Initialize output
        output = torch.zeros(
            batch_size, seq_len, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx)  # (..., top_k)
            
            if not expert_mask.any():
                continue
            
            # Get the weight for tokens assigned to this expert
            expert_weight = torch.where(
                expert_mask,
                expert_weights,
                torch.zeros_like(expert_weights)
            ).sum(dim=-1)  # (batch, seq_len)
            
            # Process all tokens through this expert (masked addition)
            expert_output = self.experts[expert_idx](x)
            output = output + expert_output * expert_weight.unsqueeze(-1)
        
        if is_2d:
            output = output.squeeze(1)
        
        return output
    
    def forward_grouped(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        More efficient grouped forward pass.
        
        Groups tokens by their primary expert assignment and processes
        them in batches. More efficient for large models.
        
        Args:
            x: Input tensor, shape (batch * seq_len, input_dim).
            expert_weights: Routing weights, shape (batch * seq_len, top_k).
            expert_indices: Expert indices, shape (batch * seq_len, top_k).
        
        Returns:
            Weighted expert outputs, shape (batch * seq_len, output_dim).
        """
        num_tokens = x.shape[0]
        top_k = expert_indices.shape[-1]
        
        # Initialize output
        output = torch.zeros(
            num_tokens, self.output_dim,
            device=x.device, dtype=x.dtype
        )
        
        # Process each expert position (k in top_k)
        for k in range(top_k):
            expert_idx_k = expert_indices[:, k]  # (num_tokens,)
            weight_k = expert_weights[:, k]  # (num_tokens,)
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Find tokens for this expert at this k position
                mask = (expert_idx_k == expert_idx)
                
                if not mask.any():
                    continue
                
                # Extract tokens
                tokens = x[mask]  # (num_selected, input_dim)
                
                # Process through expert
                expert_output = self.experts[expert_idx](tokens)
                
                # Weighted addition to output
                weights = weight_k[mask].unsqueeze(-1)
                output[mask] = output[mask] + expert_output * weights
        
        return output


class DropTokens(nn.Module):
    """
    Token dropping for capacity-limited MoE.
    
    When using capacity-limited routing, some tokens may be dropped
    if too many are assigned to the same expert. This module handles
    that logic.
    
    Args:
        capacity_factor: Expert capacity as multiple of uniform distribution.
        drop_policy: How to handle dropped tokens ('zero', 'residual', 'auxiliary').
    """
    
    def __init__(
        self,
        capacity_factor: float = 1.25,
        drop_policy: str = "residual",
    ) -> None:
        super().__init__()
        
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
    
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int,
    ) -> tuple:
        """
        Apply token dropping if needed.
        
        Args:
            x: Input tensor, shape (batch, seq_len, dim).
            expert_indices: Expert assignments, shape (batch, seq_len, top_k).
            num_experts: Number of experts.
        
        Returns:
            Tuple of (processed_x, drop_mask) where drop_mask indicates dropped tokens.
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute capacity per expert
        capacity = int(seq_len * self.capacity_factor / num_experts)
        
        # For simplicity, just return input without dropping
        # Full implementation would track counts and mask overflow
        drop_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        return x, drop_mask
