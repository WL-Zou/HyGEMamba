"""
Normalization layers for HyGE-Mamba.

This module provides RMSNorm and LayerNorm implementations optimized for
sequence modeling tasks.
"""

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm simplifies LayerNorm by removing the mean centering operation,
    which can be more efficient while maintaining similar performance.

    Reference:
        Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
        https://arxiv.org/abs/1910.07467

    Args:
        dim: The dimension to normalize over (typically the feature dimension).
        eps: A small constant for numerical stability.
        elementwise_affine: If True, apply learnable affine parameters.
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.

    This is a wrapper around PyTorch's LayerNorm that allows for
    optional bias (useful for certain architectures).

    Args:
        dim: The dimension to normalize over.
        eps: A small constant for numerical stability.
        bias: If True, include learnable bias parameters.
    """

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        return nn.functional.layer_norm(
            x,
            (self.dim,),
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"


class GroupRMSNorm(nn.Module):
    """
    Group-wise RMS Normalization.

    Normalizes features within groups, useful for multi-head attention
    or grouped convolutions.

    Args:
        dim: Total feature dimension.
        num_groups: Number of groups to normalize separately.
        eps: A small constant for numerical stability.
    """

    def __init__(
            self,
            dim: int,
            num_groups: int,
            eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_groups == 0, f"dim ({dim}) must be divisible by num_groups ({num_groups})"

        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group-wise RMS normalization.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor of the same shape.
        """
        shape = x.shape
        x = x.view(*shape[:-1], self.num_groups, self.group_dim)

        # RMS norm within each group
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        x = x.view(*shape)
        return x * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_groups={self.num_groups}, eps={self.eps}"