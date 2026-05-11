"""
State Space Model (SSM) Block for HyGE-Mamba.

Uses the official mamba-ssm CUDA implementation for high-performance
selective state space model computation.

Architecture Overview:
    The SSM block follows the Mamba architecture:
    1. Linear projection to expand input dimension
    2. Depthwise convolution for local context
    3. Selective SSM with input-dependent parameters (CUDA kernel)
    4. Gated output projection

Key Features:
    - Official mamba-ssm CUDA selective scan kernel
    - Support for bidirectional processing
    - Pre-normalization with RMSNorm and residual connection

Reference:
    Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces. https://arxiv.org/abs/2312.00752

Requirements:
    pip install mamba-ssm>=2.0.0 causal-conv1d>=1.4.0
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba as _Mamba
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    _Mamba = None
    MAMBA_SSM_AVAILABLE = False

from .layer_norm import RMSNorm

_FALLBACK_WARNING_EMITTED = False
_CPU_FALLBACK_WARNING_EMITTED = False


class _TorchMambaFallback(nn.Module):
    """Small PyTorch fallback with the same tensor interface as mamba_ssm.Mamba."""

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
    ) -> None:
        super().__init__()
        inner_dim = d_model * expand
        padding = d_conv // 2

        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=d_conv,
            padding=padding,
            groups=inner_dim,
        )
        self.state_proj = nn.Sequential(
            nn.Linear(inner_dim, max(d_state, 1)),
            nn.SiLU(),
            nn.Linear(max(d_state, 1), inner_dim),
        )
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        conv_out = self.conv(value.transpose(1, 2)).transpose(1, 2)
        if conv_out.size(1) != x.size(1):
            conv_out = conv_out[:, :x.size(1), :]
        mixed = self.activation(conv_out + self.state_proj(value))
        return self.out_proj(mixed * torch.sigmoid(gate))


class _MambaWithFallback(nn.Module):
    """Use official Mamba on CUDA tensors and the PyTorch fallback on CPU."""

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand: int = 2,
    ) -> None:
        super().__init__()
        self.cuda_mamba = _Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.cpu_fallback = _TorchMambaFallback(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _CPU_FALLBACK_WARNING_EMITTED

        if x.is_cuda:
            return self.cuda_mamba(x)

        if not _CPU_FALLBACK_WARNING_EMITTED:
            warnings.warn(
                "mamba_ssm is installed but received CPU tensors; using the "
                "pure PyTorch fallback for this forward pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            _CPU_FALLBACK_WARNING_EMITTED = True

        return self.cpu_fallback(x)


def _build_mamba(
        d_model: int,
        d_state: int,
        d_conv: int,
        expand_factor: int,
) -> nn.Module:
    global _FALLBACK_WARNING_EMITTED

    if MAMBA_SSM_AVAILABLE:
        return _MambaWithFallback(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor,
        )

    if not _FALLBACK_WARNING_EMITTED:
        warnings.warn(
            "mamba_ssm is not installed; using a pure PyTorch fallback. "
            "Install hyge-mamba[mamba] for the CUDA Mamba implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FALLBACK_WARNING_EMITTED = True

    return _TorchMambaFallback(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand_factor,
    )


class SSMBlock(nn.Module):
    """
    Full SSM Block with normalization and residual connection.

    Uses the official mamba-ssm Mamba module (CUDA accelerated) as the
    core SSM computation, wrapped with:
    - Pre-normalization (RMSNorm)
    - Residual connection
    - Optional bidirectional processing
    - Dropout

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand_factor: Expansion factor for inner dimension.
        dropout: Dropout probability.
        bidirectional: If True, process sequence in both directions.
        use_glu: If True, use GLU gating (handled internally by Mamba).
        norm_type: Type of normalization ('rmsnorm' or 'layernorm').
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: int = 2,
            dropout: float = 0.1,
            bidirectional: bool = False,
            use_glu: bool = True,
            norm_type: str = "rmsnorm",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.bidirectional = bidirectional

        # Pre-normalization
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        # Forward SSM using the official CUDA-accelerated Mamba implementation.
        # Mamba contains in_proj, conv1d, selective_scan(CUDA), out_proj, and gating.
        # Interface: (batch, seq_len, d_model) -> (batch, seq_len, d_model).
        self.ssm_forward = _build_mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        # Backward SSM (for bidirectional)
        if bidirectional:
            self.ssm_backward = _build_mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
            )
            self.merge = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through SSM block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask (not used in SSM, kept for interface).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        residual = x
        x = self.norm(x)

        if self.bidirectional:
            # Forward direction
            y_forward = self.ssm_forward(x)

            # Backward direction (flip, process, flip back)
            x_backward = torch.flip(x, dims=[1])
            y_backward = self.ssm_backward(x_backward)
            y_backward = torch.flip(y_backward, dims=[1])

            # Merge bidirectional outputs
            y = self.merge(torch.cat([y_forward, y_backward], dim=-1))
        else:
            y = self.ssm_forward(x)

        y = self.dropout(y)

        return residual + y


class MambaBlock(nn.Module):
    """
    Complete Mamba block with SSM and optional FFN.

    This provides a transformer-like block structure using SSM instead
    of attention:
    1. SSM sub-layer with residual
    2. (Optional) FFN sub-layer with residual

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand_factor: Expansion factor for SSM.
        d_ffn: FFN hidden dimension. If None, skip FFN.
        dropout: Dropout probability.
        bidirectional: If True, use bidirectional SSM.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: int = 2,
            d_ffn: Optional[int] = None,
            dropout: float = 0.1,
            bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.ssm_block = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.use_ffn = d_ffn is not None
        if self.use_ffn:
            from .glu_ffn import SwiGLUFFN
            self.ffn_norm = RMSNorm(d_model)
            self.ffn = SwiGLUFFN(d_model, d_ffn, d_model, dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional mask (for interface compatibility).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        x = self.ssm_block(x, mask)

        if self.use_ffn:
            residual = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = residual + x

        return x


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks forming a complete encoder.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand_factor: Expansion factor for SSM.
        num_layers: Number of Mamba blocks.
        dropout: Dropout probability.
        bidirectional: If True, use bidirectional SSM in all blocks.
        final_norm: If True, apply normalization after the last layer.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            d_conv: int = 4,
            expand_factor: int = 2,
            num_layers: int = 4,
            dropout: float = 0.1,
            bidirectional: bool = False,
            final_norm: bool = True,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
                bidirectional=bidirectional,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model) if final_norm else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Mamba encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional mask (for interface compatibility).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        return x
