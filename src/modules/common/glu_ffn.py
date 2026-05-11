"""
Gated Linear Unit (GLU) based Feed-Forward Networks.

This module provides various GLU variants commonly used in modern transformer
and state space model architectures:
- GLU: Original Gated Linear Unit
- SwiGLU: GLU with SiLU (Swish) activation
- GeGLU: GLU with GELU activation

Reference:
    Shazeer, N. (2020). GLU Variants Improve Transformer.
    https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GLUFFN(nn.Module):
    """
    Feed-Forward Network with Gated Linear Unit.

    Implements: FFN(x) = (xW  (xW))W
    where  is sigmoid and  is element-wise multiplication.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension (will be applied to both branches).
        output_dim: Output feature dimension. If None, same as input_dim.
        dropout: Dropout probability.
        bias: Whether to include bias in linear layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: Optional[int] = None,
            dropout: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim

        # Gate and value projections
        self.w_gate = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_value = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, self.output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GLU FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch, seq_len, output_dim).
        """
        gate = torch.sigmoid(self.w_gate(x))
        value = self.w_value(x)
        hidden = gate * value
        hidden = self.dropout(hidden)
        output = self.w_out(hidden)
        return output


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU (SiLU-gated Linear Unit).

    Implements: FFN(x) = (SiLU(xW)  xW)W
    where SiLU(x) = x * sigmoid(x) (also known as Swish).

    This is the activation used in LLaMA and other recent models.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension. If None, same as input_dim.
        dropout: Dropout probability.
        bias: Whether to include bias in linear layers.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: Optional[int] = None,
            dropout: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim

        # Gate (with SiLU) and value projections
        # Using a single fused linear for efficiency
        self.w_gate_value = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, self.output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch, seq_len, output_dim).
        """
        gate_value = self.w_gate_value(x)
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        hidden = self.dropout(hidden)
        output = self.w_out(hidden)
        return output


class GeGLUFFN(nn.Module):
    """
    Feed-Forward Network with GeGLU (GELU-gated Linear Unit).

    Implements: FFN(x) = (GELU(xW)  xW)W

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension. If None, same as input_dim.
        dropout: Dropout probability.
        bias: Whether to include bias in linear layers.
        approximate: Whether to use approximate GELU.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: Optional[int] = None,
            dropout: float = 0.0,
            bias: bool = True,
            approximate: str = "none",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.approximate = approximate

        # Gate (with GELU) and value projections
        self.w_gate_value = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, self.output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GeGLU FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch, seq_len, output_dim).
        """
        gate_value = self.w_gate_value(x)
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.gelu(gate, approximate=self.approximate) * value
        hidden = self.dropout(hidden)
        output = self.w_out(hidden)
        return output


class ExpertFFN(nn.Module):
    """
    Expert Feed-Forward Network for Mixture of Experts.

    Supports multiple activation types: relu, gelu, swiglu, glu.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output feature dimension.
        activation: Activation type ('relu', 'gelu', 'swiglu', 'glu').
        dropout: Dropout probability.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: str = "swiglu",
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.activation = activation.lower()

        if self.activation == "swiglu":
            self.ffn = SwiGLUFFN(input_dim, hidden_dim, output_dim, dropout)
        elif self.activation == "glu":
            self.ffn = GLUFFN(input_dim, hidden_dim, output_dim, dropout)
        elif self.activation == "geglu":
            self.ffn = GeGLUFFN(input_dim, hidden_dim, output_dim, dropout)
        elif self.activation == "relu":
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        elif self.activation == "gelu":
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Output tensor of shape (batch, seq_len, output_dim).
        """
        return self.ffn(x)