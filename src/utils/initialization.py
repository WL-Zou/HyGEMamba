"""
Weight initialization utilities for HyGE-Mamba.

Provides various initialization schemes for neural network weights,
following best practices for deep learning models.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Union


def init_linear(
        module: nn.Linear,
        method: str = "xavier_uniform",
        gain: float = 1.0,
        std: Optional[float] = None,
) -> None:
    """
    Initialize a linear layer.

    Args:
        module: Linear layer to initialize.
        method: Initialization method ('xavier_uniform', 'xavier_normal',
                'kaiming_uniform', 'kaiming_normal', 'trunc_normal', 'zeros').
        gain: Gain factor for Xavier/Kaiming init.
        std: Standard deviation for truncated normal init.
    """
    if method == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    elif method == "xavier_normal":
        nn.init.xavier_normal_(module.weight, gain=gain)
    elif method == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    elif method == "trunc_normal":
        std = std or 0.02
        nn.init.trunc_normal_(module.weight, std=std)
    elif method == "zeros":
        nn.init.zeros_(module.weight)
    elif method == "orthogonal":
        nn.init.orthogonal_(module.weight, gain=gain)
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_embedding(
        module: nn.Embedding,
        method: str = "normal",
        std: float = 0.02,
) -> None:
    """
    Initialize an embedding layer.

    Args:
        module: Embedding layer to initialize.
        method: Initialization method ('normal', 'trunc_normal', 'uniform').
        std: Standard deviation for normal initialization.
    """
    if method == "normal":
        nn.init.normal_(module.weight, std=std)
    elif method == "trunc_normal":
        nn.init.trunc_normal_(module.weight, std=std)
    elif method == "uniform":
        nn.init.uniform_(module.weight, -std * math.sqrt(3), std * math.sqrt(3))
    else:
        raise ValueError(f"Unknown embedding initialization: {method}")

    if module.padding_idx is not None:
        with torch.no_grad():
            module.weight[module.padding_idx].zero_()


def init_conv1d(
        module: nn.Conv1d,
        method: str = "kaiming_uniform",
) -> None:
    """
    Initialize a 1D convolution layer.

    Args:
        module: Conv1d layer to initialize.
        method: Initialization method.
    """
    if method == "kaiming_uniform":
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    elif method == "kaiming_normal":
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    else:
        raise ValueError(f"Unknown conv initialization: {method}")

    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(module.bias, -bound, bound)


def init_norm(module: Union[nn.LayerNorm, nn.BatchNorm1d]) -> None:
    """
    Initialize a normalization layer.

    Args:
        module: Normalization layer to initialize.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.ones_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


def init_gru(module: nn.GRU, method: str = "orthogonal") -> None:
    """
    Initialize a GRU layer.

    Args:
        module: GRU layer to initialize.
        method: Initialization method for recurrent weights.
    """
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            if method == "orthogonal":
                nn.init.orthogonal_(param)
            else:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            # Set forget gate bias to 1
            n = param.size(0)
            param.data[n // 3: 2 * n // 3].fill_(1.0)


def init_attention(module: nn.MultiheadAttention) -> None:
    """
    Initialize a multi-head attention layer.

    Args:
        module: MultiheadAttention layer to initialize.
    """
    # Query, key, value projections
    if module.in_proj_weight is not None:
        nn.init.xavier_uniform_(module.in_proj_weight)
    else:
        nn.init.xavier_uniform_(module.q_proj_weight)
        nn.init.xavier_uniform_(module.k_proj_weight)
        nn.init.xavier_uniform_(module.v_proj_weight)

    # Output projection
    nn.init.xavier_uniform_(module.out_proj.weight)

    # Biases
    if module.in_proj_bias is not None:
        nn.init.zeros_(module.in_proj_bias)
    nn.init.zeros_(module.out_proj.bias)


def init_weights(
        model: nn.Module,
        linear_method: str = "xavier_uniform",
        embedding_method: str = "normal",
        embedding_std: float = 0.02,
) -> None:
    """
    Initialize all weights in a model.

    Applies appropriate initialization to each layer type.

    Args:
        model: Model to initialize.
        linear_method: Method for linear layers.
        embedding_method: Method for embedding layers.
        embedding_std: Std for embedding initialization.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Special handling for output projections
            if "output" in name or "out_proj" in name:
                init_linear(module, method="zeros")
            else:
                init_linear(module, method=linear_method)

        elif isinstance(module, nn.Embedding):
            init_embedding(module, method=embedding_method, std=embedding_std)

        elif isinstance(module, nn.Conv1d):
            init_conv1d(module)

        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            init_norm(module)

        elif isinstance(module, nn.GRU):
            init_gru(module)

        elif isinstance(module, nn.MultiheadAttention):
            init_attention(module)


def get_parameter_groups(
        model: nn.Module,
        weight_decay: float = 0.01,
        no_decay_keywords: tuple = ("bias", "LayerNorm", "layernorm", "norm", "ln"),
) -> list:
    """
    Get parameter groups with/without weight decay.

    Biases and normalization parameters typically shouldn't have
    weight decay applied.

    Args:
        model: Model to get parameters from.
        weight_decay: Weight decay for other parameters.
        no_decay_keywords: Parameter names that shouldn't have decay.

    Returns:
        List of parameter group dicts for optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: Model to count parameters for.
        trainable_only: If True, only count trainable parameters.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of model parameters."""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"{'=' * 50}")
    print(f"Model Summary")
    print(f"{'=' * 50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"{'=' * 50}")

    # Per-module breakdown
    print("\nPer-module parameters:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,}")