"""Basic perturbation methods."""

import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_zero_perturbation(
    model: torch.nn.Module,
    target_weights: List[tuple],
    **kwargs
) -> None:
    """Set target weights to zero."""
    logger.info(f"Applying zero perturbation to {len(target_weights)} weights")

    named_params = dict(model.named_parameters())

    for layer_name, param_idx, _ in target_weights:
        if layer_name in named_params:
            param = named_params[layer_name]
            flat_param = param.data.flatten()
            if param_idx < len(flat_param):
                flat_param[param_idx] = 0.0


def apply_noise_perturbation(
    model: torch.nn.Module,
    target_weights: List[tuple],
    scale: float = 0.1,
    **kwargs
) -> None:
    """Add Gaussian noise to target weights."""
    logger.info(f"Applying noise perturbation (scale={scale}) to {len(target_weights)} weights")

    named_params = dict(model.named_parameters())

    for layer_name, param_idx, _ in target_weights:
        if layer_name in named_params:
            param = named_params[layer_name]
            flat_param = param.data.flatten()
            if param_idx < len(flat_param):
                noise = torch.randn(1) * scale
                flat_param[param_idx] += noise.item()