"""GPU utilities for neural network analysis."""

import torch
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def get_available_device() -> torch.device:
    """Get the best available device for computation.

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        return device
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device('cpu')


def move_to_device(
    tensor_or_model: Union[torch.Tensor, torch.nn.Module],
    device: Optional[torch.device] = None
) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or model to specified device.

    Args:
        tensor_or_model: Tensor or model to move
        device: Target device, if None uses get_available_device()

    Returns:
        Tensor or model on target device
    """
    if device is None or device == "auto":
        device = get_available_device()

    return tensor_or_model.to(device)


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage information.

    Returns:
        dict: Memory usage information
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
    }


def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")