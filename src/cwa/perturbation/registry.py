"""Registry for perturbation methods."""

from typing import Dict, Callable, Any

_perturbation_registry: Dict[str, Callable] = {}


def register_perturbation_method(name: str):
    """Decorator to register perturbation methods."""
    def decorator(func: Callable):
        _perturbation_registry[name] = func
        return func
    return decorator


def get_perturbation_method(name: str) -> Callable:
    """Get perturbation method by name."""
    if name not in _perturbation_registry:
        raise ValueError(f"Unknown perturbation method: {name}")
    return _perturbation_registry[name]


def list_perturbation_methods() -> list[str]:
    """List all available perturbation methods."""
    return list(_perturbation_registry.keys())


# Register the basic methods
from .basic_methods import apply_zero_perturbation, apply_noise_perturbation
register_perturbation_method("zero")(apply_zero_perturbation)
register_perturbation_method("noise")(apply_noise_perturbation)