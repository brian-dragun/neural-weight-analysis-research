"""Registry for sensitivity metrics."""

from typing import Dict, Callable, Any
from ..core.interfaces import SensitivityMetric

_sensitivity_registry: Dict[str, Callable] = {}


def register_sensitivity_metric(name: str):
    """Decorator to register sensitivity metrics."""
    def decorator(func: Callable):
        _sensitivity_registry[name] = func
        return func
    return decorator


def get_sensitivity_metric(name: str) -> Callable:
    """Get sensitivity metric by name."""
    if name not in _sensitivity_registry:
        raise ValueError(f"Unknown sensitivity metric: {name}")
    return _sensitivity_registry[name]


def list_sensitivity_metrics() -> list[str]:
    """List all available sensitivity metrics."""
    return list(_sensitivity_registry.keys())


# Register the basic metric
from .basic_sensitivity import compute_basic_gradient_sensitivity
register_sensitivity_metric("basic_gradient")(compute_basic_gradient_sensitivity)