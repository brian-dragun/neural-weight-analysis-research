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

# Register security-focused metrics
from .security_analyzer import (
    compute_security_gradient_sensitivity,
    compute_vulnerability_scanning_sensitivity
)
from .grad_x_weight import (
    compute_grad_x_weight_sensitivity,
    compute_security_grad_x_weight_sensitivity
)
from .hessian_diag import (
    compute_hessian_diag_sensitivity,
    compute_fault_aware_hessian_sensitivity
)

# Auto-registration happens via decorators in the imported modules