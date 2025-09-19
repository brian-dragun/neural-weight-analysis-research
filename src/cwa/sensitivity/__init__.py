"""Sensitivity analysis components."""

from .basic_sensitivity import compute_basic_gradient_sensitivity
from .registry import register_sensitivity_metric, get_sensitivity_metric, list_sensitivity_metrics
from .security_analyzer import SecurityWeightAnalyzer
from .grad_x_weight import (
    compute_grad_x_weight_sensitivity,
    compute_security_grad_x_weight_sensitivity
)
from .hessian_diag import (
    compute_hessian_diag_sensitivity,
    compute_fault_aware_hessian_sensitivity
)

__all__ = [
    "compute_basic_gradient_sensitivity",
    "register_sensitivity_metric",
    "get_sensitivity_metric",
    "list_sensitivity_metrics",
    "SecurityWeightAnalyzer",
    "compute_grad_x_weight_sensitivity",
    "compute_security_grad_x_weight_sensitivity",
    "compute_hessian_diag_sensitivity",
    "compute_fault_aware_hessian_sensitivity",
]