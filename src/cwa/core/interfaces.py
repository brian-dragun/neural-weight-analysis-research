"""Core interfaces for Critical Weight Analysis tool."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    values: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    metric_name: str
    top_k_weights: List[tuple]  # (layer_name, param_idx, score)


@dataclass
class PerturbationResult:
    """Results from perturbation experiment."""
    baseline_metrics: Dict[str, float]
    perturbed_metrics: Dict[str, float]
    delta_metrics: Dict[str, float]
    perturbation_stats: Dict[str, Any]


@dataclass
class SecurityResult:
    """Basic security analysis results."""
    vulnerability_score: float
    critical_weights: List[tuple]
    recommendations: List[str]


class SensitivityMetric(Protocol):
    """Protocol for sensitivity metrics."""

    def compute(
        self,
        model: torch.nn.Module,
        data_loader: Any,
        **kwargs
    ) -> SensitivityResult:
        """Compute sensitivity scores."""
        ...


class PerturbationMethod(Protocol):
    """Protocol for perturbation methods."""

    def apply(
        self,
        model: torch.nn.Module,
        target_weights: List[tuple],
        **kwargs
    ) -> None:
        """Apply perturbation to model weights."""
        ...