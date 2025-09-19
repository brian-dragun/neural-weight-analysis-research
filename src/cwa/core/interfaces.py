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
class SecurityAnalysisResult:
    """Security evaluation results for cybersecurity research."""
    attack_success_rate: float
    robustness_score: float
    critical_vulnerabilities: List[Dict[str, Any]]
    defense_effectiveness: Dict[str, float]
    fault_tolerance_metrics: Dict[str, float]
    vulnerability_map: Dict[str, float]  # per-layer vulnerability scores
    attack_surface: Dict[str, Any]       # potential attack vectors


@dataclass
class CriticalWeightAnalysis:
    """Results from critical weight discovery phase."""
    critical_weights: List[tuple]  # (layer_name, param_idx, vulnerability_score)
    vulnerability_map: Dict[str, float]  # per-layer vulnerability
    attack_surface: Dict[str, Any]       # potential attack vectors
    security_ranking: Dict[str, float]   # weights ranked by security criticality
    metadata: Dict[str, Any]


@dataclass
class FaultInjectionResult:
    """Fault injection experiment results."""
    injected_faults: List[Dict[str, Any]]
    performance_degradation: float
    recovery_time: Optional[float]
    critical_failures: List[str]


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


class SecurityAttack(Protocol):
    """Protocol for adversarial attack methods."""

    def execute(
        self,
        model: torch.nn.Module,
        input_data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute adversarial attack."""
        ...


class DefenseMechanism(Protocol):
    """Protocol for security defense strategies."""

    def apply(
        self,
        model: torch.nn.Module,
        **kwargs
    ) -> torch.nn.Module:
        """Apply defense mechanism to model."""
        ...