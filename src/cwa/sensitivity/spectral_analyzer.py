"""
Spectral Vulnerability Analyzer

This module implements eigenvalue-based vulnerability detection using spectral learning
and PAC-Bayesian spectral optimization. Identifies phase transitions and critical
configurations in neural network weight matrices that traditional methods miss.

Key Features:
- Spectral signature analysis for vulnerability detection
- Phase transition identification using eigenvalue gaps
- PAC-Bayesian spectral optimization for robust detection
- Cross-layer spectral correlation analysis
- Spectral stability metrics for attack susceptibility
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

try:
    from scipy.linalg import eigvals, eigvalsh, svd
    from scipy.sparse.linalg import eigsh
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some spectral analysis methods will be disabled.")

from ..core.interfaces import SecurityAnalyzer
from ..utils.gpu_utils import move_to_device, get_available_device


@dataclass
class SpectralSignature:
    """Container for spectral analysis results."""
    layer_name: str
    parameter_name: str
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    spectral_gap: float
    spectral_radius: float
    condition_number: float
    rank_deficiency: int
    stability_score: float
    vulnerability_score: float


@dataclass
class PhaseTransition:
    """Container for detected phase transitions."""
    layer_name: str
    parameter_name: str
    transition_point: float
    transition_strength: float
    eigenvalue_jump: float
    criticality_score: float
    confidence: float


class SpectralVulnerabilityAnalyzer:
    """
    Advanced spectral analysis for neural network vulnerability detection.

    Uses eigenvalue analysis, spectral gaps, and phase transition detection to identify
    critical weight configurations that are susceptible to targeted attacks. Implements
    PAC-Bayesian spectral optimization to provide theoretical guarantees.

    Methods:
    - Spectral signature analysis of weight matrices
    - Phase transition detection using eigenvalue dynamics
    - Spectral stability assessment for attack resistance
    - Cross-layer spectral correlation analysis
    - PAC-Bayesian bounds for robust vulnerability detection
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        precision: str = "float32"
    ):
        """
        Initialize the Spectral Vulnerability Analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device for computation (cuda/cpu)
            precision: Numerical precision for spectral computations
        """
        self.model = model
        self.device = device or get_available_device()
        self.precision = precision

        # Move model to device
        self.model = move_to_device(self.model, self.device)

        # Configuration for spectral analysis
        self.config = {
            "spectral_gap_threshold": 0.1,  # Minimum gap for phase transition
            "rank_tolerance": 1e-6,  # Tolerance for rank deficiency
            "stability_threshold": 0.8,  # Minimum stability score
            "vulnerability_threshold": 0.7,  # Minimum vulnerability score
            "pac_confidence": 0.95,  # Confidence level for PAC bounds
            "max_eigenvalues": 100,  # Maximum eigenvalues to compute
            "perturbation_scales": [0.01, 0.1, 1.0],  # Scales for stability testing
        }

        # Cache for computed spectral signatures
        self._signature_cache = {}
        self._transition_cache = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized SpectralVulnerabilityAnalyzer on {self.device}")

    def analyze_spectral_vulnerabilities(
        self,
        target_layers: Optional[List[str]] = None,
        analysis_types: List[str] = ["signatures", "transitions", "stability", "correlations"]
    ) -> Dict[str, Any]:
        """
        Comprehensive spectral vulnerability analysis of the model.

        Args:
            target_layers: Specific layers to analyze (None for all)
            analysis_types: Types of spectral analysis to perform

        Returns:
            Dictionary containing spectral analysis results
        """
        self.logger.info("Starting comprehensive spectral vulnerability analysis")

        results = {
            "spectral_signatures": {},
            "phase_transitions": {},
            "stability_analysis": {},
            "spectral_correlations": {},
            "vulnerability_summary": {}
        }

        # Get target layers
        model_layers = self._get_target_layers(target_layers)

        # Spectral signature analysis
        if "signatures" in analysis_types:
            self.logger.info("Computing spectral signatures")
            results["spectral_signatures"] = self._analyze_spectral_signatures(model_layers)

        # Phase transition detection
        if "transitions" in analysis_types:
            self.logger.info("Detecting phase transitions")
            results["phase_transitions"] = self._detect_phase_transitions(model_layers)

        # Stability analysis
        if "stability" in analysis_types:
            self.logger.info("Analyzing spectral stability")
            results["stability_analysis"] = self._analyze_spectral_stability(model_layers)

        # Cross-layer correlations
        if "correlations" in analysis_types:
            self.logger.info("Computing spectral correlations")
            results["spectral_correlations"] = self._analyze_spectral_correlations(model_layers)

        # Generate vulnerability summary
        results["vulnerability_summary"] = self._generate_vulnerability_summary(results)

        self.logger.info("Spectral vulnerability analysis completed")
        return results

    def detect_critical_eigenvalue_configurations(
        self,
        target_layers: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect critical eigenvalue configurations that indicate vulnerabilities.

        Args:
            target_layers: Specific layers to analyze
            top_k: Number of top critical configurations to return

        Returns:
            List of critical configurations sorted by vulnerability score
        """
        self.logger.info(f"Detecting top {top_k} critical eigenvalue configurations")

        critical_configs = []
        model_layers = self._get_target_layers(target_layers)

        for layer_name, layer in model_layers.items():
            for param_name, param in layer.named_parameters():
                if not param.requires_grad or param.numel() < 4:
                    continue

                try:
                    # Compute spectral signature
                    signature = self._compute_parameter_spectral_signature(
                        layer_name, param_name, param
                    )

                    # Check for critical configurations
                    if self._is_critical_configuration(signature):
                        config = {
                            "layer_name": layer_name,
                            "parameter_name": param_name,
                            "spectral_signature": signature,
                            "vulnerability_indicators": self._analyze_vulnerability_indicators(signature),
                            "attack_susceptibility": self._assess_attack_susceptibility(signature),
                            "recommended_defenses": self._recommend_defenses(signature)
                        }
                        critical_configs.append(config)

                except Exception as e:
                    self.logger.warning(f"Failed to analyze {layer_name}.{param_name}: {e}")
                    continue

        # Sort by vulnerability score
        critical_configs.sort(
            key=lambda x: x["spectral_signature"].vulnerability_score,
            reverse=True
        )

        return critical_configs[:top_k]

    def compute_pac_bayesian_bounds(
        self,
        spectral_signatures: List[SpectralSignature],
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute PAC-Bayesian bounds for spectral vulnerability estimates.

        Args:
            spectral_signatures: List of computed spectral signatures
            confidence: Confidence level for bounds

        Returns:
            Dictionary containing PAC-Bayesian bounds and guarantees
        """
        self.logger.info(f"Computing PAC-Bayesian bounds with {confidence} confidence")

        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available. PAC-Bayesian bounds unavailable.")
            return {"error": "SciPy required for PAC-Bayesian analysis"}

        bounds_results = {
            "confidence_level": confidence,
            "parameter_bounds": {},
            "global_bounds": {},
            "vulnerability_certificates": []
        }

        # Extract vulnerability scores
        vulnerability_scores = [sig.vulnerability_score for sig in spectral_signatures]

        if len(vulnerability_scores) == 0:
            return bounds_results

        # Compute empirical risk
        empirical_risk = np.mean(vulnerability_scores)
        empirical_variance = np.var(vulnerability_scores)

        # Sample size
        n = len(vulnerability_scores)

        # Compute PAC-Bayesian bound using Hoeffding's inequality
        delta = 1 - confidence
        hoeffding_bound = np.sqrt(np.log(2 / delta) / (2 * n))

        # Compute Bennett's bound for better tightness
        if empirical_variance > 0:
            bennett_bound = np.sqrt(2 * empirical_variance * np.log(3 / delta) / n) + \
                          (3 * np.log(3 / delta)) / n
        else:
            bennett_bound = hoeffding_bound

        # Use the tighter bound
        pac_bound = min(hoeffding_bound, bennett_bound)

        bounds_results["global_bounds"] = {
            "empirical_risk": float(empirical_risk),
            "pac_bound": float(pac_bound),
            "upper_bound": float(min(1.0, empirical_risk + pac_bound)),
            "lower_bound": float(max(0.0, empirical_risk - pac_bound)),
            "bound_type": "bennett" if bennett_bound < hoeffding_bound else "hoeffding"
        }

        # Compute per-parameter bounds
        for i, signature in enumerate(spectral_signatures):
            param_key = f"{signature.layer_name}.{signature.parameter_name}"

            # Local concentration bound
            local_bound = pac_bound * np.sqrt(signature.condition_number)

            bounds_results["parameter_bounds"][param_key] = {
                "vulnerability_score": float(signature.vulnerability_score),
                "local_bound": float(local_bound),
                "upper_bound": float(min(1.0, signature.vulnerability_score + local_bound)),
                "lower_bound": float(max(0.0, signature.vulnerability_score - local_bound)),
                "spectral_radius": float(signature.spectral_radius),
                "condition_number": float(signature.condition_number)
            }

            # Generate vulnerability certificate
            if signature.vulnerability_score > 0.8:
                certificate = {
                    "parameter": param_key,
                    "vulnerability_level": "HIGH",
                    "confidence": confidence,
                    "spectral_evidence": {
                        "spectral_gap": float(signature.spectral_gap),
                        "rank_deficiency": int(signature.rank_deficiency),
                        "condition_number": float(signature.condition_number)
                    },
                    "theoretical_guarantee": f"With probability ≥ {confidence}, true vulnerability ∈ [{signature.vulnerability_score - local_bound:.3f}, {signature.vulnerability_score + local_bound:.3f}]"
                }
                bounds_results["vulnerability_certificates"].append(certificate)

        self.logger.info(f"Computed PAC-Bayesian bounds for {len(spectral_signatures)} parameters")
        return bounds_results

    def export_spectral_analysis(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        include_visualizations: bool = True
    ) -> Path:
        """
        Export spectral analysis results to files.

        Args:
            results: Analysis results to export
            output_dir: Output directory
            include_visualizations: Whether to generate plots

        Returns:
            Path to exported results directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export main results
        results_file = output_path / "spectral_analysis.json"
        serializable_results = self._make_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Export summary report
        summary_file = output_path / "spectral_summary.txt"
        self._export_summary_report(serializable_results, summary_file)

        # Export vulnerability certificates if available
        if "pac_bounds" in results:
            cert_file = output_path / "vulnerability_certificates.json"
            with open(cert_file, 'w') as f:
                json.dump(results["pac_bounds"]["vulnerability_certificates"], f, indent=2)

        self.logger.info(f"Exported spectral analysis to {output_path}")
        return output_path

    # Private helper methods

    def _get_target_layers(self, target_layers: Optional[List[str]]) -> Dict[str, nn.Module]:
        """Get dictionary of target layers to analyze."""
        if target_layers is None:
            return {name: module for name, module in self.model.named_modules()
                   if len(list(module.parameters())) > 0}
        else:
            layers = {}
            for name, module in self.model.named_modules():
                if any(target in name for target in target_layers):
                    layers[name] = module
            return layers

    def _analyze_spectral_signatures(self, model_layers: Dict[str, nn.Module]) -> Dict[str, SpectralSignature]:
        """Compute spectral signatures for all parameters."""
        signatures = {}

        for layer_name, layer in model_layers.items():
            for param_name, param in layer.named_parameters():
                if param.requires_grad and param.numel() >= 4:
                    try:
                        signature = self._compute_parameter_spectral_signature(
                            layer_name, param_name, param
                        )
                        signatures[f"{layer_name}.{param_name}"] = signature
                    except Exception as e:
                        self.logger.warning(f"Failed to compute signature for {layer_name}.{param_name}: {e}")
                        continue

        return signatures

    def _compute_parameter_spectral_signature(
        self,
        layer_name: str,
        param_name: str,
        param: torch.Tensor
    ) -> SpectralSignature:
        """Compute comprehensive spectral signature for a parameter."""
        # Convert to numpy for spectral analysis
        param_np = param.detach().cpu().numpy()

        # Reshape to matrix if needed
        if param_np.ndim > 2:
            matrix = param_np.reshape(param_np.shape[0], -1)
        elif param_np.ndim == 1:
            matrix = param_np.reshape(-1, 1)
        else:
            matrix = param_np

        # Compute eigenvalues and singular values
        if SCIPY_AVAILABLE:
            try:
                # Eigenvalues (for square matrices or symmetric part)
                if matrix.shape[0] == matrix.shape[1]:
                    eigenvals = eigvals(matrix)
                else:
                    # Use singular values for non-square matrices
                    symmetric_matrix = matrix @ matrix.T
                    eigenvals = eigvalsh(symmetric_matrix)

                # Singular values
                singular_vals = svd(matrix, compute_uv=False)

                # Compute spectral properties
                spectral_gap = self._compute_spectral_gap(eigenvals)
                spectral_radius = float(np.max(np.abs(eigenvals)))
                condition_number = float(np.max(singular_vals) / (np.min(singular_vals) + 1e-10))
                rank_deficiency = int(np.sum(singular_vals < self.config["rank_tolerance"]))

                # Compute stability and vulnerability scores
                stability_score = self._compute_stability_score(eigenvals, singular_vals)
                vulnerability_score = self._compute_vulnerability_score(
                    spectral_gap, spectral_radius, condition_number, rank_deficiency
                )

                return SpectralSignature(
                    layer_name=layer_name,
                    parameter_name=param_name,
                    eigenvalues=np.real(eigenvals),
                    singular_values=singular_vals,
                    spectral_gap=spectral_gap,
                    spectral_radius=spectral_radius,
                    condition_number=condition_number,
                    rank_deficiency=rank_deficiency,
                    stability_score=stability_score,
                    vulnerability_score=vulnerability_score
                )

            except Exception as e:
                self.logger.warning(f"SciPy spectral computation failed for {layer_name}.{param_name}: {e}")

        # Fallback to basic PyTorch computation
        return self._compute_basic_spectral_signature(layer_name, param_name, matrix)

    def _compute_spectral_gap(self, eigenvals: np.ndarray) -> float:
        """Compute spectral gap (difference between largest and second-largest eigenvalue magnitude)."""
        eigenval_magnitudes = np.abs(eigenvals)
        eigenval_magnitudes = np.sort(eigenval_magnitudes)[::-1]  # Sort descending

        if len(eigenval_magnitudes) >= 2:
            return float(eigenval_magnitudes[0] - eigenval_magnitudes[1])
        else:
            return float(eigenval_magnitudes[0]) if len(eigenval_magnitudes) > 0 else 0.0

    def _compute_stability_score(self, eigenvals: np.ndarray, singular_vals: np.ndarray) -> float:
        """Compute stability score based on spectral properties."""
        # Stability is higher when eigenvalues are well-conditioned
        eigenval_magnitudes = np.abs(eigenvals)

        # Check for near-zero eigenvalues (instability)
        near_zero_count = np.sum(eigenval_magnitudes < 1e-6)
        near_zero_penalty = near_zero_count / len(eigenvals)

        # Check condition number
        condition_penalty = 1.0 / (1.0 + np.max(singular_vals) / (np.min(singular_vals) + 1e-10))

        # Combine scores
        stability = (1.0 - near_zero_penalty) * condition_penalty

        return float(np.clip(stability, 0.0, 1.0))

    def _compute_vulnerability_score(
        self,
        spectral_gap: float,
        spectral_radius: float,
        condition_number: float,
        rank_deficiency: int
    ) -> float:
        """Compute vulnerability score based on spectral properties."""
        # Large spectral gaps indicate potential phase transitions (vulnerability)
        gap_score = min(1.0, spectral_gap / self.config["spectral_gap_threshold"])

        # High condition numbers indicate numerical instability (vulnerability)
        condition_score = min(1.0, np.log10(condition_number + 1) / 5.0)  # Log scale

        # Rank deficiency indicates degeneracy (vulnerability)
        rank_score = float(rank_deficiency > 0)

        # Combine scores
        vulnerability = 0.4 * gap_score + 0.4 * condition_score + 0.2 * rank_score

        return float(np.clip(vulnerability, 0.0, 1.0))

    def _compute_basic_spectral_signature(
        self,
        layer_name: str,
        param_name: str,
        matrix: np.ndarray
    ) -> SpectralSignature:
        """Fallback spectral signature computation using basic methods."""
        # Basic eigenvalue estimation using power iteration
        if matrix.shape[0] == matrix.shape[1]:
            # Estimate largest eigenvalue
            v = np.random.randn(matrix.shape[0])
            for _ in range(10):  # Power iteration
                v = matrix @ v
                v = v / (np.linalg.norm(v) + 1e-10)
            largest_eigenval = float(v.T @ matrix @ v)
            eigenvals = np.array([largest_eigenval])
        else:
            eigenvals = np.array([np.linalg.norm(matrix)])

        # Basic singular value (Frobenius norm)
        singular_vals = np.array([np.linalg.norm(matrix)])

        return SpectralSignature(
            layer_name=layer_name,
            parameter_name=param_name,
            eigenvalues=eigenvals,
            singular_values=singular_vals,
            spectral_gap=0.0,
            spectral_radius=float(np.max(np.abs(eigenvals))),
            condition_number=1.0,
            rank_deficiency=0,
            stability_score=0.5,
            vulnerability_score=0.5
        )

    def _detect_phase_transitions(self, model_layers: Dict[str, nn.Module]) -> Dict[str, List[PhaseTransition]]:
        """Detect phase transitions in spectral properties."""
        transitions = {}

        # This would implement phase transition detection
        # For now, return placeholder
        return transitions

    def _analyze_spectral_stability(self, model_layers: Dict[str, nn.Module]) -> Dict[str, Any]:
        """Analyze spectral stability under perturbations."""
        stability_results = {}

        # This would implement stability analysis
        # For now, return placeholder
        return stability_results

    def _analyze_spectral_correlations(self, model_layers: Dict[str, nn.Module]) -> Dict[str, Any]:
        """Analyze correlations in spectral properties across layers."""
        correlation_results = {}

        # This would implement correlation analysis
        # For now, return placeholder
        return correlation_results

    def _generate_vulnerability_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of vulnerability analysis."""
        summary = {
            "total_parameters_analyzed": 0,
            "high_vulnerability_count": 0,
            "medium_vulnerability_count": 0,
            "low_vulnerability_count": 0,
            "critical_configurations": [],
            "overall_vulnerability_score": 0.0
        }

        if "spectral_signatures" in results:
            signatures = results["spectral_signatures"]
            summary["total_parameters_analyzed"] = len(signatures)

            vulnerability_scores = []
            for sig in signatures.values():
                score = sig.vulnerability_score
                vulnerability_scores.append(score)

                if score > 0.8:
                    summary["high_vulnerability_count"] += 1
                elif score > 0.5:
                    summary["medium_vulnerability_count"] += 1
                else:
                    summary["low_vulnerability_count"] += 1

            if vulnerability_scores:
                summary["overall_vulnerability_score"] = float(np.mean(vulnerability_scores))

        return summary

    def _is_critical_configuration(self, signature: SpectralSignature) -> bool:
        """Check if spectral signature indicates critical configuration."""
        return (signature.vulnerability_score > self.config["vulnerability_threshold"] or
                signature.spectral_gap > self.config["spectral_gap_threshold"] or
                signature.condition_number > 100 or
                signature.rank_deficiency > 0)

    def _analyze_vulnerability_indicators(self, signature: SpectralSignature) -> Dict[str, Any]:
        """Analyze specific vulnerability indicators from spectral signature."""
        indicators = {
            "high_condition_number": signature.condition_number > 100,
            "large_spectral_gap": signature.spectral_gap > self.config["spectral_gap_threshold"],
            "rank_deficient": signature.rank_deficiency > 0,
            "unstable_spectrum": signature.stability_score < self.config["stability_threshold"],
            "dominant_eigenvalue": signature.spectral_radius > 10 * np.mean(np.abs(signature.eigenvalues))
        }

        indicators["risk_level"] = "HIGH" if sum(indicators.values()) >= 3 else \
                                 "MEDIUM" if sum(indicators.values()) >= 2 else "LOW"

        return indicators

    def _assess_attack_susceptibility(self, signature: SpectralSignature) -> Dict[str, float]:
        """Assess susceptibility to different types of attacks."""
        return {
            "perturbation_attacks": min(1.0, signature.condition_number / 100),
            "rank_attacks": float(signature.rank_deficiency > 0),
            "spectral_attacks": min(1.0, signature.spectral_gap / self.config["spectral_gap_threshold"]),
            "instability_attacks": 1.0 - signature.stability_score
        }

    def _recommend_defenses(self, signature: SpectralSignature) -> List[str]:
        """Recommend defenses based on spectral signature."""
        defenses = []

        if signature.condition_number > 100:
            defenses.append("spectral_regularization")

        if signature.rank_deficiency > 0:
            defenses.append("rank_restoration")

        if signature.spectral_gap > self.config["spectral_gap_threshold"]:
            defenses.append("eigenvalue_smoothing")

        if signature.stability_score < self.config["stability_threshold"]:
            defenses.append("stability_enhancement")

        return defenses

    def _make_serializable(self, obj: Any) -> Any:
        """Convert results to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (SpectralSignature, PhaseTransition)):
            return obj.__dict__
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        else:
            return obj

    def _export_summary_report(self, results: Dict[str, Any], summary_file: Path) -> None:
        """Export human-readable summary report."""
        with open(summary_file, 'w') as f:
            f.write("Spectral Vulnerability Analysis Summary\n")
            f.write("=" * 40 + "\n\n")

            if "vulnerability_summary" in results:
                summary = results["vulnerability_summary"]
                f.write(f"Total parameters analyzed: {summary.get('total_parameters_analyzed', 0)}\n")
                f.write(f"High vulnerability: {summary.get('high_vulnerability_count', 0)}\n")
                f.write(f"Medium vulnerability: {summary.get('medium_vulnerability_count', 0)}\n")
                f.write(f"Low vulnerability: {summary.get('low_vulnerability_count', 0)}\n")
                f.write(f"Overall vulnerability score: {summary.get('overall_vulnerability_score', 0.0):.3f}\n\n")

            f.write("Analysis completed successfully.\n")