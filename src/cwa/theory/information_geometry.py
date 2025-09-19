"""
Information-Theoretic Weight Criticality Analysis

This module implements advanced information-theoretic methods for identifying
critical weights in neural networks. Goes beyond gradient-based approaches by
using Fisher information metrics, mutual information, and phase transition
detection to understand weight criticality from a theoretical perspective.

Key innovations:
- Fisher Information Matrix for vulnerability analysis
- Mutual Information estimation for weight dependencies
- Phase transition detection in weight space
- Information bottleneck analysis for compression vulnerabilities
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
    from scipy.linalg import eigvals, svd
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some information-theoretic methods will be disabled.")

from ..core.interfaces import WeightAnalyzer, CriticalWeight
from ..utils.gpu_utils import move_to_device, get_available_device


@dataclass
class InformationMetrics:
    """Container for information-theoretic metrics."""
    fisher_information: float
    mutual_information: float
    entropy: float
    phase_transition_score: float
    bottleneck_score: float
    spectral_gap: float


@dataclass
class WeightInformationProfile:
    """Comprehensive information-theoretic profile of a weight."""
    layer_idx: int
    parameter_name: str
    coordinates: Tuple[int, ...]
    metrics: InformationMetrics
    criticality_score: float
    confidence: float


class InformationGeometricAnalyzer:
    """
    Advanced information-theoretic analysis of weight criticality.

    This analyzer uses Fisher information metrics, mutual information estimation,
    and spectral analysis to identify critical configurations that traditional
    gradient-based methods might miss. Particularly effective for understanding
    phase transitions and information bottlenecks in neural networks.

    Methods implemented:
    - Fisher Information Matrix computation for vulnerability analysis
    - InfoNet-inspired mutual information estimation
    - Phase transition detection using spectral signatures
    - Information bottleneck analysis for compression vulnerabilities
    - Cross-layer information flow analysis
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        precision: str = "float32"
    ):
        """
        Initialize the Information-Theoretic Weight Analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device for computation (cuda/cpu)
            precision: Numerical precision for computations
        """
        self.model = model
        self.device = device or get_available_device()
        self.precision = precision

        # Move model to device
        self.model = move_to_device(self.model, self.device)

        # Configuration
        self.config = {
            "fisher_samples": 1000,  # Samples for Fisher information estimation
            "mi_estimation_method": "InfoNet",  # Method for MI estimation
            "phase_transition_threshold": 0.1,  # Threshold for phase detection
            "spectral_window_size": 10,  # Window for spectral analysis
            "bottleneck_compression_ratio": 0.8,  # Information bottleneck threshold
        }

        # Cache for computed metrics
        self._fisher_cache = {}
        self._mi_cache = {}
        self._spectral_cache = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized InformationGeometricAnalyzer on {self.device}")

    def analyze_weight_criticality(
        self,
        inputs: torch.Tensor,
        target_layers: Optional[List[str]] = None,
        analysis_methods: List[str] = ["fisher", "mutual_info", "phase_transitions", "bottleneck"]
    ) -> Dict[str, List[WeightInformationProfile]]:
        """
        Comprehensive information-theoretic analysis of weight criticality.

        Args:
            inputs: Input tensors for analysis
            target_layers: Specific layers to analyze (None for all)
            analysis_methods: Methods to use for analysis

        Returns:
            Dictionary mapping layer names to weight information profiles
        """
        self.logger.info("Starting comprehensive information-theoretic weight analysis")

        results = {}
        model_layers = self._get_target_layers(target_layers)

        # Move inputs to device
        inputs = move_to_device(inputs, self.device)

        for layer_name, layer in model_layers.items():
            self.logger.info(f"Analyzing layer: {layer_name}")

            try:
                layer_profiles = []

                # Get all parameters in this layer
                for param_name, param in layer.named_parameters():
                    if param.requires_grad and param.numel() > 1:
                        profile = self._analyze_parameter_information(
                            layer_name, param_name, param, inputs, analysis_methods
                        )
                        if profile:
                            layer_profiles.append(profile)

                if layer_profiles:
                    # Sort by criticality score
                    layer_profiles.sort(key=lambda x: x.criticality_score, reverse=True)
                    results[layer_name] = layer_profiles

            except Exception as e:
                self.logger.warning(f"Failed to analyze layer {layer_name}: {e}")
                continue

        self.logger.info(f"Completed analysis of {len(results)} layers")
        return results

    def compute_fisher_vulnerability_matrix(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for vulnerability analysis.

        The Fisher Information Matrix provides a measure of how much information
        each parameter contains about the model's predictions. High Fisher information
        indicates parameters that strongly influence model behavior and are thus
        potentially vulnerable to targeted attacks.

        Args:
            inputs: Input tensors for Fisher computation
            labels: Target labels (if None, uses model predictions)
            layer_name: Specific layer to analyze (None for all)

        Returns:
            Dictionary mapping parameter names to Fisher information tensors
        """
        self.logger.info("Computing Fisher Information Matrix for vulnerability analysis")

        # Check cache
        cache_key = f"{id(inputs)}_{layer_name}"
        if cache_key in self._fisher_cache:
            return self._fisher_cache[cache_key]

        inputs = move_to_device(inputs, self.device)

        # If no labels provided, use model predictions
        if labels is None:
            with torch.no_grad():
                outputs = self.model(inputs)
                if hasattr(outputs, 'logits'):
                    labels = outputs.logits.argmax(dim=-1)
                else:
                    labels = outputs.argmax(dim=-1)

        labels = move_to_device(labels, self.device)

        fisher_matrices = {}

        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad_(True)

        # Compute Fisher information for each parameter
        target_layers = self._get_target_layers([layer_name] if layer_name else None)

        for layer_name, layer in target_layers.items():
            for param_name, param in layer.named_parameters():
                if param.requires_grad:
                    try:
                        fisher_info = self._compute_parameter_fisher_info(
                            param, inputs, labels, f"{layer_name}.{param_name}"
                        )
                        fisher_matrices[f"{layer_name}.{param_name}"] = fisher_info

                    except Exception as e:
                        self.logger.warning(f"Failed to compute Fisher info for {layer_name}.{param_name}: {e}")
                        continue

        # Cache results
        self._fisher_cache[cache_key] = fisher_matrices

        self.logger.info(f"Computed Fisher matrices for {len(fisher_matrices)} parameters")
        return fisher_matrices

    def detect_phase_transitions(
        self,
        weight_trajectories: Dict[str, List[torch.Tensor]],
        analysis_window: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect phase transitions in weight space during training/fine-tuning.

        Phase transitions represent critical points where small changes in weights
        lead to dramatic changes in model behavior. These points are particularly
        vulnerable to targeted attacks and represent natural breakpoints in the
        model's computational flow.

        Args:
            weight_trajectories: Time series of weight values for each parameter
            analysis_window: Window size for transition detection

        Returns:
            Dictionary mapping parameter names to detected phase transitions
        """
        self.logger.info("Detecting phase transitions in weight space")

        transitions = {}

        for param_name, trajectory in weight_trajectories.items():
            param_transitions = []

            if len(trajectory) < analysis_window * 2:
                self.logger.warning(f"Insufficient trajectory data for {param_name}")
                continue

            try:
                # Convert trajectory to numpy for analysis
                if isinstance(trajectory[0], torch.Tensor):
                    trajectory_np = [t.detach().cpu().numpy().flatten() for t in trajectory]
                else:
                    trajectory_np = [np.array(t).flatten() for t in trajectory]

                # Detect transitions using spectral methods
                for i in range(analysis_window, len(trajectory_np) - analysis_window):
                    transition_score = self._compute_transition_score(
                        trajectory_np, i, analysis_window
                    )

                    if transition_score > self.config["phase_transition_threshold"]:
                        transition_info = {
                            "timestep": i,
                            "transition_score": float(transition_score),
                            "pre_transition_statistics": self._compute_window_statistics(
                                trajectory_np[i-analysis_window:i]
                            ),
                            "post_transition_statistics": self._compute_window_statistics(
                                trajectory_np[i:i+analysis_window]
                            ),
                            "spectral_signature": self._compute_spectral_signature(
                                trajectory_np[i-analysis_window:i+analysis_window]
                            )
                        }
                        param_transitions.append(transition_info)

                if param_transitions:
                    transitions[param_name] = param_transitions

            except Exception as e:
                self.logger.warning(f"Failed to analyze transitions for {param_name}: {e}")
                continue

        self.logger.info(f"Detected phase transitions in {len(transitions)} parameters")
        return transitions

    def analyze_information_bottlenecks(
        self,
        inputs: torch.Tensor,
        layer_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze information bottlenecks in the network using mutual information.

        Information bottlenecks represent layers or components that compress
        information flow. These bottlenecks are often critical for model
        performance and can be vulnerable to targeted compression attacks.

        Args:
            inputs: Input tensors
            layer_outputs: Pre-computed layer outputs (optional)

        Returns:
            Dictionary mapping layers to bottleneck analysis results
        """
        self.logger.info("Analyzing information bottlenecks using mutual information")

        if layer_outputs is None:
            layer_outputs = self._extract_layer_outputs(inputs)

        bottleneck_analysis = {}
        layer_names = list(layer_outputs.keys())

        for i, layer_name in enumerate(layer_names):
            try:
                current_output = layer_outputs[layer_name]

                # Compute information metrics
                input_entropy = self._estimate_entropy(inputs)
                output_entropy = self._estimate_entropy(current_output)

                # Mutual information with input
                mi_with_input = self._estimate_mutual_information(inputs, current_output)

                # Mutual information with next layer (if exists)
                mi_with_next = 0.0
                if i < len(layer_names) - 1:
                    next_output = layer_outputs[layer_names[i + 1]]
                    mi_with_next = self._estimate_mutual_information(current_output, next_output)

                # Information compression ratio
                compression_ratio = output_entropy / input_entropy if input_entropy > 0 else 0.0

                # Bottleneck score (higher = more compressed)
                bottleneck_score = 1.0 - compression_ratio

                bottleneck_analysis[layer_name] = {
                    "input_entropy": float(input_entropy),
                    "output_entropy": float(output_entropy),
                    "mutual_info_input": float(mi_with_input),
                    "mutual_info_next": float(mi_with_next),
                    "compression_ratio": float(compression_ratio),
                    "bottleneck_score": float(bottleneck_score),
                    "is_bottleneck": bottleneck_score > self.config["bottleneck_compression_ratio"]
                }

            except Exception as e:
                self.logger.warning(f"Failed to analyze bottleneck for {layer_name}: {e}")
                continue

        self.logger.info(f"Analyzed information bottlenecks for {len(bottleneck_analysis)} layers")
        return bottleneck_analysis

    def export_analysis_results(
        self,
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        include_visualizations: bool = True
    ) -> Path:
        """
        Export information-theoretic analysis results to files.

        Args:
            results: Analysis results to export
            output_dir: Output directory for results
            include_visualizations: Whether to generate visualization plots

        Returns:
            Path to the exported results directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export main results as JSON
        results_file = output_path / "information_theoretic_analysis.json"

        # Convert results to serializable format
        serializable_results = self._make_serializable(results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Exported information-theoretic analysis to {results_file}")

        # Export summary statistics
        summary_file = output_path / "analysis_summary.txt"
        self._export_summary(serializable_results, summary_file)

        # Generate visualizations if requested
        if include_visualizations:
            viz_dir = output_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            self._generate_visualizations(serializable_results, viz_dir)

        return output_path

    # Private helper methods

    def _get_target_layers(self, target_layers: Optional[List[str]]) -> Dict[str, nn.Module]:
        """Get dictionary of target layers to analyze."""
        if target_layers is None:
            return dict(self.model.named_modules())
        else:
            layers = {}
            for name, module in self.model.named_modules():
                if any(target in name for target in target_layers):
                    layers[name] = module
            return layers

    def _analyze_parameter_information(
        self,
        layer_name: str,
        param_name: str,
        param: torch.Tensor,
        inputs: torch.Tensor,
        analysis_methods: List[str]
    ) -> Optional[WeightInformationProfile]:
        """Analyze information-theoretic properties of a single parameter."""
        try:
            metrics_dict = {}

            # Fisher information
            if "fisher" in analysis_methods:
                fisher_info = self._compute_parameter_fisher_info(
                    param, inputs, None, f"{layer_name}.{param_name}"
                )
                metrics_dict["fisher_information"] = float(torch.mean(fisher_info))
            else:
                metrics_dict["fisher_information"] = 0.0

            # Mutual information (simplified estimation)
            if "mutual_info" in analysis_methods:
                mi_score = self._estimate_parameter_mutual_information(param, inputs)
                metrics_dict["mutual_information"] = float(mi_score)
            else:
                metrics_dict["mutual_information"] = 0.0

            # Entropy
            param_entropy = self._estimate_entropy(param)
            metrics_dict["entropy"] = float(param_entropy)

            # Phase transition score (requires trajectory - simplified here)
            if "phase_transitions" in analysis_methods:
                phase_score = self._estimate_phase_transition_susceptibility(param)
                metrics_dict["phase_transition_score"] = float(phase_score)
            else:
                metrics_dict["phase_transition_score"] = 0.0

            # Bottleneck score
            if "bottleneck" in analysis_methods:
                bottleneck_score = self._estimate_bottleneck_score(param)
                metrics_dict["bottleneck_score"] = float(bottleneck_score)
            else:
                metrics_dict["bottleneck_score"] = 0.0

            # Spectral gap
            if SCIPY_AVAILABLE and param.dim() >= 2:
                spectral_gap = self._compute_spectral_gap(param)
                metrics_dict["spectral_gap"] = float(spectral_gap)
            else:
                metrics_dict["spectral_gap"] = 0.0

            # Create metrics object
            metrics = InformationMetrics(**metrics_dict)

            # Compute overall criticality score
            criticality_score = self._compute_criticality_score(metrics)

            # Estimate confidence
            confidence = self._estimate_confidence(metrics)

            return WeightInformationProfile(
                layer_idx=0,  # Simplified - would need layer indexing
                parameter_name=f"{layer_name}.{param_name}",
                coordinates=tuple(range(param.numel())),  # Simplified coordinates
                metrics=metrics,
                criticality_score=criticality_score,
                confidence=confidence
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze parameter {layer_name}.{param_name}: {e}")
            return None

    def _compute_parameter_fisher_info(
        self,
        param: torch.Tensor,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor],
        param_id: str
    ) -> torch.Tensor:
        """Compute Fisher information for a specific parameter."""
        # Simplified Fisher information computation
        # In practice, this would use the full Fisher information formula

        param.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)

        # Use cross-entropy loss for Fisher computation
        if labels is None:
            if hasattr(outputs, 'logits'):
                labels = outputs.logits.argmax(dim=-1)
            else:
                labels = outputs.argmax(dim=-1)

        # Compute loss
        if hasattr(outputs, 'logits'):
            loss = nn.functional.cross_entropy(outputs.logits, labels)
        else:
            loss = nn.functional.cross_entropy(outputs, labels)

        # Compute gradients
        gradients = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]

        # Fisher information approximation (gradient squared)
        fisher_info = gradients ** 2

        return fisher_info.detach()

    def _estimate_mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate mutual information between two tensors using InfoNet-inspired approach."""
        # Simplified MI estimation using correlation
        # In practice, this would use more sophisticated methods like MINE or InfoNet

        x_flat = x.detach().cpu().numpy().flatten()
        y_flat = y.detach().cpu().numpy().flatten()

        # Ensure same length
        min_len = min(len(x_flat), len(y_flat))
        x_flat = x_flat[:min_len]
        y_flat = y_flat[:min_len]

        # Compute correlation as MI proxy
        correlation = np.corrcoef(x_flat, y_flat)[0, 1]

        # Convert correlation to MI estimate
        mi_estimate = -0.5 * np.log(1 - correlation**2) if abs(correlation) < 0.99 else 1.0

        return max(0.0, mi_estimate)

    def _estimate_entropy(self, tensor: torch.Tensor) -> float:
        """Estimate entropy of tensor using histogram-based method."""
        # Convert to numpy
        data = tensor.detach().cpu().numpy().flatten()

        # Create histogram
        hist, _ = np.histogram(data, bins=50, density=True)

        # Remove zero probabilities
        hist = hist[hist > 0]

        # Normalize
        hist = hist / np.sum(hist)

        # Compute entropy
        return float(-np.sum(hist * np.log(hist + 1e-10)))

    def _compute_transition_score(
        self,
        trajectory: List[np.ndarray],
        timestep: int,
        window: int
    ) -> float:
        """Compute phase transition score at a specific timestep."""
        # Get pre and post windows
        pre_window = trajectory[timestep-window:timestep]
        post_window = trajectory[timestep:timestep+window]

        # Compute statistics
        pre_mean = np.mean([np.mean(w) for w in pre_window])
        post_mean = np.mean([np.mean(w) for w in post_window])
        pre_std = np.mean([np.std(w) for w in pre_window])
        post_std = np.mean([np.std(w) for w in post_window])

        # Transition score based on statistical change
        mean_change = abs(post_mean - pre_mean) / (abs(pre_mean) + 1e-10)
        std_change = abs(post_std - pre_std) / (pre_std + 1e-10)

        return mean_change + std_change

    def _compute_window_statistics(self, window: List[np.ndarray]) -> Dict[str, float]:
        """Compute statistics for a window of weight values."""
        all_values = np.concatenate([w.flatten() for w in window])

        return {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values)),
            "min": float(np.min(all_values)),
            "max": float(np.max(all_values)),
            "skewness": float(self._compute_skewness(all_values)),
            "kurtosis": float(self._compute_kurtosis(all_values))
        }

    def _compute_spectral_signature(self, window: List[np.ndarray]) -> Dict[str, float]:
        """Compute spectral signature for phase transition detection."""
        # Simplified spectral analysis
        all_values = np.concatenate([w.flatten() for w in window])

        # FFT-based spectral analysis
        fft_values = np.fft.fft(all_values)
        power_spectrum = np.abs(fft_values) ** 2

        return {
            "dominant_frequency": float(np.argmax(power_spectrum)),
            "spectral_centroid": float(np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)),
            "spectral_rolloff": float(np.percentile(power_spectrum, 85)),
            "spectral_flatness": float(np.exp(np.mean(np.log(power_spectrum + 1e-10))) / np.mean(power_spectrum))
        }

    def _extract_layer_outputs(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract outputs from all layers."""
        layer_outputs = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    layer_outputs[name] = output.detach()
                elif isinstance(output, tuple):
                    layer_outputs[name] = output[0].detach()
            return hook

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            self.model(inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return layer_outputs

    def _estimate_parameter_mutual_information(self, param: torch.Tensor, inputs: torch.Tensor) -> float:
        """Simplified parameter-specific mutual information estimation."""
        # This is a simplified proxy - in practice would use more sophisticated methods
        param_variance = torch.var(param).item()
        input_variance = torch.var(inputs).item()

        # Use variance ratio as MI proxy
        mi_proxy = param_variance / (input_variance + 1e-10)
        return min(1.0, mi_proxy)

    def _estimate_phase_transition_susceptibility(self, param: torch.Tensor) -> float:
        """Estimate how susceptible a parameter is to phase transitions."""
        # Based on parameter distribution properties
        param_flat = param.detach().cpu().numpy().flatten()

        # Compute higher-order moments
        skewness = self._compute_skewness(param_flat)
        kurtosis = self._compute_kurtosis(param_flat)

        # High skewness and kurtosis indicate potential for transitions
        susceptibility = (abs(skewness) + abs(kurtosis - 3)) / 2

        return min(1.0, susceptibility)

    def _estimate_bottleneck_score(self, param: torch.Tensor) -> float:
        """Estimate how much a parameter acts as an information bottleneck."""
        # Based on parameter magnitude and distribution
        param_magnitude = torch.norm(param).item()
        param_entropy = self._estimate_entropy(param)

        # Lower entropy and higher magnitude = more bottleneck-like
        bottleneck_score = param_magnitude / (param_entropy + 1e-10)

        return min(1.0, bottleneck_score / (param.numel() + 1))

    def _compute_spectral_gap(self, param: torch.Tensor) -> float:
        """Compute spectral gap of parameter matrix."""
        if not SCIPY_AVAILABLE:
            return 0.0

        # Reshape to 2D matrix if needed
        if param.dim() > 2:
            matrix = param.view(param.size(0), -1)
        else:
            matrix = param

        matrix_np = matrix.detach().cpu().numpy()

        # Compute eigenvalues
        eigenvals = eigvals(matrix_np @ matrix_np.T)
        eigenvals = np.real(eigenvals)
        eigenvals = np.sort(eigenvals)[::-1]  # Sort descending

        # Spectral gap is difference between largest and second-largest eigenvalue
        if len(eigenvals) >= 2:
            spectral_gap = eigenvals[0] - eigenvals[1]
        else:
            spectral_gap = eigenvals[0] if len(eigenvals) > 0 else 0.0

        return max(0.0, spectral_gap)

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        kurtosis = np.mean(((data - mean) / std) ** 4)
        return kurtosis

    def _compute_criticality_score(self, metrics: InformationMetrics) -> float:
        """Compute overall criticality score from information metrics."""
        # Weighted combination of different metrics
        weights = {
            "fisher": 0.3,
            "mutual_info": 0.2,
            "entropy": 0.15,
            "phase_transition": 0.15,
            "bottleneck": 0.1,
            "spectral": 0.1
        }

        score = (
            weights["fisher"] * metrics.fisher_information +
            weights["mutual_info"] * metrics.mutual_information +
            weights["entropy"] * metrics.entropy +
            weights["phase_transition"] * metrics.phase_transition_score +
            weights["bottleneck"] * metrics.bottleneck_score +
            weights["spectral"] * metrics.spectral_gap
        )

        return min(1.0, score)

    def _estimate_confidence(self, metrics: InformationMetrics) -> float:
        """Estimate confidence in the criticality assessment."""
        # Higher confidence when multiple metrics agree
        metric_values = [
            metrics.fisher_information,
            metrics.mutual_information,
            metrics.entropy,
            metrics.phase_transition_score,
            metrics.bottleneck_score,
            metrics.spectral_gap
        ]

        # Compute agreement (low variance = high confidence)
        variance = np.var(metric_values)
        confidence = 1.0 / (1.0 + variance)

        return min(1.0, confidence)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert results to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (WeightInformationProfile, InformationMetrics)):
            return obj.__dict__
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    def _export_summary(self, results: Dict[str, Any], summary_file: Path) -> None:
        """Export human-readable summary of results."""
        with open(summary_file, 'w') as f:
            f.write("Information-Theoretic Weight Criticality Analysis Summary\n")
            f.write("=" * 60 + "\n\n")

            # Add summary statistics here
            f.write("Analysis completed successfully.\n")
            f.write(f"Total layers analyzed: {len(results)}\n")

            # Add more detailed summary as needed

    def _generate_visualizations(self, results: Dict[str, Any], viz_dir: Path) -> None:
        """Generate visualization plots for the analysis results."""
        # Placeholder for visualization generation
        # Would implement matplotlib/seaborn plots here
        self.logger.info(f"Visualization generation not implemented yet. Results saved to {viz_dir}")