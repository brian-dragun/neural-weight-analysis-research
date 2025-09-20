"""
Super Weight Analysis Module for PhD Research

Implements methodologies for discovering and validating "super weights" in transformer models
based on activation magnitude monitoring, Hessian-based sensitivity scoring, and perplexity impact analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict
import hashlib
from dataclasses import dataclass

# Import our new information-theoretic analyzer
from ..theory.information_geometry import InformationGeometricAnalyzer

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeightCandidate:
    """Container for ensemble weight discovery results."""
    layer_idx: int
    parameter_name: str
    coordinates: Tuple[int, ...]
    scores: Dict[str, float]  # Score from each discovery method
    ensemble_score: float
    confidence: float
    discovery_methods: List[str]

class SuperWeightAnalyzer:
    """
    PhD Research-focused analyzer for discovering and validating super weights in transformer models.

    Based on research methodology for identifying critical weights that cause 100x perplexity increases
    when modified, with focus on mlp.down_proj layers in early transformer layers.
    """

    def __init__(self, model: torch.nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device

        # Initialize information-theoretic analyzer for ensemble methods
        self.info_analyzer = InformationGeometricAnalyzer(model, device=str(self.device))

        # Known super weight coordinates from research
        self.known_super_weights = {
            "llama-7b": [(2, 'mlp.down_proj', [3968, 7003])],
            "mistral-7b": [(1, 'mlp.down_proj', [2070, 7310])],
            # Add more models as discovered
        }

        # Ensemble discovery configuration
        self.ensemble_config = {
            "methods": ["activation_outliers", "causal_intervention", "information_bottleneck", "spectral_anomaly"],
            "weights": {"activation_outliers": 0.3, "causal_intervention": 0.3, "information_bottleneck": 0.2, "spectral_anomaly": 0.2},
            "confidence_threshold": 0.7,
            "agreement_threshold": 0.6  # Require 60% of methods to agree
        }

        # Research parameters
        self.activation_threshold = 1e3
        self.sensitivity_percentile = 99.999  # Top 0.001%
        self.perplexity_threshold = 100
        self.early_layer_range = (0, 4)  # Layers 0-3 as per research

        # Storage for analysis results
        self.discovered_weights = []
        self.sensitivity_scores = {}
        self.activation_magnitudes = {}
        self.perplexity_impacts = {}

    def extract_critical_weights(
        self,
        mode: str = "super_weight_discovery",
        sensitivity_threshold: float = 0.7,
        top_k_percent: float = 0.001,
        layer_focus: str = "early",
        output_dir: str = "research_output"
    ) -> Dict[str, Any]:
        """
        Extract critical weights using PhD research methodology.

        Args:
            mode: Analysis mode ('super_weight_discovery', 'validation', 'comprehensive')
            sensitivity_threshold: Minimum sensitivity score for weight inclusion
            top_k_percent: Percentage of top weights to extract (0.001 = 0.001%)
            layer_focus: Layer range focus ('early', 'middle', 'late', 'all')
            output_dir: Directory for research outputs

        Returns:
            Dict containing discovered weights, statistics, and metadata
        """
        logger.info(f"Starting super weight extraction in {mode} mode")

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Identify target layers based on focus
        target_layers = self._get_target_layers(layer_focus)
        logger.info(f"Analyzing {len(target_layers)} target layers: {target_layers}")

        # Step 2: Monitor activation magnitudes
        activation_data = self._monitor_activation_magnitudes(target_layers)

        # Step 3: Calculate Hessian-based sensitivity scores
        sensitivity_data = self._calculate_hessian_sensitivity(target_layers)

        # Step 4: Discover new critical weights using ensemble methods
        if mode in ["super_weight_discovery", "comprehensive"]:
            discovered_weights = self._discover_super_weights_ensemble(
                activation_data, sensitivity_data, top_k_percent, target_layers
            )
        else:
            discovered_weights = []

        # Step 5: Validate known super weights if available
        validation_results = {}
        if mode in ["validation", "comprehensive"]:
            validation_results = self._validate_known_super_weights()

        # Step 6: Perform research-level analysis
        analysis_results = self._perform_research_analysis(
            discovered_weights, validation_results, activation_data, sensitivity_data
        )

        # Step 7: Generate research outputs
        research_data = {
            "model_name": self.model_name,
            "extraction_mode": mode,
            "parameters": {
                "sensitivity_threshold": sensitivity_threshold,
                "top_k_percent": top_k_percent,
                "layer_focus": layer_focus,
                "activation_threshold": self.activation_threshold
            },
            "target_layers": target_layers,
            "discovered_weights": discovered_weights,
            "validation_results": validation_results,
            "analysis_results": analysis_results,
            "statistics": self._generate_statistics(discovered_weights, sensitivity_data)
        }

        # Export results in multiple formats
        self._export_research_data(research_data, output_path)

        logger.info(f"Super weight extraction complete. Found {len(discovered_weights)} critical weights")
        return research_data

    def validate_super_weights(
        self,
        coordinates: List[Tuple[int, str, List[int]]],
        perplexity_threshold: float = 100,
        output_dir: str = "validation_output"
    ) -> Dict[str, Any]:
        """
        Validate specific super weight coordinates using 100x perplexity methodology.

        Args:
            coordinates: List of (layer, component, indices) tuples to validate
            perplexity_threshold: Minimum perplexity increase to confirm super weight
            output_dir: Directory for validation outputs

        Returns:
            Dict containing validation results and analysis
        """
        logger.info(f"Validating {len(coordinates)} super weight coordinates")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        validation_results = {}

        for coord_idx, (layer_idx, component, indices) in enumerate(coordinates):
            logger.info(f"Validating coordinate {coord_idx + 1}/{len(coordinates)}: "
                       f"Layer {layer_idx}, {component}, indices {indices}")

            # Perform perplexity impact test
            perplexity_impact = self._test_perplexity_impact(layer_idx, component, indices)

            # Perform bit-level criticality analysis
            bit_criticality = self._analyze_bit_level_criticality(layer_idx, component, indices)

            # Generate coordinate validation result
            coord_key = f"L{layer_idx}_{component}_{hash(str(indices))}"
            validation_results[coord_key] = {
                "coordinates": (layer_idx, component, indices),
                "perplexity_impact": perplexity_impact,
                "is_super_weight": perplexity_impact["multiplier"] >= perplexity_threshold,
                "bit_criticality": bit_criticality,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }

        # Generate validation report
        validation_report = {
            "model_name": self.model_name,
            "validation_parameters": {
                "perplexity_threshold": perplexity_threshold,
                "coordinates_tested": len(coordinates)
            },
            "results": validation_results,
            "summary": self._summarize_validation_results(validation_results),
            "methodology": self._get_validation_methodology()
        }

        # Export validation results
        self._export_validation_data(validation_report, output_path)

        logger.info(f"Validation complete. {sum(1 for r in validation_results.values() if r['is_super_weight'])} confirmed super weights")
        return validation_report

    def _get_target_layers(self, layer_focus: str) -> List[int]:
        """Identify target layers based on focus specification."""
        # Count total layers in model
        total_layers = self._count_transformer_layers()

        if layer_focus == "early":
            return list(range(min(self.early_layer_range[1], total_layers)))
        elif layer_focus == "middle":
            start = total_layers // 3
            end = 2 * total_layers // 3
            return list(range(start, end))
        elif layer_focus == "late":
            start = 2 * total_layers // 3
            return list(range(start, total_layers))
        elif layer_focus == "all":
            return list(range(total_layers))
        else:
            raise ValueError(f"Unknown layer_focus: {layer_focus}")

    def _count_transformer_layers(self) -> int:
        """Count transformer layers in the model."""
        layer_count = 0

        # Common transformer layer patterns
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in ['layer.', 'h.', 'layers.']):
                if 'mlp' in name and 'down_proj' in name:
                    # Extract layer number
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            layer_count = max(layer_count, int(part) + 1)
                            break

        return layer_count

    def _monitor_activation_magnitudes(self, target_layers: List[int]) -> Dict[str, Any]:
        """Monitor activation magnitudes in mlp.down_proj layers."""
        logger.info("Monitoring activation magnitudes in target layers")

        activation_data = defaultdict(list)
        hooks = []

        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    magnitude = torch.abs(output).max().item()
                    activation_data[layer_name].append(magnitude)

                    # Log high activations
                    if magnitude > self.activation_threshold:
                        logger.debug(f"High activation in {layer_name}: {magnitude:.2e}")
            return hook_fn

        # Register hooks on target mlp.down_proj layers
        for name, module in self.model.named_modules():
            if 'mlp.down_proj' in name:
                layer_num = self._extract_layer_number(name)
                if layer_num in target_layers:
                    hook = module.register_forward_hook(create_hook(name))
                    hooks.append(hook)

        # Run forward passes to collect activation data
        self._run_sample_forward_passes(num_samples=100)

        # Clean up hooks
        for hook in hooks:
            hook.remove()

        # Process activation data
        processed_data = {}
        for layer_name, magnitudes in activation_data.items():
            processed_data[layer_name] = {
                "max_magnitude": max(magnitudes) if magnitudes else 0,
                "mean_magnitude": np.mean(magnitudes) if magnitudes else 0,
                "std_magnitude": np.std(magnitudes) if magnitudes else 0,
                "high_activation_count": sum(1 for m in magnitudes if m > self.activation_threshold),
                "total_samples": len(magnitudes)
            }

        return processed_data

    def _calculate_hessian_sensitivity(self, target_layers: List[int]) -> Dict[str, Any]:
        """Calculate Hessian-based sensitivity scores for weights."""
        logger.info("Calculating Hessian-based sensitivity scores")

        sensitivity_data = {}

        # For each target layer
        for layer_idx in target_layers:
            layer_data = {}

            # Find mlp.down_proj module for this layer
            mlp_module = self._get_mlp_down_proj_module(layer_idx)
            if mlp_module is None:
                continue

            # Calculate weight sensitivity using approximated Hessian
            weight_sensitivity = self._approximate_hessian_sensitivity(mlp_module)

            layer_data = {
                "layer_index": layer_idx,
                "module_name": f"layer_{layer_idx}.mlp.down_proj",
                "weight_shape": list(mlp_module.weight.shape),
                "sensitivity_scores": weight_sensitivity,
                "top_sensitive_indices": self._get_top_sensitive_indices(weight_sensitivity)
            }

            sensitivity_data[f"layer_{layer_idx}"] = layer_data

        return sensitivity_data

    def _approximate_hessian_sensitivity(self, module: nn.Module) -> torch.Tensor:
        """Approximate Hessian-based sensitivity for module weights."""
        # Simplified Hessian approximation using gradient magnitude
        # In practice, you might want to use more sophisticated methods

        if not hasattr(module, 'weight') or module.weight is None:
            return torch.zeros((1, 1))

        weight = module.weight
        original_weight = weight.clone()

        # Calculate gradient-based sensitivity approximation
        sensitivity_scores = torch.zeros_like(weight)

        # Small perturbation for finite difference
        epsilon = 1e-4

        # Sample a subset of weights for efficiency
        total_weights = weight.numel()
        sample_size = min(1000, total_weights)  # Sample up to 1000 weights

        # Flatten weight tensor for sampling
        flat_weight = weight.view(-1)
        flat_sensitivity = torch.zeros_like(flat_weight)

        # Random sampling of weight indices
        sample_indices = torch.randperm(total_weights)[:sample_size]

        for idx in sample_indices:
            # Perturb weight
            original_val = flat_weight[idx].item()

            # Forward pass with positive perturbation
            flat_weight[idx] = original_val + epsilon
            loss_pos = self._compute_sample_loss()

            # Forward pass with negative perturbation
            flat_weight[idx] = original_val - epsilon
            loss_neg = self._compute_sample_loss()

            # Restore original weight
            flat_weight[idx] = original_val

            # Calculate sensitivity (second derivative approximation)
            sensitivity = abs(loss_pos + loss_neg - 2 * self._compute_sample_loss()) / (epsilon ** 2)
            flat_sensitivity[idx] = sensitivity

        # Reshape back to original weight shape
        sensitivity_scores = flat_sensitivity.view(weight.shape)

        return sensitivity_scores

    def _compute_sample_loss(self) -> float:
        """Compute loss on a small sample for sensitivity calculation."""
        self.model.eval()

        # Create a simple sample input (this should be adapted based on your model)
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                vocab_size = 50257  # Default for GPT-2
        else:
            vocab_size = 50257

        # Simple sample input
        sample_input = torch.randint(0, vocab_size, (1, 32), device=self.device)

        with torch.no_grad():
            try:
                outputs = self.model(sample_input)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # Simple perplexity-based loss
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -log_probs.mean()
                return loss.item()
            except Exception as e:
                logger.warning(f"Error computing sample loss: {e}")
                return 0.0

    def _discover_super_weights_ensemble(
        self,
        activation_data: Dict[str, Any],
        sensitivity_data: Dict[str, Any],
        top_k_percent: float,
        target_layers: List[str]
    ) -> List[EnsembleWeightCandidate]:
        """
        Enhanced super weight discovery using ensemble methods.

        Combines multiple discovery approaches:
        1. Activation outliers (existing method)
        2. Causal intervention analysis
        3. Information bottleneck detection
        4. Spectral anomaly detection

        Returns ensemble candidates with confidence scores.
        """
        logger.info(f"Starting ensemble super weight discovery (top {top_k_percent}%)")

        # Generate sample inputs for analysis
        sample_inputs = self._generate_sample_inputs(batch_size=16)

        # Method 1: Activation outliers (existing approach)
        logger.info("Running Method 1: Activation outlier detection")
        activation_candidates = self._method_activation_outliers(
            activation_data, sensitivity_data, top_k_percent
        )

        # Method 2: Causal intervention analysis
        logger.info("Running Method 2: Causal intervention analysis")
        causal_candidates = self._method_causal_intervention(
            sample_inputs, target_layers, top_k_percent
        )

        # Method 3: Information bottleneck detection
        logger.info("Running Method 3: Information bottleneck detection")
        bottleneck_candidates = self._method_information_bottleneck(
            sample_inputs, target_layers, top_k_percent
        )

        # Method 4: Spectral anomaly detection
        logger.info("Running Method 4: Spectral anomaly detection")
        spectral_candidates = self._method_spectral_anomaly(
            target_layers, top_k_percent
        )

        # Combine results using ensemble voting
        ensemble_candidates = self._ensemble_vote_candidates([
            ("activation_outliers", activation_candidates),
            ("causal_intervention", causal_candidates),
            ("information_bottleneck", bottleneck_candidates),
            ("spectral_anomaly", spectral_candidates)
        ])

        logger.info(f"Ensemble discovery completed. Found {len(ensemble_candidates)} high-confidence candidates")
        return ensemble_candidates

    def _method_activation_outliers(
        self,
        activation_data: Dict[str, Any],
        sensitivity_data: Dict[str, Any],
        top_k_percent: float
    ) -> List[EnsembleWeightCandidate]:
        """Method 1: Traditional activation outlier detection."""
        candidates = []

        for layer_key, layer_sensitivity in sensitivity_data.items():
            layer_idx = layer_sensitivity["layer_index"]
            sensitivity_scores = layer_sensitivity["sensitivity_scores"]

            flat_scores = sensitivity_scores.view(-1)
            total_weights = len(flat_scores)
            num_top_weights = max(1, int(total_weights * top_k_percent / 100))

            top_values, top_indices = torch.topk(flat_scores, num_top_weights)

            for i, (score, idx) in enumerate(zip(top_values, top_indices)):
                # Convert flat index to coordinates
                coords = self._flat_to_coordinates(idx.item(), sensitivity_scores.shape)

                candidate = EnsembleWeightCandidate(
                    layer_idx=layer_idx,
                    parameter_name=layer_key,
                    coordinates=coords,
                    scores={"activation_outliers": float(score)},
                    ensemble_score=float(score),
                    confidence=0.5,  # Will be updated in ensemble voting
                    discovery_methods=["activation_outliers"]
                )
                candidates.append(candidate)

        return candidates

    def _method_causal_intervention(
        self,
        sample_inputs: torch.Tensor,
        target_layers: List[str],
        top_k_percent: float
    ) -> List[EnsembleWeightCandidate]:
        """Method 2: Causal intervention analysis for weight criticality."""
        candidates = []

        # Get baseline performance
        baseline_loss = self._compute_sample_loss()

        for layer_name in target_layers:
            try:
                layer = dict(self.model.named_modules())[layer_name]

                for param_name, param in layer.named_parameters():
                    if not param.requires_grad or param.numel() < 10:
                        continue

                    # Sample random subset for efficiency
                    total_weights = param.numel()
                    num_test_weights = min(100, int(total_weights * top_k_percent / 100))

                    # Random sampling of weight positions
                    flat_param = param.view(-1)
                    test_indices = torch.randperm(total_weights)[:num_test_weights]

                    intervention_scores = []

                    for idx in test_indices:
                        # Causal intervention: temporarily zero out the weight
                        original_value = flat_param[idx].item()
                        flat_param[idx] = 0.0

                        # Measure causal effect
                        intervened_loss = self._compute_sample_loss()
                        causal_effect = abs(intervened_loss - baseline_loss)

                        # Restore original value
                        flat_param[idx] = original_value

                        intervention_scores.append((idx.item(), causal_effect))

                    # Select top causal interventions
                    intervention_scores.sort(key=lambda x: x[1], reverse=True)
                    top_interventions = intervention_scores[:max(1, len(intervention_scores)//4)]

                    for idx, score in top_interventions:
                        coords = self._flat_to_coordinates(idx, param.shape)

                        candidate = EnsembleWeightCandidate(
                            layer_idx=0,  # Will be updated
                            parameter_name=f"{layer_name}.{param_name}",
                            coordinates=coords,
                            scores={"causal_intervention": score},
                            ensemble_score=score,
                            confidence=0.5,
                            discovery_methods=["causal_intervention"]
                        )
                        candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Causal intervention failed for {layer_name}: {e}")
                continue

        return candidates

    def _method_information_bottleneck(
        self,
        sample_inputs: torch.Tensor,
        target_layers: List[str],
        top_k_percent: float
    ) -> List[EnsembleWeightCandidate]:
        """Method 3: Information bottleneck detection using information theory."""
        candidates = []

        try:
            # Use our information-theoretic analyzer
            bottleneck_analysis = self.info_analyzer.analyze_information_bottlenecks(sample_inputs)

            for layer_name, analysis in bottleneck_analysis.items():
                if layer_name not in target_layers:
                    continue

                if analysis.get("is_bottleneck", False):
                    bottleneck_score = analysis["bottleneck_score"]

                    # Get parameters for this layer
                    try:
                        layer = dict(self.model.named_modules())[layer_name]
                        for param_name, param in layer.named_parameters():
                            if param.requires_grad and param.numel() > 1:
                                # Sample weights proportional to bottleneck strength
                                num_weights = max(1, int(param.numel() * top_k_percent / 100 * bottleneck_score))

                                # Use information-theoretic metrics to select weights
                                info_scores = self._compute_information_scores(param)
                                top_indices = torch.topk(info_scores, min(num_weights, len(info_scores)))[1]

                                for idx in top_indices:
                                    coords = self._flat_to_coordinates(idx.item(), param.shape)

                                    candidate = EnsembleWeightCandidate(
                                        layer_idx=0,
                                        parameter_name=f"{layer_name}.{param_name}",
                                        coordinates=coords,
                                        scores={"information_bottleneck": bottleneck_score},
                                        ensemble_score=bottleneck_score,
                                        confidence=0.5,
                                        discovery_methods=["information_bottleneck"]
                                    )
                                    candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"Bottleneck analysis failed for {layer_name}: {e}")
                        continue

        except Exception as e:
            logger.warning(f"Information bottleneck method failed: {e}")

        return candidates

    def _method_spectral_anomaly(
        self,
        target_layers: List[str],
        top_k_percent: float
    ) -> List[EnsembleWeightCandidate]:
        """Method 4: Spectral anomaly detection using eigenvalue analysis."""
        candidates = []

        for layer_name in target_layers:
            try:
                layer = dict(self.model.named_modules())[layer_name]

                for param_name, param in layer.named_parameters():
                    if not param.requires_grad or param.dim() < 2:
                        continue

                    # Reshape to matrix for spectral analysis
                    if param.dim() > 2:
                        matrix = param.view(param.size(0), -1)
                    else:
                        matrix = param

                    # Compute SVD for spectral analysis
                    try:
                        U, S, V = torch.svd(matrix)

                        # Detect spectral anomalies
                        singular_values = S.detach().cpu().numpy()

                        # Look for gaps in singular value spectrum
                        if len(singular_values) > 2:
                            # Compute spectral gaps
                            gaps = np.diff(singular_values)
                            gap_threshold = np.std(gaps) * 2  # 2-sigma threshold

                            anomaly_indices = np.where(gaps > gap_threshold)[0]

                            if len(anomaly_indices) > 0:
                                # Get corresponding weight positions
                                num_anomalies = min(len(anomaly_indices),
                                                  max(1, int(param.numel() * top_k_percent / 100)))

                                for i in range(num_anomalies):
                                    sv_idx = anomaly_indices[i] if i < len(anomaly_indices) else 0
                                    spectral_score = float(gaps[sv_idx])

                                    # Sample random coordinate (simplified)
                                    rand_coord = tuple(torch.randint(0, s, (1,)).item() for s in param.shape)

                                    candidate = EnsembleWeightCandidate(
                                        layer_idx=0,
                                        parameter_name=f"{layer_name}.{param_name}",
                                        coordinates=rand_coord,
                                        scores={"spectral_anomaly": spectral_score},
                                        ensemble_score=spectral_score,
                                        confidence=0.5,
                                        discovery_methods=["spectral_anomaly"]
                                    )
                                    candidates.append(candidate)

                    except Exception as e:
                        logger.warning(f"SVD failed for {layer_name}.{param_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Spectral analysis failed for {layer_name}: {e}")
                continue

        return candidates

    def _ensemble_vote_candidates(
        self,
        method_results: List[Tuple[str, List[EnsembleWeightCandidate]]]
    ) -> List[EnsembleWeightCandidate]:
        """Combine candidates from multiple methods using weighted voting."""

        # Create candidate lookup by coordinates
        candidate_map = {}

        for method_name, candidates in method_results:
            for candidate in candidates:
                key = (candidate.parameter_name, candidate.coordinates)

                if key not in candidate_map:
                    candidate_map[key] = {
                        "candidate": candidate,
                        "methods": set(),
                        "scores": {}
                    }

                candidate_map[key]["methods"].add(method_name)
                candidate_map[key]["scores"][method_name] = candidate.scores.get(method_name, 0.0)

        # Compute ensemble scores and filter by agreement
        final_candidates = []

        for key, data in candidate_map.items():
            candidate = data["candidate"]
            methods = data["methods"]
            scores = data["scores"]

            # Require minimum method agreement
            agreement_ratio = len(methods) / len(self.ensemble_config["methods"])

            if agreement_ratio >= self.ensemble_config["agreement_threshold"]:
                # Compute weighted ensemble score
                ensemble_score = 0.0
                total_weight = 0.0

                for method in methods:
                    weight = self.ensemble_config["weights"].get(method, 0.25)
                    score = scores.get(method, 0.0)
                    ensemble_score += weight * score
                    total_weight += weight

                if total_weight > 0:
                    ensemble_score /= total_weight

                # Confidence based on method agreement and score consistency
                score_std = np.std(list(scores.values())) if len(scores) > 1 else 0.0
                confidence = agreement_ratio * (1.0 - min(1.0, score_std))

                # Update candidate with ensemble results
                candidate.scores.update(scores)
                candidate.ensemble_score = ensemble_score
                candidate.confidence = confidence
                candidate.discovery_methods = list(methods)

                if confidence >= self.ensemble_config["confidence_threshold"]:
                    final_candidates.append(candidate)

        # Sort by ensemble score
        final_candidates.sort(key=lambda x: x.ensemble_score, reverse=True)

        logger.info(f"Ensemble voting: {len(final_candidates)} high-confidence candidates from {len(candidate_map)} total")
        return final_candidates

    def _generate_sample_inputs(self, batch_size: int = 8, seq_length: int = 32) -> torch.Tensor:
        """Generate sample inputs for analysis."""
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                vocab_size = 50257  # Default for GPT-2

            return torch.randint(0, vocab_size, (batch_size, seq_length), device=self.device)
        except:
            return torch.randint(0, 50257, (batch_size, seq_length), device=self.device)

    def _compute_information_scores(self, param: torch.Tensor) -> torch.Tensor:
        """Compute information-theoretic scores for parameter selection."""
        # Simplified information scoring
        flat_param = param.view(-1)

        # Use magnitude and variance as information proxies
        magnitude_scores = torch.abs(flat_param)

        # Local variance (information content)
        if len(flat_param) > 1:
            padded = torch.cat([flat_param[:1], flat_param, flat_param[-1:]])
            local_var = torch.var(padded.unfold(0, 3, 1), dim=1)
            variance_scores = local_var
        else:
            variance_scores = torch.ones_like(magnitude_scores)

        # Combine scores
        info_scores = magnitude_scores * (1.0 + variance_scores)
        return info_scores

    def _flat_to_coordinates(self, flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional coordinates."""
        coords = []
        remaining = flat_idx

        for dim_size in reversed(shape):
            coords.append(remaining % dim_size)
            remaining //= dim_size

        return tuple(reversed(coords))

    def _discover_super_weights(
        self,
        activation_data: Dict[str, Any],
        sensitivity_data: Dict[str, Any],
        top_k_percent: float
    ) -> List[Dict[str, Any]]:
        """Discover new super weights based on activation and sensitivity analysis."""
        logger.info(f"Discovering super weights (top {top_k_percent}%)")

        discovered_weights = []

        for layer_key, layer_sensitivity in sensitivity_data.items():
            layer_idx = layer_sensitivity["layer_index"]
            sensitivity_scores = layer_sensitivity["sensitivity_scores"]

            # Flatten sensitivity scores and get indices
            flat_scores = sensitivity_scores.view(-1)
            total_weights = len(flat_scores)

            # Calculate number of top weights to extract
            num_top_weights = max(1, int(total_weights * top_k_percent / 100))

            # Get top sensitive indices
            top_values, top_indices = torch.topk(flat_scores, num_top_weights)

            # Convert flat indices back to 2D coordinates
            weight_shape = sensitivity_scores.shape

            for i, (value, flat_idx) in enumerate(zip(top_values, top_indices)):
                # Convert flat index to 2D coordinates
                row_idx = flat_idx.item() // weight_shape[1]
                col_idx = flat_idx.item() % weight_shape[1]

                # Skip known super weights to discover new ones
                coord = (layer_idx, 'mlp.down_proj', [row_idx, col_idx])
                if not self._is_known_super_weight(coord):
                    discovered_weights.append({
                        "layer_index": layer_idx,
                        "component": "mlp.down_proj",
                        "coordinates": [row_idx, col_idx],
                        "sensitivity_score": value.item(),
                        "rank": i + 1,
                        "discovery_method": "hessian_sensitivity"
                    })

        # Sort by sensitivity score
        discovered_weights.sort(key=lambda x: x["sensitivity_score"], reverse=True)

        return discovered_weights

    def _validate_known_super_weights(self) -> Dict[str, Any]:
        """Validate known super weights for the current model."""
        model_type = self._identify_model_type()

        if model_type not in self.known_super_weights:
            logger.info(f"No known super weights for model type: {model_type}")
            return {}

        coordinates = self.known_super_weights[model_type]
        return self.validate_super_weights(coordinates)

    def _test_perplexity_impact(
        self,
        layer_idx: int,
        component: str,
        indices: List[int]
    ) -> Dict[str, Any]:
        """Test perplexity impact of modifying specific weight coordinates."""
        logger.info(f"Testing perplexity impact for Layer {layer_idx}, {component}, indices {indices}")

        # Get the target module
        target_module = self._get_module_by_layer_and_component(layer_idx, component)
        if target_module is None:
            return {"error": "Module not found", "multiplier": 0}

        # Get original weight value
        original_weight = target_module.weight[indices[0], indices[1]].item()

        # Calculate baseline perplexity
        baseline_perplexity = self._calculate_model_perplexity()

        # Test different perturbation magnitudes
        perturbation_results = []
        perturbation_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for scale in perturbation_scales:
            # Apply perturbation
            target_module.weight[indices[0], indices[1]] = original_weight * scale

            # Calculate perturbed perplexity
            perturbed_perplexity = self._calculate_model_perplexity()

            # Calculate impact
            multiplier = perturbed_perplexity / baseline_perplexity if baseline_perplexity > 0 else 0

            perturbation_results.append({
                "scale": scale,
                "perplexity": perturbed_perplexity,
                "multiplier": multiplier
            })

            # Restore original weight
            target_module.weight[indices[0], indices[1]] = original_weight

        # Find maximum impact
        max_impact = max(perturbation_results, key=lambda x: x["multiplier"])

        return {
            "baseline_perplexity": baseline_perplexity,
            "perturbation_results": perturbation_results,
            "max_impact": max_impact,
            "multiplier": max_impact["multiplier"],
            "original_weight": original_weight
        }

    def _calculate_model_perplexity(self, num_samples: int = 50) -> float:
        """Calculate model perplexity on sample data."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Get vocabulary size
        if hasattr(self.model, 'config'):
            vocab_size = getattr(self.model.config, 'vocab_size', 50257)
        else:
            vocab_size = 50257

        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random sample
                sample_length = torch.randint(10, 50, (1,)).item()
                sample_input = torch.randint(0, vocab_size, (1, sample_length), device=self.device)

                try:
                    outputs = self.model(sample_input, labels=sample_input)
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    else:
                        # Calculate loss manually
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = sample_input[..., 1:].contiguous()
                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                    total_loss += loss.item() * sample_length
                    total_tokens += sample_length

                except Exception as e:
                    logger.warning(f"Error calculating perplexity: {e}")
                    continue

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def _analyze_bit_level_criticality(
        self,
        layer_idx: int,
        component: str,
        indices: List[int]
    ) -> Dict[str, Any]:
        """Analyze bit-level criticality (MSB→LSB ranking)."""
        target_module = self._get_module_by_layer_and_component(layer_idx, component)
        if target_module is None:
            return {"error": "Module not found"}

        original_weight = target_module.weight[indices[0], indices[1]].item()
        baseline_perplexity = self._calculate_model_perplexity(num_samples=10)  # Faster for bit analysis

        # Convert to binary representation
        weight_bytes = np.float32(original_weight).tobytes()
        bit_impacts = []

        # Test flipping each bit
        for byte_idx in range(len(weight_bytes)):
            for bit_idx in range(8):
                # Create modified bytes
                modified_bytes = bytearray(weight_bytes)
                modified_bytes[byte_idx] ^= (1 << bit_idx)

                # Convert back to float
                modified_weight = np.frombuffer(modified_bytes, dtype=np.float32)[0]

                # Test impact
                target_module.weight[indices[0], indices[1]] = modified_weight
                perturbed_perplexity = self._calculate_model_perplexity(num_samples=10)

                # Calculate impact
                impact = abs(perturbed_perplexity - baseline_perplexity) / baseline_perplexity

                bit_impacts.append({
                    "byte_index": byte_idx,
                    "bit_index": bit_idx,
                    "bit_position": byte_idx * 8 + bit_idx,
                    "impact": impact,
                    "is_msb": bit_idx == 7,  # Most significant bit of byte
                    "modified_weight": modified_weight
                })

                # Restore original weight
                target_module.weight[indices[0], indices[1]] = original_weight

        # Sort by impact (MSB→LSB criticality)
        bit_impacts.sort(key=lambda x: x["impact"], reverse=True)

        return {
            "original_weight": original_weight,
            "baseline_perplexity": baseline_perplexity,
            "bit_impacts": bit_impacts[:10],  # Top 10 most critical bits
            "msb_criticality_ratio": sum(1 for b in bit_impacts[:8] if b["is_msb"]) / 8
        }

    def _perform_research_analysis(
        self,
        discovered_weights: List[Dict[str, Any]],
        validation_results: Dict[str, Any],
        activation_data: Dict[str, Any],
        sensitivity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive research-level analysis."""
        logger.info("Performing research-level analysis")

        # Statistical analysis
        statistical_analysis = self._statistical_analysis(discovered_weights, sensitivity_data)

        # Behavioral analysis
        behavioral_analysis = self._behavioral_analysis(discovered_weights)

        # Architectural analysis
        architectural_analysis = self._architectural_analysis(discovered_weights, activation_data)

        return {
            "statistical": statistical_analysis,
            "behavioral": behavioral_analysis,
            "architectural": architectural_analysis,
            "summary": {
                "total_discovered": len(discovered_weights),
                "validated_super_weights": len([r for r in validation_results.values() if r.get("is_super_weight", False)]),
                "avg_sensitivity": np.mean([w["sensitivity_score"] for w in discovered_weights]) if discovered_weights else 0,
                "layer_distribution": self._analyze_layer_distribution(discovered_weights)
            }
        }

    def _statistical_analysis(self, discovered_weights: List[Dict[str, Any]], sensitivity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on discovered weights."""
        if not discovered_weights:
            return {"error": "No weights to analyze"}

        sensitivity_scores = [w["sensitivity_score"] for w in discovered_weights]
        layer_indices = [w["layer_index"] for w in discovered_weights]

        return {
            "sensitivity_distribution": {
                "mean": np.mean(sensitivity_scores),
                "std": np.std(sensitivity_scores),
                "min": np.min(sensitivity_scores),
                "max": np.max(sensitivity_scores),
                "percentiles": {
                    "25th": np.percentile(sensitivity_scores, 25),
                    "50th": np.percentile(sensitivity_scores, 50),
                    "75th": np.percentile(sensitivity_scores, 75),
                    "95th": np.percentile(sensitivity_scores, 95),
                    "99th": np.percentile(sensitivity_scores, 99)
                }
            },
            "layer_distribution": {
                "mean_layer": np.mean(layer_indices),
                "layer_counts": {str(layer): layer_indices.count(layer) for layer in set(layer_indices)},
                "early_layer_bias": sum(1 for l in layer_indices if l < 4) / len(layer_indices)
            },
            "correlations": self._calculate_correlations(discovered_weights)
        }

    def _behavioral_analysis(self, discovered_weights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behavioral impact of discovered weights."""
        if not discovered_weights:
            return {"error": "No weights to analyze"}

        # Sample a subset for behavioral testing (expensive operation)
        sample_size = min(10, len(discovered_weights))
        sample_weights = discovered_weights[:sample_size]

        behavioral_results = []

        for weight in sample_weights:
            layer_idx = weight["layer_index"]
            component = weight["component"]
            indices = weight["coordinates"]

            # Test perplexity impact
            perplexity_impact = self._test_perplexity_impact(layer_idx, component, indices)

            behavioral_results.append({
                "weight_id": f"L{layer_idx}_{component}_{indices[0]}_{indices[1]}",
                "sensitivity_score": weight["sensitivity_score"],
                "perplexity_multiplier": perplexity_impact.get("multiplier", 0),
                "is_significant": perplexity_impact.get("multiplier", 0) > 10  # 10x threshold
            })

        return {
            "tested_weights": len(behavioral_results),
            "significant_weights": sum(1 for r in behavioral_results if r["is_significant"]),
            "avg_perplexity_impact": np.mean([r["perplexity_multiplier"] for r in behavioral_results]),
            "max_perplexity_impact": max([r["perplexity_multiplier"] for r in behavioral_results]) if behavioral_results else 0,
            "detailed_results": behavioral_results
        }

    def _architectural_analysis(self, discovered_weights: List[Dict[str, Any]], activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architectural patterns in discovered weights."""
        if not discovered_weights:
            return {"error": "No weights to analyze"}

        # Component analysis
        component_counts = defaultdict(int)
        layer_counts = defaultdict(int)

        for weight in discovered_weights:
            component_counts[weight["component"]] += 1
            layer_counts[weight["layer_index"]] += 1

        # Vulnerability hotspots
        hotspots = []
        for layer_idx, count in layer_counts.items():
            if count > np.mean(list(layer_counts.values())):
                hotspots.append({
                    "layer_index": layer_idx,
                    "weight_count": count,
                    "vulnerability_density": count / len(discovered_weights)
                })

        return {
            "component_distribution": dict(component_counts),
            "layer_distribution": dict(layer_counts),
            "vulnerability_hotspots": hotspots,
            "early_layer_concentration": sum(layer_counts[i] for i in range(4)) / len(discovered_weights),
            "architectural_patterns": self._identify_architectural_patterns(discovered_weights)
        }

    # Helper methods
    def _extract_layer_number(self, module_name: str) -> int:
        """Extract layer number from module name."""
        parts = module_name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1

    def _get_mlp_down_proj_module(self, layer_idx: int) -> Optional[nn.Module]:
        """Get mlp.down_proj module for specific layer."""
        for name, module in self.model.named_modules():
            if f"layer.{layer_idx}.mlp.down_proj" in name or f"h.{layer_idx}.mlp.c_proj" in name:
                return module
        return None

    def _get_module_by_layer_and_component(self, layer_idx: int, component: str) -> Optional[nn.Module]:
        """Get module by layer index and component name."""
        for name, module in self.model.named_modules():
            if (f"layer.{layer_idx}.{component}" in name or
                f"h.{layer_idx}.{component}" in name or
                f"layers.{layer_idx}.{component}" in name):
                return module
        return None

    def _identify_model_type(self) -> str:
        """Identify model type from model name or architecture."""
        model_name_lower = self.model_name.lower()
        if "llama" in model_name_lower:
            return "llama-7b"
        elif "mistral" in model_name_lower:
            return "mistral-7b"
        else:
            return "unknown"

    def _is_known_super_weight(self, coord: Tuple[int, str, List[int]]) -> bool:
        """Check if coordinate is a known super weight."""
        model_type = self._identify_model_type()
        if model_type not in self.known_super_weights:
            return False

        known_coords = self.known_super_weights[model_type]
        return coord in known_coords

    def _run_sample_forward_passes(self, num_samples: int = 100):
        """Run forward passes to collect activation data."""
        self.model.eval()

        if hasattr(self.model, 'config'):
            vocab_size = getattr(self.model.config, 'vocab_size', 50257)
        else:
            vocab_size = 50257

        with torch.no_grad():
            for _ in range(num_samples):
                sample_length = torch.randint(10, 50, (1,)).item()
                sample_input = torch.randint(0, vocab_size, (1, sample_length), device=self.device)

                try:
                    _ = self.model(sample_input)
                except Exception as e:
                    logger.warning(f"Error in forward pass: {e}")
                    continue

    def _get_top_sensitive_indices(self, sensitivity_scores: torch.Tensor, top_k: int = 100) -> List[Tuple[int, int]]:
        """Get top-k most sensitive weight indices."""
        flat_scores = sensitivity_scores.view(-1)
        _, top_indices = torch.topk(flat_scores, min(top_k, len(flat_scores)))

        indices = []
        for flat_idx in top_indices:
            row_idx = flat_idx.item() // sensitivity_scores.shape[1]
            col_idx = flat_idx.item() % sensitivity_scores.shape[1]
            indices.append((row_idx, col_idx))

        return indices

    def _calculate_correlations(self, discovered_weights: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlations between different metrics."""
        if len(discovered_weights) < 2:
            return {}

        df = pd.DataFrame(discovered_weights)
        correlations = {}

        if 'sensitivity_score' in df.columns and 'layer_index' in df.columns:
            correlations['sensitivity_layer'] = df['sensitivity_score'].corr(df['layer_index'])

        return correlations

    def _analyze_layer_distribution(self, discovered_weights: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of weights across layers."""
        layer_counts = defaultdict(int)
        for weight in discovered_weights:
            layer_counts[weight["layer_index"]] += 1
        return dict(layer_counts)

    def _identify_architectural_patterns(self, discovered_weights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in architectural distribution."""
        patterns = {
            "early_layer_dominance": False,
            "mlp_concentration": False,
            "attention_concentration": False
        }

        # Early layer dominance (layers 0-3)
        early_weights = sum(1 for w in discovered_weights if w["layer_index"] < 4)
        if early_weights / len(discovered_weights) > 0.6:
            patterns["early_layer_dominance"] = True

        # Component concentration
        mlp_weights = sum(1 for w in discovered_weights if "mlp" in w["component"])
        if mlp_weights / len(discovered_weights) > 0.8:
            patterns["mlp_concentration"] = True

        attn_weights = sum(1 for w in discovered_weights if "attn" in w["component"])
        if attn_weights / len(discovered_weights) > 0.8:
            patterns["attention_concentration"] = True

        return patterns

    def _generate_statistics(self, discovered_weights: List[Dict[str, Any]], sensitivity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistics."""
        if not discovered_weights:
            return {"error": "No weights to analyze"}

        return {
            "total_weights_analyzed": sum(
                np.prod(data["weight_shape"]) for data in sensitivity_data.values()
            ),
            "critical_weights_found": len(discovered_weights),
            "discovery_rate": len(discovered_weights) / sum(
                np.prod(data["weight_shape"]) for data in sensitivity_data.values()
            ) * 100,
            "avg_sensitivity": np.mean([w["sensitivity_score"] for w in discovered_weights]),
            "layer_coverage": len(set(w["layer_index"] for w in discovered_weights)),
            "component_diversity": len(set(w["component"] for w in discovered_weights))
        }

    def _export_research_data(self, research_data: Dict[str, Any], output_path: Path):
        """Export research data in multiple formats."""
        logger.info(f"Exporting research data to {output_path}")

        # JSON export (full data)
        with open(output_path / "research_data.json", 'w') as f:
            json.dump(research_data, f, indent=2, default=str)

        # CSV export (discovered weights)
        if research_data["discovered_weights"]:
            df_weights = pd.DataFrame(research_data["discovered_weights"])
            df_weights.to_csv(output_path / "discovered_weights.csv", index=False)

        # Statistics CSV
        stats_data = []
        for key, value in research_data["statistics"].items():
            if isinstance(value, (int, float)):
                stats_data.append({"metric": key, "value": value})

        if stats_data:
            pd.DataFrame(stats_data).to_csv(output_path / "statistics.csv", index=False)

        # Research summary
        self._generate_research_summary(research_data, output_path)

        # Generate visualizations
        self._generate_visualizations(research_data, output_path)

    def _format_findings_section(self, research_data: Dict[str, Any]) -> str:
        """Format the findings section handling both success and error cases."""
        if 'error' in research_data['statistics']:
            return f"**Analysis Status**: {research_data['statistics']['error']}\n- **Total Critical Weights Discovered**: 0\n- **Discovery Rate**: 0.000000%\n- **Average Sensitivity Score**: 0.000000\n- **Layer Coverage**: 0 layers"

        stats = research_data['statistics']
        arch_results = research_data['analysis_results'].get('architectural', {})

        findings = f"""- **Total Critical Weights Discovered**: {stats.get('critical_weights_found', 0)}
- **Discovery Rate**: {stats.get('discovery_rate', 0):.6f}%
- **Average Sensitivity Score**: {stats.get('avg_sensitivity', 0):.6f}
- **Layer Coverage**: {stats.get('layer_coverage', 0)} layers"""

        if 'early_layer_concentration' in arch_results:
            findings += f"\n- **Early Layer Concentration**: {arch_results['early_layer_concentration']:.2%}"

        return findings

    def _export_validation_data(self, validation_report: Dict[str, Any], output_path: Path):
        """Export validation data in research formats."""
        logger.info(f"Exporting validation data to {output_path}")

        # JSON export
        with open(output_path / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)

        # CSV export
        validation_rows = []
        for coord_key, result in validation_report["results"].items():
            coord = result["coordinates"]
            validation_rows.append({
                "layer_index": coord[0],
                "component": coord[1],
                "row_index": coord[2][0],
                "col_index": coord[2][1],
                "perplexity_multiplier": result["perplexity_impact"]["multiplier"],
                "is_super_weight": result["is_super_weight"],
                "baseline_perplexity": result["perplexity_impact"]["baseline_perplexity"]
            })

        if validation_rows:
            pd.DataFrame(validation_rows).to_csv(output_path / "validation_results.csv", index=False)

    def _generate_research_summary(self, research_data: Dict[str, Any], output_path: Path):
        """Generate a research summary document."""
        summary = f"""
# Super Weight Analysis Research Summary

## Model: {research_data['model_name']}
## Analysis Mode: {research_data['extraction_mode']}
## Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

{self._format_findings_section(research_data)}

## Layer Distribution
"""

        layer_dist = research_data['analysis_results']['summary']['layer_distribution']
        for layer, count in sorted(layer_dist.items()):
            summary += f"- Layer {layer}: {count} weights\n"

        summary += f"""
## Validation Results
- **Validated Super Weights**: {research_data['analysis_results']['summary']['validated_super_weights']}

## Methodology
- Activation threshold: {research_data['parameters']['activation_threshold']}
- Top-K percentage: {research_data['parameters']['top_k_percent']}%
- Layer focus: {research_data['parameters']['layer_focus']}
- Sensitivity threshold: {research_data['parameters']['sensitivity_threshold']}

## Research Notes
This analysis uses PhD-level methodology for super weight discovery based on:
1. Activation magnitude monitoring (>1e3 threshold)
2. Hessian-based sensitivity scoring
3. 100× perplexity increase validation
4. Focus on early transformer layers (0-3)
5. Emphasis on mlp.down_proj components
"""

        with open(output_path / "research_summary.md", 'w') as f:
            f.write(summary)

    def _generate_visualizations(self, research_data: Dict[str, Any], output_path: Path):
        """Generate research visualizations."""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
            return

        try:
            # Set style
            plt.style.use('default')  # Use default style if seaborn not available
            if sns is not None:
                sns.set_palette("husl")

            # Create visualization directory
            viz_path = output_path / "visualizations"
            viz_path.mkdir(exist_ok=True)

            discovered_weights = research_data["discovered_weights"]
            if not discovered_weights:
                return

            df = pd.DataFrame(discovered_weights)

            # 1. Sensitivity Score Distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df['sensitivity_score'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Sensitivity Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Sensitivity Scores')
            plt.savefig(viz_path / 'sensitivity_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Layer Distribution
            plt.figure(figsize=(12, 6))
            layer_counts = df['layer_index'].value_counts().sort_index()
            layer_counts.plot(kind='bar')
            plt.xlabel('Layer Index')
            plt.ylabel('Number of Critical Weights')
            plt.title('Critical Weights Distribution Across Layers')
            plt.xticks(rotation=45)
            plt.savefig(viz_path / 'layer_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Sensitivity vs Layer Index
            plt.figure(figsize=(10, 6))
            plt.scatter(df['layer_index'], df['sensitivity_score'], alpha=0.6)
            plt.xlabel('Layer Index')
            plt.ylabel('Sensitivity Score')
            plt.title('Sensitivity Score vs Layer Index')

            # Add trend line
            z = np.polyfit(df['layer_index'], df['sensitivity_score'], 1)
            p = np.poly1d(z)
            plt.plot(df['layer_index'].sort_values(), p(df['layer_index'].sort_values()), "r--", alpha=0.8)

            plt.savefig(viz_path / 'sensitivity_vs_layer.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Heatmap of weight coordinates
            if len(df) > 10:  # Only if we have enough data
                plt.figure(figsize=(12, 8))
                coord_df = pd.DataFrame(df['coordinates'].tolist(), columns=['row', 'col'])
                coord_df['sensitivity'] = df['sensitivity_score']

                # Create a pivot table for heatmap
                pivot_data = coord_df.pivot_table(
                    values='sensitivity',
                    index='row',
                    columns='col',
                    aggfunc='mean'
                )

                if sns is not None:
                    sns.heatmap(pivot_data, cmap='viridis', cbar_kws={'label': 'Sensitivity Score'})
                else:
                    plt.imshow(pivot_data, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Sensitivity Score')
                plt.title('Sensitivity Heatmap by Weight Coordinates')
                plt.xlabel('Column Index')
                plt.ylabel('Row Index')
                plt.savefig(viz_path / 'coordinate_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()

            logger.info(f"Generated visualizations in {viz_path}")

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")

    def _summarize_validation_results(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize validation results."""
        confirmed_super_weights = sum(1 for r in validation_results.values() if r.get("is_super_weight", False))
        total_tested = len(validation_results)

        perplexity_multipliers = [
            r["perplexity_impact"]["multiplier"]
            for r in validation_results.values()
            if "perplexity_impact" in r
        ]

        return {
            "total_tested": total_tested,
            "confirmed_super_weights": confirmed_super_weights,
            "confirmation_rate": confirmed_super_weights / max(total_tested, 1),
            "avg_perplexity_impact": np.mean(perplexity_multipliers) if perplexity_multipliers else 0,
            "max_perplexity_impact": max(perplexity_multipliers) if perplexity_multipliers else 0,
            "super_weight_threshold_met": confirmed_super_weights > 0
        }

    def _get_validation_methodology(self) -> Dict[str, str]:
        """Get validation methodology description."""
        return {
            "perplexity_calculation": "Cross-entropy loss on random samples, converted to perplexity",
            "perturbation_method": "Weight scaling with multiple magnitudes (0.1x to 10x)",
            "bit_analysis": "Individual bit flipping with impact measurement",
            "threshold_criteria": f"Perplexity increase ≥{self.perplexity_threshold}x for super weight confirmation",
            "sample_size": "50 random sequences per perplexity calculation",
            "restoration_protocol": "Original weight restored after each test"
        }