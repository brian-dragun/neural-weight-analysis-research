"""Targeted attack methods focusing on critical weights discovered in Phase A."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import random

from ..core.interfaces import CriticalWeightAnalysis

logger = logging.getLogger(__name__)


class TargetedAttackSimulator:
    """
    Specialized attack simulator that focuses attacks on critical weights
    identified through Phase A vulnerability analysis.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.targeted_attack_history = []

    def simulate_attacks_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, int]],  # (layer, param_idx, vulnerability_score)
        attack_methods: List[str],
        test_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Phase B: Test attacks specifically on identified critical weights.

        This method implements focused attacks that target only the most
        vulnerable weights identified in Phase A analysis.

        Args:
            model: Target model
            critical_weights: Critical weights from Phase A
            attack_methods: List of attack methods to use
            test_data: Optional test data for evaluation

        Returns:
            Comprehensive attack results including:
            - attack_results: Per-method results
            - performance_degradation: Impact measurements
            - critical_failures: Weights causing major failures
            - recovery_metrics: Recovery analysis
        """
        logger.info(f"Simulating targeted attacks on {len(critical_weights)} critical weights")

        results = {
            "attack_results": {},
            "performance_degradation": {},
            "critical_failures": [],
            "recovery_metrics": {},
            "weight_vulnerability_ranking": {},
            "attack_success_correlation": {}
        }

        # Store baseline state
        baseline_state = self._capture_model_state()
        baseline_performance = self._measure_baseline_performance(test_data) if test_data else None

        # Group critical weights by vulnerability score for targeted analysis
        weight_groups = self._group_weights_by_vulnerability(critical_weights)

        for attack_method in attack_methods:
            logger.info(f"Executing targeted {attack_method} attack")

            try:
                # Execute targeted attack
                attack_result = self._execute_targeted_attack(
                    attack_method, weight_groups, test_data
                )

                results["attack_results"][attack_method] = attack_result

                # Measure performance impact
                if baseline_performance is not None:
                    performance_impact = self._measure_performance_degradation(
                        baseline_performance, test_data
                    )
                    results["performance_degradation"][attack_method] = performance_impact

                    # Check for critical failures
                    critical_failures = self._identify_critical_failures(
                        performance_impact, weight_groups
                    )
                    if critical_failures:
                        results["critical_failures"].extend(critical_failures)

                # Restore model for next attack
                self._restore_model_state(baseline_state)

            except Exception as e:
                logger.error(f"Targeted attack {attack_method} failed: {e}")
                results["attack_results"][attack_method] = {"error": str(e)}

        # Compute advanced analysis
        results["weight_vulnerability_ranking"] = self._analyze_weight_vulnerability_ranking(
            critical_weights, results["attack_results"]
        )

        results["attack_success_correlation"] = self._analyze_attack_success_correlation(
            results["attack_results"], weight_groups
        )

        results["recovery_metrics"] = self._compute_targeted_recovery_metrics(
            results, critical_weights
        )

        logger.info("Targeted attack simulation completed")
        return results

    def measure_attack_impact(
        self,
        original_model: torch.nn.Module,
        attacked_model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, int]]
    ) -> Dict[str, float]:
        """
        Measure specific impact of attacks on critical weights.

        Analyzes how attacking critical weights affects different aspects
        of model performance and behavior.
        """
        impact_metrics = {}

        try:
            # Weight-specific impact analysis
            weight_impact = self._analyze_weight_specific_impact(
                original_model, attacked_model, critical_weights
            )
            impact_metrics.update(weight_impact)

            # Layer-wise impact analysis
            layer_impact = self._analyze_layer_impact(
                original_model, attacked_model, critical_weights
            )
            impact_metrics["layer_impact"] = layer_impact

            # Functional impact analysis
            functional_impact = self._analyze_functional_impact(
                original_model, attacked_model
            )
            impact_metrics.update(functional_impact)

        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            impact_metrics["error"] = str(e)

        return impact_metrics

    def _group_weights_by_vulnerability(
        self, critical_weights: List[Tuple[str, int, int]]
    ) -> Dict[str, List[Tuple[str, int, int]]]:
        """Group weights by vulnerability level for targeted analysis."""
        groups = {
            "critical": [],    # Top 10% most vulnerable
            "high": [],        # 10-30% most vulnerable
            "medium": [],      # 30-60% most vulnerable
            "monitoring": []   # 60-100% for monitoring
        }

        if not critical_weights:
            return groups

        # Sort by vulnerability score
        sorted_weights = sorted(critical_weights, key=lambda x: x[2], reverse=True)
        total_weights = len(sorted_weights)

        # Distribute into groups
        critical_count = max(1, total_weights // 10)
        high_count = max(1, total_weights // 5)
        medium_count = max(1, total_weights // 3)

        groups["critical"] = sorted_weights[:critical_count]
        groups["high"] = sorted_weights[critical_count:critical_count + high_count]
        groups["medium"] = sorted_weights[critical_count + high_count:critical_count + high_count + medium_count]
        groups["monitoring"] = sorted_weights[critical_count + high_count + medium_count:]

        logger.info(f"Weight groups: critical={len(groups['critical'])}, high={len(groups['high'])}, "
                   f"medium={len(groups['medium'])}, monitoring={len(groups['monitoring'])}")

        return groups

    def _execute_targeted_attack(
        self,
        attack_method: str,
        weight_groups: Dict[str, List[Tuple[str, int, int]]],
        test_data: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute attack targeting specific weight groups."""

        if attack_method == "critical_weight_perturbation":
            return self._critical_weight_perturbation_attack(weight_groups)

        elif attack_method == "layered_vulnerability_attack":
            return self._layered_vulnerability_attack(weight_groups)

        elif attack_method == "cascading_failure_attack":
            return self._cascading_failure_attack(weight_groups, test_data)

        elif attack_method == "precision_targeting_attack":
            return self._precision_targeting_attack(weight_groups)

        else:
            # Fallback to weight-level perturbation
            return self._generic_weight_perturbation(weight_groups, attack_method)

    def _critical_weight_perturbation_attack(
        self, weight_groups: Dict[str, List[Tuple[str, int, int]]]
    ) -> Dict[str, Any]:
        """
        Direct perturbation attack on most critical weights.

        Targets the highest vulnerability weights with carefully calculated
        perturbations designed to maximize impact.
        """
        successful_attacks = 0
        total_attacks = 0
        attack_details = {}

        param_dict = dict(self.model.named_parameters())

        # Focus on critical and high vulnerability weights
        target_weights = weight_groups["critical"] + weight_groups["high"]

        for layer_name, param_idx, vulnerability_score in target_weights:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    original_value = flat_param[param_idx].item()

                    # Calculate perturbation based on vulnerability score
                    perturbation_strength = min(0.5 * vulnerability_score, 0.3)
                    perturbation = perturbation_strength * torch.sign(torch.randn(1)).item()

                    # Apply perturbation
                    with torch.no_grad():
                        flat_param[param_idx] += perturbation

                    attack_details[f"{layer_name}[{param_idx}]"] = {
                        "original_value": original_value,
                        "perturbation": perturbation,
                        "vulnerability_score": vulnerability_score
                    }

                    successful_attacks += 1

            total_attacks += 1

        success_rate = successful_attacks / max(total_attacks, 1)

        return {
            "attack_type": "critical_weight_perturbation",
            "success_rate": success_rate,
            "successful_attacks": successful_attacks,
            "total_attacks": total_attacks,
            "attack_details": attack_details,
            "target_groups": ["critical", "high"]
        }

    def _layered_vulnerability_attack(
        self, weight_groups: Dict[str, List[Tuple[str, int, int]]]
    ) -> Dict[str, Any]:
        """
        Attack that targets vulnerabilities layer by layer.

        Systematically attacks weights in each layer based on their
        vulnerability rankings.
        """
        layer_attacks = defaultdict(list)
        layer_results = {}

        # Group weights by layer
        all_weights = []
        for group_weights in weight_groups.values():
            all_weights.extend(group_weights)

        for layer_name, param_idx, vulnerability_score in all_weights:
            layer_attacks[layer_name].append((param_idx, vulnerability_score))

        param_dict = dict(self.model.named_parameters())
        total_successful = 0

        # Attack each layer systematically
        for layer_name, weight_list in layer_attacks.items():
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                # Sort weights in this layer by vulnerability
                weight_list.sort(key=lambda x: x[1], reverse=True)

                layer_successful = 0
                layer_total = len(weight_list)

                # Attack top vulnerable weights in this layer
                for param_idx, vulnerability_score in weight_list[:min(10, len(weight_list))]:
                    if param_idx < len(flat_param):
                        # Layer-specific perturbation strategy
                        if "attn" in layer_name.lower():
                            perturbation_factor = 0.4  # Attention layers more sensitive
                        elif "embed" in layer_name.lower():
                            perturbation_factor = 0.5  # Embeddings very sensitive
                        else:
                            perturbation_factor = 0.3  # Default

                        perturbation = perturbation_factor * vulnerability_score * torch.randn(1).item()

                        with torch.no_grad():
                            flat_param[param_idx] += perturbation

                        layer_successful += 1

                layer_results[layer_name] = {
                    "successful_attacks": layer_successful,
                    "total_weights": layer_total,
                    "success_rate": layer_successful / max(layer_total, 1)
                }

                total_successful += layer_successful

        overall_success_rate = total_successful / max(len(all_weights), 1)

        return {
            "attack_type": "layered_vulnerability",
            "success_rate": overall_success_rate,
            "total_successful": total_successful,
            "layer_results": layer_results,
            "layers_attacked": len(layer_results)
        }

    def _cascading_failure_attack(
        self,
        weight_groups: Dict[str, List[Tuple[str, int, int]]],
        test_data: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Attack designed to trigger cascading failures.

        Targets critical weights in a sequence designed to maximize
        the propagation of errors through the model.
        """
        cascading_results = {
            "stages": [],
            "cumulative_impact": [],
            "failure_threshold_reached": False
        }

        # Stage 1: Attack embedding layer critical weights
        embedding_weights = [w for w in weight_groups["critical"] if "embed" in w[0].lower()]
        if embedding_weights:
            stage1_result = self._attack_weight_subset(embedding_weights[:5], "embedding_stage")
            cascading_results["stages"].append(stage1_result)

            # Measure cumulative impact
            if test_data:
                impact = self._measure_performance_degradation(1.0, test_data[:3])
                cascading_results["cumulative_impact"].append(impact.get("performance_loss", 0))

        # Stage 2: Attack attention mechanism critical weights
        attention_weights = [w for w in weight_groups["critical"] if "attn" in w[0].lower()]
        if attention_weights:
            stage2_result = self._attack_weight_subset(attention_weights[:8], "attention_stage")
            cascading_results["stages"].append(stage2_result)

            if test_data:
                impact = self._measure_performance_degradation(1.0, test_data[:3])
                cascading_results["cumulative_impact"].append(impact.get("performance_loss", 0))

        # Stage 3: Attack feed-forward critical weights
        ffn_weights = [w for w in weight_groups["critical"] if any(x in w[0].lower() for x in ["ffn", "mlp", "fc"])]
        if ffn_weights:
            stage3_result = self._attack_weight_subset(ffn_weights[:6], "ffn_stage")
            cascading_results["stages"].append(stage3_result)

            if test_data:
                impact = self._measure_performance_degradation(1.0, test_data[:3])
                cascading_results["cumulative_impact"].append(impact.get("performance_loss", 0))

        # Check if failure threshold reached
        if cascading_results["cumulative_impact"]:
            max_impact = max(cascading_results["cumulative_impact"])
            cascading_results["failure_threshold_reached"] = max_impact > 0.7

        overall_success = len([s for s in cascading_results["stages"] if s.get("success_rate", 0) > 0.5])

        return {
            "attack_type": "cascading_failure",
            "success_rate": overall_success / max(len(cascading_results["stages"]), 1),
            "cascading_analysis": cascading_results,
            "max_cumulative_impact": max(cascading_results["cumulative_impact"]) if cascading_results["cumulative_impact"] else 0
        }

    def _precision_targeting_attack(
        self, weight_groups: Dict[str, List[Tuple[str, int, int]]]
    ) -> Dict[str, Any]:
        """
        High-precision attack on individual critical weights.

        Uses sophisticated perturbation strategies tailored to each
        specific weight's characteristics.
        """
        precision_results = {}
        total_precision_attacks = 0
        successful_precision_attacks = 0

        param_dict = dict(self.model.named_parameters())

        # Focus on top critical weights only
        top_critical = weight_groups["critical"][:20]  # Top 20 most critical

        for layer_name, param_idx, vulnerability_score in top_critical:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    # Analyze weight characteristics
                    weight_analysis = self._analyze_individual_weight(param, param_idx, layer_name)

                    # Calculate precision perturbation
                    perturbation = self._calculate_precision_perturbation(
                        weight_analysis, vulnerability_score
                    )

                    # Apply precision attack
                    original_value = flat_param[param_idx].item()
                    with torch.no_grad():
                        flat_param[param_idx] += perturbation

                    # Record precision attack details
                    precision_results[f"{layer_name}[{param_idx}]"] = {
                        "weight_analysis": weight_analysis,
                        "precision_perturbation": perturbation,
                        "original_value": original_value,
                        "vulnerability_score": vulnerability_score
                    }

                    successful_precision_attacks += 1

            total_precision_attacks += 1

        precision_success_rate = successful_precision_attacks / max(total_precision_attacks, 1)

        return {
            "attack_type": "precision_targeting",
            "success_rate": precision_success_rate,
            "precision_attacks": successful_precision_attacks,
            "total_targets": total_precision_attacks,
            "precision_details": precision_results
        }

    def _attack_weight_subset(
        self, weight_subset: List[Tuple[str, int, int]], stage_name: str
    ) -> Dict[str, Any]:
        """Attack a specific subset of weights."""
        successful = 0
        total = len(weight_subset)

        param_dict = dict(self.model.named_parameters())

        for layer_name, param_idx, vulnerability_score in weight_subset:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    perturbation = 0.4 * vulnerability_score * torch.randn(1).item()
                    with torch.no_grad():
                        flat_param[param_idx] += perturbation
                    successful += 1

        return {
            "stage_name": stage_name,
            "success_rate": successful / max(total, 1),
            "successful": successful,
            "total": total
        }

    def _analyze_individual_weight(
        self, param: torch.Tensor, param_idx: int, layer_name: str
    ) -> Dict[str, Any]:
        """Analyze characteristics of an individual weight."""
        flat_param = param.flatten()
        weight_value = flat_param[param_idx].item()

        # Local neighborhood analysis
        neighborhood_size = min(10, len(flat_param))
        start_idx = max(0, param_idx - neighborhood_size // 2)
        end_idx = min(len(flat_param), param_idx + neighborhood_size // 2)
        neighborhood = flat_param[start_idx:end_idx]

        analysis = {
            "weight_value": weight_value,
            "weight_magnitude": abs(weight_value),
            "neighborhood_mean": neighborhood.mean().item(),
            "neighborhood_std": neighborhood.std().item(),
            "relative_importance": abs(weight_value) / (neighborhood.abs().mean().item() + 1e-8),
            "layer_type": self._classify_layer_type(layer_name)
        }

        return analysis

    def _calculate_precision_perturbation(
        self, weight_analysis: Dict[str, Any], vulnerability_score: float
    ) -> float:
        """Calculate a precision-targeted perturbation."""
        base_perturbation = 0.3 * vulnerability_score

        # Adjust based on weight characteristics
        if weight_analysis["relative_importance"] > 2.0:
            # High importance weight - smaller perturbation
            base_perturbation *= 0.7
        elif weight_analysis["relative_importance"] < 0.5:
            # Low importance weight - larger perturbation
            base_perturbation *= 1.3

        # Layer-specific adjustments
        layer_type = weight_analysis["layer_type"]
        if layer_type == "attention":
            base_perturbation *= 1.2
        elif layer_type == "embedding":
            base_perturbation *= 1.4

        # Add directional component
        direction = 1 if weight_analysis["weight_value"] > 0 else -1
        precision_perturbation = base_perturbation * direction

        return precision_perturbation

    def _classify_layer_type(self, layer_name: str) -> str:
        """Classify the type of layer from its name."""
        name_lower = layer_name.lower()

        if any(x in name_lower for x in ['attn', 'attention']):
            return 'attention'
        elif any(x in name_lower for x in ['embed']):
            return 'embedding'
        elif any(x in name_lower for x in ['ffn', 'mlp', 'fc']):
            return 'feedforward'
        elif any(x in name_lower for x in ['norm', 'ln']):
            return 'normalization'
        else:
            return 'other'

    def _generic_weight_perturbation(
        self, weight_groups: Dict[str, List[Tuple[str, int, int]]], attack_method: str
    ) -> Dict[str, Any]:
        """Generic weight perturbation for unknown attack methods."""
        total_weights = sum(len(group) for group in weight_groups.values())
        target_weights = weight_groups["critical"] + weight_groups["high"]

        successful = 0
        param_dict = dict(self.model.named_parameters())

        for layer_name, param_idx, vulnerability_score in target_weights[:50]:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    perturbation = 0.2 * vulnerability_score * torch.randn(1).item()
                    with torch.no_grad():
                        flat_param[param_idx] += perturbation
                    successful += 1

        return {
            "attack_type": f"generic_{attack_method}",
            "success_rate": successful / max(len(target_weights), 1),
            "successful_attacks": successful,
            "total_targets": len(target_weights)
        }

    def _capture_model_state(self) -> Dict[str, torch.Tensor]:
        """Capture current model state."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def _restore_model_state(self, saved_state: Dict[str, torch.Tensor]) -> None:
        """Restore model to saved state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state:
                    param.copy_(saved_state[name])

    def _measure_baseline_performance(self, test_data: List[str]) -> float:
        """Measure baseline performance."""
        if not test_data or not self.tokenizer:
            return 1.0

        return self._evaluate_model_performance(test_data)

    def _measure_performance_degradation(
        self, baseline_performance: float, test_data: List[str]
    ) -> Dict[str, float]:
        """Measure performance degradation."""
        if not test_data:
            return {"performance_loss": 0.0}

        current_performance = self._evaluate_model_performance(test_data)
        performance_loss = (baseline_performance - current_performance) / baseline_performance

        return {
            "performance_loss": max(0.0, performance_loss),
            "current_performance": current_performance,
            "baseline_performance": baseline_performance
        }

    def _evaluate_model_performance(self, test_data: List[str]) -> float:
        """Evaluate model performance."""
        if not self.tokenizer:
            return 1.0

        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for text in test_data[:10]:  # Sample for efficiency
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                    outputs = self.model(**inputs)

                    if hasattr(outputs, 'logits'):
                        loss = outputs.logits.mean()
                    else:
                        loss = outputs[0].mean()

                    total_loss += loss.item()
                    num_samples += 1

                except Exception:
                    continue

        return 1.0 / (1.0 + total_loss / max(num_samples, 1))

    def _identify_critical_failures(
        self, performance_impact: Dict[str, float], weight_groups: Dict[str, List[Tuple[str, int, int]]]
    ) -> List[str]:
        """Identify critical failures."""
        critical_failures = []

        performance_loss = performance_impact.get("performance_loss", 0.0)

        if performance_loss > 0.6:  # Critical failure threshold
            # Identify responsible weights
            critical_weights = weight_groups["critical"][:10]
            for layer_name, param_idx, score in critical_weights:
                critical_failures.append(f"{layer_name}[{param_idx}]")

        return critical_failures

    def _analyze_weight_vulnerability_ranking(
        self, critical_weights: List[Tuple[str, int, int]], attack_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between vulnerability scores and attack success."""
        vulnerability_analysis = {
            "high_vulnerability_success": 0,
            "medium_vulnerability_success": 0,
            "low_vulnerability_success": 0,
            "correlation_strength": 0.0
        }

        # Count successful attacks by vulnerability level
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and result.get("success_rate", 0) > 0.5:
                # Successful attack - analyze which vulnerability levels were targeted
                vulnerability_analysis["high_vulnerability_success"] += 1

        return vulnerability_analysis

    def _analyze_attack_success_correlation(
        self, attack_results: Dict[str, Any], weight_groups: Dict[str, List[Tuple[str, int, int]]]
    ) -> Dict[str, float]:
        """Analyze correlation between attack methods and weight group success."""
        correlation_analysis = {}

        for attack_name, result in attack_results.items():
            if isinstance(result, dict):
                success_rate = result.get("success_rate", 0.0)
                target_groups = result.get("target_groups", ["unknown"])

                correlation_analysis[attack_name] = {
                    "success_rate": success_rate,
                    "primary_targets": target_groups,
                    "effectiveness_score": success_rate * len(target_groups)
                }

        return correlation_analysis

    def _compute_targeted_recovery_metrics(
        self, results: Dict[str, Any], critical_weights: List[Tuple[str, int, int]]
    ) -> Dict[str, Any]:
        """Compute recovery metrics specific to targeted attacks."""
        recovery_metrics = {
            "estimated_recovery_complexity": "medium",
            "critical_weights_compromised": 0,
            "layer_recovery_priorities": {},
            "recommended_recovery_order": []
        }

        # Count compromised critical weights
        successful_attacks = sum(
            1 for result in results["attack_results"].values()
            if isinstance(result, dict) and result.get("success_rate", 0) > 0.3
        )

        recovery_metrics["critical_weights_compromised"] = successful_attacks

        # Recovery complexity assessment
        if successful_attacks > 5:
            recovery_metrics["estimated_recovery_complexity"] = "high"
        elif successful_attacks > 2:
            recovery_metrics["estimated_recovery_complexity"] = "medium"
        else:
            recovery_metrics["estimated_recovery_complexity"] = "low"

        # Layer-wise recovery priorities
        layer_priorities = defaultdict(int)
        for layer_name, _, vulnerability_score in critical_weights[:20]:
            layer_priorities[layer_name] += vulnerability_score

        recovery_metrics["layer_recovery_priorities"] = dict(layer_priorities)

        # Recommended recovery order (highest priority first)
        recovery_order = sorted(layer_priorities.keys(), key=layer_priorities.get, reverse=True)
        recovery_metrics["recommended_recovery_order"] = recovery_order

        return recovery_metrics

    def _analyze_weight_specific_impact(
        self,
        original_model: torch.nn.Module,
        attacked_model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, int]]
    ) -> Dict[str, Any]:
        """Analyze impact specific to the attacked weights."""
        weight_impact = {}

        try:
            # Compare parameters between models
            orig_params = dict(original_model.named_parameters())
            attack_params = dict(attacked_model.named_parameters())

            total_deviation = 0.0
            max_deviation = 0.0

            for layer_name, param_idx, vulnerability_score in critical_weights[:50]:
                if layer_name in orig_params and layer_name in attack_params:
                    orig_param = orig_params[layer_name].flatten()
                    attack_param = attack_params[layer_name].flatten()

                    if param_idx < len(orig_param):
                        deviation = abs(orig_param[param_idx].item() - attack_param[param_idx].item())
                        total_deviation += deviation
                        max_deviation = max(max_deviation, deviation)

            weight_impact = {
                "average_weight_deviation": total_deviation / max(len(critical_weights), 1),
                "maximum_weight_deviation": max_deviation,
                "total_weights_analyzed": min(len(critical_weights), 50)
            }

        except Exception as e:
            logger.warning(f"Weight-specific impact analysis failed: {e}")

        return weight_impact

    def _analyze_layer_impact(
        self,
        original_model: torch.nn.Module,
        attacked_model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, int]]
    ) -> Dict[str, float]:
        """Analyze impact on different layers."""
        layer_impact = defaultdict(float)

        try:
            orig_params = dict(original_model.named_parameters())
            attack_params = dict(attacked_model.named_parameters())

            layer_deviations = defaultdict(list)

            for layer_name, param_idx, vulnerability_score in critical_weights:
                if layer_name in orig_params and layer_name in attack_params:
                    orig_param = orig_params[layer_name].flatten()
                    attack_param = attack_params[layer_name].flatten()

                    if param_idx < len(orig_param):
                        deviation = abs(orig_param[param_idx].item() - attack_param[param_idx].item())
                        layer_deviations[layer_name].append(deviation)

            # Compute layer-level impact
            for layer_name, deviations in layer_deviations.items():
                layer_impact[layer_name] = np.mean(deviations) if deviations else 0.0

        except Exception as e:
            logger.warning(f"Layer impact analysis failed: {e}")

        return dict(layer_impact)

    def _analyze_functional_impact(
        self, original_model: torch.nn.Module, attacked_model: torch.nn.Module
    ) -> Dict[str, float]:
        """Analyze functional impact of the attack."""
        functional_impact = {}

        try:
            # Parameter count comparison
            orig_param_count = sum(p.numel() for p in original_model.parameters())
            attack_param_count = sum(p.numel() for p in attacked_model.parameters())

            functional_impact["parameter_count_change"] = (
                attack_param_count - orig_param_count
            ) / orig_param_count if orig_param_count > 0 else 0.0

            # Model size impact
            orig_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
            attack_size = sum(p.numel() * p.element_size() for p in attacked_model.parameters())

            functional_impact["model_size_change"] = (attack_size - orig_size) / orig_size if orig_size > 0 else 0.0

        except Exception as e:
            logger.warning(f"Functional impact analysis failed: {e}")

        return functional_impact