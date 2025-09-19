"""Security-focused weight analysis for critical weight discovery (Phase A)."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging

from ..core.interfaces import SensitivityResult, CriticalWeightAnalysis
from .registry import register_sensitivity_metric

logger = logging.getLogger(__name__)


class SecurityWeightAnalyzer:
    """
    Phase A: Critical Weight Discovery & Vulnerability Analysis

    Identifies "super weights" most vulnerable to attacks through security-aware
    sensitivity analysis and vulnerability scoring.
    """

    def __init__(self, vulnerability_threshold: float = 0.8):
        self.vulnerability_threshold = vulnerability_threshold
        self.layer_importance_cache = {}

    def discover_critical_weights(
        self,
        model: torch.nn.Module,
        data_loader: Any,
        vulnerability_threshold: Optional[float] = None,
        top_k: int = 500,
        **kwargs
    ) -> CriticalWeightAnalysis:
        """
        Phase A: Identify super weights most vulnerable to attacks

        Returns:
            CriticalWeightAnalysis with:
            - critical_weights: List[Tuple[str, int, int]]  # layer, param_idx, vulnerability_score
            - vulnerability_map: Dict[str, float]           # per-layer vulnerability
            - attack_surface: Dict[str, Any]                # potential attack vectors
        """
        threshold = vulnerability_threshold or self.vulnerability_threshold

        logger.info(f"Starting critical weight discovery with threshold {threshold}")

        # Compute multiple security-aware sensitivity metrics
        security_gradients = self._compute_security_gradient_sensitivity(model, data_loader)
        weight_importance = self._compute_weight_importance_scores(model)
        layer_criticality = self._analyze_layer_criticality(model)

        # Combine metrics for vulnerability assessment
        vulnerability_scores = self._compute_vulnerability_scores(
            security_gradients, weight_importance, layer_criticality
        )

        # Identify critical weights above threshold
        critical_weights = self._identify_critical_weights(
            vulnerability_scores, threshold, top_k
        )

        # Analyze attack surface
        attack_surface = self._analyze_attack_surface(model, critical_weights)

        # Generate per-layer vulnerability map
        vulnerability_map = self._generate_vulnerability_map(vulnerability_scores)

        # Rank weights by security criticality
        security_ranking = self._rank_security_criticality(vulnerability_scores)

        logger.info(f"Discovered {len(critical_weights)} critical weights above threshold {threshold}")

        return CriticalWeightAnalysis(
            critical_weights=critical_weights,
            vulnerability_map=vulnerability_map,
            attack_surface=attack_surface,
            security_ranking=security_ranking,
            metadata={
                "threshold": threshold,
                "total_weights_analyzed": sum(len(scores) for scores in vulnerability_scores.values()),
                "analysis_method": "security_gradient_combined"
            }
        )

    def rank_attack_criticality(
        self,
        sensitivity_results: SensitivityResult
    ) -> Dict[str, float]:
        """Rank weights by their potential impact if compromised."""
        criticality_scores = {}

        for layer_name, param_idx, score in sensitivity_results.top_k_weights:
            weight_key = f"{layer_name}[{param_idx}]"

            # Base criticality from sensitivity score
            base_score = float(score)

            # Apply layer-specific multipliers
            layer_multiplier = self._get_layer_criticality_multiplier(layer_name)

            # Calculate position-based importance (attention heads, FFN bottlenecks)
            position_multiplier = self._get_position_criticality_multiplier(layer_name, param_idx)

            # Final criticality score
            criticality_scores[weight_key] = base_score * layer_multiplier * position_multiplier

        return criticality_scores

    def _compute_security_gradient_sensitivity(
        self,
        model: torch.nn.Module,
        data_loader: Any
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient-based sensitivity with security focus."""
        model.eval()
        all_gradients = {}

        # Initialize gradient storage
        for name, param in model.named_parameters():
            if param.requires_grad:
                all_gradients[name] = torch.zeros_like(param)

        num_batches = 0
        for batch in data_loader:
            try:
                model.zero_grad()

                # Compute loss with security-aware objective
                loss = self._compute_security_aware_loss(model, batch)
                loss.backward()

                # Accumulate gradients with security weighting
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Apply security weighting based on layer type
                        security_weight = self._get_security_weight(name)
                        weighted_grad = param.grad.abs() * security_weight
                        all_gradients[name] += weighted_grad

                num_batches += 1
                if num_batches >= 50:  # More batches for better security analysis
                    break

            except Exception as e:
                logger.warning(f"Skipping batch in security analysis: {e}")
                continue

        # Average gradients
        if num_batches > 0:
            for name in all_gradients:
                all_gradients[name] /= num_batches

        return all_gradients

    def _compute_weight_importance_scores(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Compute importance scores based on weight magnitudes and patterns."""
        importance_scores = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Base importance: magnitude
                magnitude_importance = param.abs()

                # Add variance-based importance (weights with high variance are more critical)
                if len(param.shape) > 1:
                    variance_importance = torch.var(param, dim=-1, keepdim=True).expand_as(param)
                else:
                    variance_importance = torch.var(param).expand_as(param)

                # Combine importance factors
                combined_importance = magnitude_importance + 0.5 * variance_importance
                importance_scores[name] = combined_importance

        return importance_scores

    def _analyze_layer_criticality(self, model: torch.nn.Module) -> Dict[str, float]:
        """Analyze which layers are most critical from security perspective."""
        layer_criticality = {}

        for name, param in model.named_parameters():
            # Extract layer type and position
            layer_type = self._classify_layer_type(name)
            layer_position = self._get_layer_position(name)

            # Assign criticality based on layer type
            base_criticality = {
                'attention': 0.9,    # Attention layers are highly critical
                'ffn': 0.8,          # Feed-forward networks
                'embedding': 0.95,   # Embeddings are extremely critical
                'output': 0.85,      # Output projections
                'layernorm': 0.6,    # Layer norm less critical but still important
                'bias': 0.4,         # Bias terms least critical
                'unknown': 0.5       # Default for unclassified
            }.get(layer_type, 0.5)

            # Apply position-based adjustments
            position_factor = self._get_position_factor(layer_position)

            layer_criticality[name] = base_criticality * position_factor

        return layer_criticality

    def _compute_vulnerability_scores(
        self,
        gradients: Dict[str, torch.Tensor],
        importance: Dict[str, torch.Tensor],
        criticality: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """Combine multiple metrics to compute final vulnerability scores."""
        vulnerability_scores = {}

        for name in gradients.keys():
            if name in importance and name in criticality:
                # Normalize gradients and importance to [0, 1]
                grad_norm = self._normalize_tensor(gradients[name])
                importance_norm = self._normalize_tensor(importance[name])

                # Combine with layer criticality
                combined_score = (
                    0.4 * grad_norm +           # Gradient sensitivity weight
                    0.35 * importance_norm +    # Importance weight
                    0.25 * criticality[name]    # Layer criticality weight
                )

                vulnerability_scores[name] = combined_score

        return vulnerability_scores

    def _identify_critical_weights(
        self,
        vulnerability_scores: Dict[str, torch.Tensor],
        threshold: float,
        top_k: int
    ) -> List[Tuple[str, int, float]]:
        """Identify weights above vulnerability threshold."""
        all_scores = []

        for layer_name, scores in vulnerability_scores.items():
            flat_scores = scores.flatten()
            for param_idx, score in enumerate(flat_scores):
                score_val = float(score.item())
                if score_val >= threshold:
                    all_scores.append((layer_name, param_idx, score_val))

        # Sort by vulnerability score and take top-k
        all_scores.sort(key=lambda x: x[2], reverse=True)
        return all_scores[:top_k]

    def _analyze_attack_surface(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]]
    ) -> Dict[str, Any]:
        """Analyze potential attack vectors based on critical weights."""
        attack_surface = {
            "total_critical_weights": len(critical_weights),
            "vulnerable_layers": set(),
            "attack_vectors": [],
            "risk_assessment": {}
        }

        # Group critical weights by layer
        layer_groups = defaultdict(list)
        for layer_name, param_idx, score in critical_weights:
            layer_groups[layer_name].append((param_idx, score))
            attack_surface["vulnerable_layers"].add(layer_name)

        # Analyze attack vectors
        for layer_name, weights in layer_groups.items():
            layer_type = self._classify_layer_type(layer_name)

            if layer_type == 'attention':
                attack_surface["attack_vectors"].append({
                    "type": "attention_disruption",
                    "target": layer_name,
                    "weight_count": len(weights),
                    "severity": "high"
                })
            elif layer_type == 'embedding':
                attack_surface["attack_vectors"].append({
                    "type": "embedding_poisoning",
                    "target": layer_name,
                    "weight_count": len(weights),
                    "severity": "critical"
                })
            elif layer_type == 'ffn':
                attack_surface["attack_vectors"].append({
                    "type": "computation_disruption",
                    "target": layer_name,
                    "weight_count": len(weights),
                    "severity": "medium"
                })

        # Risk assessment
        attack_surface["risk_assessment"] = {
            "overall_risk": "high" if len(critical_weights) > 100 else "medium",
            "most_vulnerable_layer": max(layer_groups.keys(), key=lambda x: len(layer_groups[x])),
            "attack_vector_count": len(attack_surface["attack_vectors"])
        }

        return attack_surface

    def _generate_vulnerability_map(
        self,
        vulnerability_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Generate per-layer vulnerability map."""
        vulnerability_map = {}

        for layer_name, scores in vulnerability_scores.items():
            # Calculate layer-level vulnerability as mean of top 10% of weights
            flat_scores = scores.flatten()
            top_10_percent = int(len(flat_scores) * 0.1)
            if top_10_percent < 1:
                top_10_percent = 1

            top_scores, _ = torch.topk(flat_scores, top_10_percent)
            vulnerability_map[layer_name] = float(top_scores.mean().item())

        return vulnerability_map

    def _rank_security_criticality(
        self,
        vulnerability_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Rank all weights by security criticality."""
        all_weights = []

        for layer_name, scores in vulnerability_scores.items():
            flat_scores = scores.flatten()
            for param_idx, score in enumerate(flat_scores):
                weight_key = f"{layer_name}[{param_idx}]"
                all_weights.append((weight_key, float(score.item())))

        # Sort and create ranking dictionary
        all_weights.sort(key=lambda x: x[1], reverse=True)

        ranking = {}
        for rank, (weight_key, score) in enumerate(all_weights):
            ranking[weight_key] = score

        return ranking

    # Helper methods
    def _compute_security_aware_loss(self, model: torch.nn.Module, batch: Dict) -> torch.Tensor:
        """Compute loss with security-aware objective."""
        try:
            if hasattr(model, 'transformer') or hasattr(model, 'bert'):
                # Transformer-style model
                outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})

                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                    # Use variance in hidden states as security-relevant signal
                    loss = hidden_states.var() + hidden_states.mean().abs()
                else:
                    loss = outputs[0].mean()
            else:
                # Fallback for other models
                outputs = model(**batch)
                loss = outputs.mean() if torch.is_tensor(outputs) else outputs[0].mean()

            return loss

        except Exception as e:
            logger.warning(f"Fallback loss computation: {e}")
            # Simple fallback: sum of all parameters
            return sum(param.sum() for param in model.parameters() if param.requires_grad)

    def _get_security_weight(self, layer_name: str) -> float:
        """Get security weighting factor for different layer types."""
        layer_type = self._classify_layer_type(layer_name)
        return {
            'attention': 1.5,
            'embedding': 1.8,
            'ffn': 1.2,
            'output': 1.3,
            'layernorm': 0.8,
            'bias': 0.5
        }.get(layer_type, 1.0)

    def _classify_layer_type(self, layer_name: str) -> str:
        """Classify layer type from parameter name."""
        name_lower = layer_name.lower()

        if any(x in name_lower for x in ['attn', 'attention', 'self_attn']):
            return 'attention'
        elif any(x in name_lower for x in ['embed', 'wte', 'wpe']):
            return 'embedding'
        elif any(x in name_lower for x in ['ffn', 'mlp', 'fc', 'linear']):
            return 'ffn'
        elif any(x in name_lower for x in ['ln', 'norm', 'layernorm']):
            return 'layernorm'
        elif 'bias' in name_lower:
            return 'bias'
        elif any(x in name_lower for x in ['output', 'head', 'proj']):
            return 'output'
        else:
            return 'unknown'

    def _get_layer_position(self, layer_name: str) -> int:
        """Extract layer position/number from parameter name."""
        import re
        numbers = re.findall(r'\d+', layer_name)
        return int(numbers[0]) if numbers else 0

    def _get_position_factor(self, position: int) -> float:
        """Get position-based criticality factor."""
        # Later layers often more critical in transformers
        return 1.0 + (position * 0.1)

    def _get_layer_criticality_multiplier(self, layer_name: str) -> float:
        """Get layer-specific criticality multiplier."""
        layer_type = self._classify_layer_type(layer_name)
        return {
            'attention': 1.5,
            'embedding': 1.8,
            'ffn': 1.2,
            'output': 1.4,
            'layernorm': 0.7,
            'bias': 0.3
        }.get(layer_type, 1.0)

    def _get_position_criticality_multiplier(self, layer_name: str, param_idx: int) -> float:
        """Get position-specific criticality multiplier."""
        # For attention layers, certain positions (query/key) more critical
        if 'attn' in layer_name.lower():
            return 1.2 if param_idx % 3 in [0, 1] else 1.0  # Q, K more critical than V
        return 1.0

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor values to [0, 1] range."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            return (tensor - min_val) / (max_val - min_val)
        return tensor


# Register the security-focused sensitivity metrics
@register_sensitivity_metric("security_gradient")
def compute_security_gradient_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    vulnerability_threshold: float = 0.8,
    **kwargs
) -> SensitivityResult:
    """
    Security-focused gradient sensitivity analysis.

    This metric identifies weights most vulnerable to security attacks
    by combining gradient information with security-aware objectives.
    """
    analyzer = SecurityWeightAnalyzer(vulnerability_threshold)

    # Get critical weight analysis
    critical_analysis = analyzer.discover_critical_weights(
        model, data_loader, vulnerability_threshold, top_k
    )

    # Convert to SensitivityResult format for compatibility
    values = {}
    for layer_name, param_idx, score in critical_analysis.critical_weights:
        if layer_name not in values:
            # Create a tensor to hold the scores for this layer
            param = dict(model.named_parameters())[layer_name]
            values[layer_name] = torch.zeros_like(param)

        # Set the specific parameter score
        flat_param = values[layer_name].flatten()
        if param_idx < len(flat_param):
            flat_param[param_idx] = score

    return SensitivityResult(
        values=values,
        metadata={
            "method": "security_gradient",
            "vulnerability_threshold": vulnerability_threshold,
            "critical_weight_count": len(critical_analysis.critical_weights),
            "attack_surface": critical_analysis.attack_surface,
            "vulnerability_map": critical_analysis.vulnerability_map
        },
        metric_name="security_gradient",
        top_k_weights=critical_analysis.critical_weights
    )


@register_sensitivity_metric("vulnerability_scanner")
def compute_vulnerability_scanning_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    scan_depth: str = "deep",
    **kwargs
) -> SensitivityResult:
    """
    Vulnerability scanning sensitivity analysis.

    Performs deep vulnerability scanning to identify potential
    security weaknesses in model weights.
    """
    analyzer = SecurityWeightAnalyzer()

    logger.info(f"Starting vulnerability scan with depth: {scan_depth}")

    # Perform vulnerability analysis
    critical_analysis = analyzer.discover_critical_weights(
        model, data_loader, vulnerability_threshold=0.7, top_k=top_k
    )

    # Additional vulnerability-specific analysis
    vuln_metadata = {
        "scan_depth": scan_depth,
        "vulnerabilities_found": len(critical_analysis.critical_weights),
        "risk_level": "high" if len(critical_analysis.critical_weights) > 200 else "medium",
        "recommended_monitoring": [w[0] for w in critical_analysis.critical_weights[:10]]
    }

    # Convert to SensitivityResult
    values = {}
    for layer_name, param_idx, score in critical_analysis.critical_weights:
        if layer_name not in values:
            param = dict(model.named_parameters())[layer_name]
            values[layer_name] = torch.zeros_like(param)

        flat_param = values[layer_name].flatten()
        if param_idx < len(flat_param):
            flat_param[param_idx] = score

    return SensitivityResult(
        values=values,
        metadata=vuln_metadata,
        metric_name="vulnerability_scanner",
        top_k_weights=critical_analysis.critical_weights
    )