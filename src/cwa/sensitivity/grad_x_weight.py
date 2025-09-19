"""Gradient × Weight sensitivity analysis with security focus."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.interfaces import SensitivityResult
from .registry import register_sensitivity_metric

logger = logging.getLogger(__name__)


@register_sensitivity_metric("grad_x_weight")
def compute_grad_x_weight_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    include_security_analysis: bool = True,
    **kwargs
) -> SensitivityResult:
    """
    Compute gradient × weight sensitivity with security analysis.

    This metric multiplies gradients by weight values to identify parameters
    that are both sensitive to changes and have high magnitude, making them
    prime targets for security attacks.

    Args:
        model: PyTorch model to analyze
        data_loader: Data loader for calibration samples
        top_k: Number of top sensitive weights to return
        include_security_analysis: Whether to include security-focused analysis

    Returns:
        SensitivityResult with gradient × weight sensitivity scores
    """
    model.eval()
    logger.info("Computing gradient × weight sensitivity with security analysis")

    # Storage for accumulated sensitivity scores
    sensitivity_scores = {}
    gradient_stats = {}
    weight_stats = {}

    # Initialize storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            sensitivity_scores[name] = torch.zeros_like(param)
            gradient_stats[name] = {"mean": 0.0, "var": 0.0, "count": 0}
            weight_stats[name] = {
                "magnitude": param.abs().mean().item(),
                "sparsity": (param.abs() < 1e-6).float().mean().item()
            }

    num_batches = 0
    total_samples = 0

    # Process calibration data
    for batch_idx, batch in enumerate(data_loader):
        try:
            model.zero_grad()

            # Forward pass with security-aware loss
            if include_security_analysis:
                loss = _compute_security_aware_loss(model, batch)
            else:
                loss = _compute_standard_loss(model, batch)

            # Backward pass
            loss.backward()

            # Accumulate gradient × weight scores
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Core gradient × weight computation
                    grad_x_weight = param.grad.abs() * param.abs()

                    # Apply security weighting if enabled
                    if include_security_analysis:
                        security_weight = _get_security_importance_weight(name)
                        grad_x_weight *= security_weight

                    # Accumulate scores
                    sensitivity_scores[name] += grad_x_weight

                    # Update gradient statistics
                    grad_stats = gradient_stats[name]
                    grad_mean = param.grad.abs().mean().item()
                    grad_stats["mean"] = (grad_stats["mean"] * grad_stats["count"] + grad_mean) / (grad_stats["count"] + 1)
                    grad_stats["var"] += (grad_mean - grad_stats["mean"]) ** 2
                    grad_stats["count"] += 1

            num_batches += 1
            total_samples += batch.get('input_ids', torch.tensor([1])).size(0)

            # Limit computation for efficiency
            if num_batches >= 100:
                break

        except Exception as e:
            logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
            continue

    if num_batches == 0:
        logger.error("No successful batches processed in grad_x_weight analysis")
        return SensitivityResult(
            values={},
            metadata={"error": "No successful batches"},
            metric_name="grad_x_weight",
            top_k_weights=[]
        )

    # Average sensitivity scores
    for name in sensitivity_scores:
        sensitivity_scores[name] /= num_batches

    # Get top-k weights with security ranking
    top_weights = _get_top_k_weights_with_security_ranking(
        sensitivity_scores, top_k, include_security_analysis
    )

    # Compute additional security metrics if requested
    security_metadata = {}
    if include_security_analysis:
        security_metadata = _compute_security_metadata(
            sensitivity_scores, gradient_stats, weight_stats, model
        )

    # Prepare metadata
    metadata = {
        "method": "grad_x_weight",
        "num_batches": num_batches,
        "total_samples": total_samples,
        "security_analysis": include_security_analysis,
        "gradient_statistics": gradient_stats,
        "weight_statistics": weight_stats,
        **security_metadata
    }

    logger.info(f"Grad×Weight analysis complete. Top score: {top_weights[0][2]:.6f}")

    return SensitivityResult(
        values=sensitivity_scores,
        metadata=metadata,
        metric_name="grad_x_weight",
        top_k_weights=top_weights
    )


@register_sensitivity_metric("security_grad_x_weight")
def compute_security_grad_x_weight_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    security_focus: str = "vulnerability",  # "vulnerability", "robustness", "integrity"
    **kwargs
) -> SensitivityResult:
    """
    Security-focused gradient × weight analysis with specific security objectives.

    Args:
        security_focus: Type of security analysis
            - "vulnerability": Focus on finding attack-vulnerable weights
            - "robustness": Focus on weights critical for model robustness
            - "integrity": Focus on weights important for model integrity
    """
    model.eval()
    logger.info(f"Computing security grad×weight with focus: {security_focus}")

    sensitivity_scores = {}
    security_scores = {}

    # Initialize storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            sensitivity_scores[name] = torch.zeros_like(param)
            security_scores[name] = torch.zeros_like(param)

    num_batches = 0

    for batch in data_loader:
        try:
            model.zero_grad()

            # Security-specific loss computation
            if security_focus == "vulnerability":
                loss = _compute_vulnerability_focused_loss(model, batch)
            elif security_focus == "robustness":
                loss = _compute_robustness_focused_loss(model, batch)
            elif security_focus == "integrity":
                loss = _compute_integrity_focused_loss(model, batch)
            else:
                loss = _compute_security_aware_loss(model, batch)

            loss.backward()

            # Compute security-weighted grad × weight
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Base grad × weight
                    grad_x_weight = param.grad.abs() * param.abs()

                    # Apply security-specific weighting
                    security_multiplier = _get_security_focus_multiplier(name, security_focus)
                    security_weighted = grad_x_weight * security_multiplier

                    sensitivity_scores[name] += security_weighted
                    security_scores[name] += security_multiplier

            num_batches += 1
            if num_batches >= 75:  # Focused analysis
                break

        except Exception as e:
            logger.warning(f"Skipping batch in security analysis: {e}")
            continue

    # Average scores
    if num_batches > 0:
        for name in sensitivity_scores:
            sensitivity_scores[name] /= num_batches
            security_scores[name] /= num_batches

    # Get top weights with security focus
    top_weights = _get_security_focused_top_weights(
        sensitivity_scores, security_scores, top_k, security_focus
    )

    # Security-specific metadata
    security_metadata = {
        "security_focus": security_focus,
        "security_weighting_applied": True,
        "focus_specific_metrics": _compute_focus_specific_metrics(
            sensitivity_scores, security_focus, model
        )
    }

    metadata = {
        "method": "security_grad_x_weight",
        "num_batches": num_batches,
        **security_metadata
    }

    return SensitivityResult(
        values=sensitivity_scores,
        metadata=metadata,
        metric_name="security_grad_x_weight",
        top_k_weights=top_weights
    )


def _compute_security_aware_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Compute loss with security-aware objectives."""
    try:
        # Standard forward pass
        if hasattr(model, 'transformer') or hasattr(model, 'bert'):
            outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})

            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                # Security objective: maximize gradient information
                loss = hidden_states.var() + hidden_states.norm()
            else:
                loss = outputs[0].mean().abs()
        else:
            outputs = model(**batch)
            loss = outputs.mean() if torch.is_tensor(outputs) else outputs[0].mean()

        return loss

    except Exception as e:
        logger.warning(f"Fallback in security loss: {e}")
        return sum(param.abs().mean() for param in model.parameters() if param.requires_grad)


def _compute_standard_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Compute standard forward pass loss."""
    try:
        if hasattr(model, 'transformer'):
            outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state.mean()
            else:
                return outputs[0].mean()
        else:
            outputs = model(**batch)
            return outputs.mean() if torch.is_tensor(outputs) else outputs[0].mean()
    except:
        return torch.tensor(0.0, requires_grad=True)


def _compute_vulnerability_focused_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Loss focusing on vulnerability discovery."""
    try:
        outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # Focus on activation patterns that reveal vulnerabilities
            return hidden.var(dim=-1).mean() + hidden.abs().max()
        else:
            return outputs[0].abs().mean()
    except:
        return torch.tensor(1.0, requires_grad=True)


def _compute_robustness_focused_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Loss focusing on model robustness."""
    try:
        outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # Focus on stability of representations
            return hidden.std(dim=1).mean() + hidden.norm()
        else:
            return outputs[0].std()
    except:
        return torch.tensor(1.0, requires_grad=True)


def _compute_integrity_focused_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Loss focusing on model integrity."""
    try:
        outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
            # Focus on consistency of outputs
            return hidden.mean().abs() + hidden.var()
        else:
            return outputs[0].mean().abs()
    except:
        return torch.tensor(1.0, requires_grad=True)


def _get_security_importance_weight(layer_name: str) -> float:
    """Get security importance weighting for different layer types."""
    name_lower = layer_name.lower()

    # Attention layers are highly important for security
    if any(x in name_lower for x in ['attn', 'attention', 'self_attn']):
        return 1.8
    # Embeddings are critical attack targets
    elif any(x in name_lower for x in ['embed', 'wte', 'wpe']):
        return 2.0
    # Feed-forward networks
    elif any(x in name_lower for x in ['ffn', 'mlp', 'fc', 'linear']):
        return 1.5
    # Output layers
    elif any(x in name_lower for x in ['output', 'head', 'proj']):
        return 1.6
    # Layer norm
    elif any(x in name_lower for x in ['ln', 'norm', 'layernorm']):
        return 1.0
    # Bias terms
    elif 'bias' in name_lower:
        return 0.7
    else:
        return 1.0


def _get_security_focus_multiplier(layer_name: str, focus: str) -> float:
    """Get multiplier based on security focus."""
    base_weight = _get_security_importance_weight(layer_name)

    focus_multipliers = {
        "vulnerability": {
            "attention": 1.5, "embedding": 1.8, "ffn": 1.2, "output": 1.3
        },
        "robustness": {
            "attention": 1.3, "embedding": 1.5, "ffn": 1.4, "output": 1.2
        },
        "integrity": {
            "attention": 1.4, "embedding": 1.6, "ffn": 1.1, "output": 1.5
        }
    }

    # Classify layer type
    name_lower = layer_name.lower()
    if any(x in name_lower for x in ['attn', 'attention']):
        layer_type = "attention"
    elif any(x in name_lower for x in ['embed']):
        layer_type = "embedding"
    elif any(x in name_lower for x in ['ffn', 'mlp', 'fc']):
        layer_type = "ffn"
    elif any(x in name_lower for x in ['output', 'head']):
        layer_type = "output"
    else:
        layer_type = "other"

    focus_mult = focus_multipliers.get(focus, {}).get(layer_type, 1.0)
    return base_weight * focus_mult


def _get_top_k_weights_with_security_ranking(
    sensitivity_scores: Dict[str, torch.Tensor],
    top_k: int,
    include_security: bool
) -> List[tuple]:
    """Get top-k weights with optional security ranking."""
    all_scores = []

    for layer_name, scores in sensitivity_scores.items():
        flat_scores = scores.flatten()
        for param_idx, score in enumerate(flat_scores):
            score_val = float(score.item())

            # Apply security ranking if enabled
            if include_security:
                security_mult = _get_security_importance_weight(layer_name)
                score_val *= security_mult

            all_scores.append((layer_name, param_idx, score_val))

    # Sort and return top-k
    all_scores.sort(key=lambda x: x[2], reverse=True)
    return all_scores[:top_k]


def _get_security_focused_top_weights(
    sensitivity_scores: Dict[str, torch.Tensor],
    security_scores: Dict[str, torch.Tensor],
    top_k: int,
    focus: str
) -> List[tuple]:
    """Get top weights with specific security focus."""
    all_scores = []

    for layer_name, scores in sensitivity_scores.items():
        sec_scores = security_scores[layer_name]
        flat_scores = scores.flatten()
        flat_sec = sec_scores.flatten()

        for param_idx, (score, sec_score) in enumerate(zip(flat_scores, flat_sec)):
            # Combine sensitivity and security scores
            combined_score = float(score.item()) * (1.0 + float(sec_score.item()))
            all_scores.append((layer_name, param_idx, combined_score))

    all_scores.sort(key=lambda x: x[2], reverse=True)
    return all_scores[:top_k]


def _compute_security_metadata(
    sensitivity_scores: Dict[str, torch.Tensor],
    gradient_stats: Dict[str, Dict],
    weight_stats: Dict[str, Dict],
    model: torch.nn.Module
) -> Dict[str, Any]:
    """Compute additional security-related metadata."""

    # Analyze layer-wise security characteristics
    layer_security = {}
    total_high_sensitivity = 0

    for layer_name, scores in sensitivity_scores.items():
        flat_scores = scores.flatten()

        # Security metrics for this layer
        layer_security[layer_name] = {
            "max_sensitivity": float(flat_scores.max().item()),
            "mean_sensitivity": float(flat_scores.mean().item()),
            "high_sensitivity_count": int((flat_scores > flat_scores.quantile(0.95)).sum().item()),
            "vulnerability_score": float(flat_scores.quantile(0.99).item())
        }

        total_high_sensitivity += layer_security[layer_name]["high_sensitivity_count"]

    # Overall security assessment
    security_assessment = {
        "total_high_sensitivity_weights": total_high_sensitivity,
        "most_vulnerable_layer": max(layer_security.keys(),
                                   key=lambda x: layer_security[x]["vulnerability_score"]),
        "security_distribution": layer_security,
        "risk_level": "high" if total_high_sensitivity > 1000 else "medium"
    }

    return {
        "security_analysis": security_assessment,
        "vulnerability_hotspots": [
            layer for layer, metrics in layer_security.items()
            if metrics["vulnerability_score"] > 0.5
        ]
    }


def _compute_focus_specific_metrics(
    sensitivity_scores: Dict[str, torch.Tensor],
    focus: str,
    model: torch.nn.Module
) -> Dict[str, Any]:
    """Compute metrics specific to the security focus."""

    metrics = {}

    if focus == "vulnerability":
        # Count potential vulnerability points
        vuln_count = 0
        for scores in sensitivity_scores.values():
            vuln_count += (scores > scores.quantile(0.9)).sum().item()
        metrics["potential_vulnerability_points"] = vuln_count

    elif focus == "robustness":
        # Analyze robustness characteristics
        robustness_scores = []
        for scores in sensitivity_scores.values():
            robustness_scores.append(float(scores.std().item()))
        metrics["robustness_variance"] = np.mean(robustness_scores)

    elif focus == "integrity":
        # Analyze integrity characteristics
        integrity_scores = []
        for scores in sensitivity_scores.values():
            integrity_scores.append(float(scores.mean().item()))
        metrics["integrity_baseline"] = np.mean(integrity_scores)

    return metrics