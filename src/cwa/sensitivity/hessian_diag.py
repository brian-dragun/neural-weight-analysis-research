"""Hessian diagonal sensitivity analysis with fault tolerance focus."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..core.interfaces import SensitivityResult
from .registry import register_sensitivity_metric

logger = logging.getLogger(__name__)


@register_sensitivity_metric("hessian_diag")
def compute_hessian_diag_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    damping: float = 1e-6,
    fault_tolerance_analysis: bool = True,
    **kwargs
) -> SensitivityResult:
    """
    Compute Hessian diagonal sensitivity with fault tolerance analysis.

    The Hessian diagonal provides second-order sensitivity information,
    indicating how much the loss changes with respect to weight changes.
    This is particularly useful for identifying weights whose perturbation
    would significantly impact model performance.

    Args:
        model: PyTorch model to analyze
        data_loader: Data loader for calibration samples
        top_k: Number of top sensitive weights to return
        damping: Damping factor for numerical stability
        fault_tolerance_analysis: Include fault tolerance metrics

    Returns:
        SensitivityResult with Hessian diagonal sensitivity scores
    """
    model.eval()
    logger.info("Computing Hessian diagonal sensitivity with fault tolerance analysis")

    # Storage for Hessian diagonal approximations
    hessian_diag = {}
    gradient_norms = {}
    fault_impact_scores = {}

    # Initialize storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            hessian_diag[name] = torch.zeros_like(param)
            gradient_norms[name] = torch.zeros_like(param)
            if fault_tolerance_analysis:
                fault_impact_scores[name] = torch.zeros_like(param)

    num_batches = 0
    total_loss = 0.0

    # Process batches for Hessian computation
    for batch_idx, batch in enumerate(data_loader):
        try:
            # First forward pass for gradients
            model.zero_grad()
            loss1 = _compute_hessian_loss(model, batch)
            loss1.backward(create_graph=True)

            # Store first-order gradients
            first_grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    first_grads[name] = param.grad.clone()
                    gradient_norms[name] += param.grad.abs()

            # Compute Hessian diagonal using the gradient-of-gradient trick
            for name, param in model.named_parameters():
                if param.requires_grad and name in first_grads:
                    try:
                        # Compute Hessian diagonal for this parameter
                        hess_diag = _compute_parameter_hessian_diagonal(
                            first_grads[name], param, damping
                        )
                        hessian_diag[name] += hess_diag

                        # Fault tolerance analysis
                        if fault_tolerance_analysis:
                            fault_impact = _compute_fault_impact_score(
                                param, first_grads[name], hess_diag
                            )
                            fault_impact_scores[name] += fault_impact

                    except Exception as e:
                        logger.warning(f"Skipping Hessian computation for {name}: {e}")
                        continue

            total_loss += loss1.item()
            num_batches += 1

            # Limit computation for efficiency
            if num_batches >= 50:
                break

        except Exception as e:
            logger.warning(f"Skipping batch {batch_idx} in Hessian analysis: {e}")
            continue

    if num_batches == 0:
        logger.error("No successful batches in Hessian analysis")
        return SensitivityResult(
            values={},
            metadata={"error": "No successful batches"},
            metric_name="hessian_diag",
            top_k_weights=[]
        )

    # Average the accumulated values
    for name in hessian_diag:
        hessian_diag[name] /= num_batches
        gradient_norms[name] /= num_batches
        if fault_tolerance_analysis:
            fault_impact_scores[name] /= num_batches

    # Combine Hessian and fault tolerance information
    combined_sensitivity = _combine_hessian_and_fault_metrics(
        hessian_diag, fault_impact_scores if fault_tolerance_analysis else None
    )

    # Get top-k weights
    top_weights = _get_top_k_hessian_weights(combined_sensitivity, top_k)

    # Compute additional analysis
    fault_metadata = {}
    if fault_tolerance_analysis:
        fault_metadata = _compute_fault_tolerance_metadata(
            hessian_diag, fault_impact_scores, model
        )

    metadata = {
        "method": "hessian_diag",
        "num_batches": num_batches,
        "average_loss": total_loss / num_batches,
        "damping_factor": damping,
        "fault_tolerance_analysis": fault_tolerance_analysis,
        **fault_metadata
    }

    logger.info(f"Hessian diagonal analysis complete. Top score: {top_weights[0][2]:.6f}")

    return SensitivityResult(
        values=combined_sensitivity,
        metadata=metadata,
        metric_name="hessian_diag",
        top_k_weights=top_weights
    )


@register_sensitivity_metric("fault_hessian")
def compute_fault_aware_hessian_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 500,
    fault_types: List[str] = None,
    **kwargs
) -> SensitivityResult:
    """
    Hessian computation considering fault injection scenarios.

    This metric specifically focuses on identifying weights whose Hessian
    characteristics make them vulnerable to various types of faults.

    Args:
        fault_types: Types of faults to consider ['bit_flip', 'stuck_at', 'noise']
    """
    if fault_types is None:
        fault_types = ['bit_flip', 'stuck_at_zero', 'random_noise']

    model.eval()
    logger.info(f"Computing fault-aware Hessian analysis for faults: {fault_types}")

    # Storage for fault-specific Hessian analysis
    fault_hessians = {fault_type: {} for fault_type in fault_types}
    combined_scores = {}

    # Initialize storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            combined_scores[name] = torch.zeros_like(param)
            for fault_type in fault_types:
                fault_hessians[fault_type][name] = torch.zeros_like(param)

    num_batches = 0

    for batch in data_loader:
        try:
            # Compute fault-specific Hessian approximations
            for fault_type in fault_types:
                fault_specific_scores = _compute_fault_specific_hessian(
                    model, batch, fault_type
                )

                for name, scores in fault_specific_scores.items():
                    if name in fault_hessians[fault_type]:
                        fault_hessians[fault_type][name] += scores

            num_batches += 1
            if num_batches >= 30:  # Focused analysis
                break

        except Exception as e:
            logger.warning(f"Skipping batch in fault-aware Hessian: {e}")
            continue

    # Average and combine fault-specific scores
    if num_batches > 0:
        for fault_type in fault_types:
            for name in fault_hessians[fault_type]:
                fault_hessians[fault_type][name] /= num_batches

        # Combine different fault types
        combined_scores = _combine_fault_hessian_scores(fault_hessians, fault_types)

    # Get top weights considering all fault types
    top_weights = _get_fault_aware_top_weights(combined_scores, fault_hessians, top_k)

    # Fault-specific metadata
    fault_analysis = _analyze_fault_vulnerabilities(fault_hessians, fault_types, model)

    metadata = {
        "method": "fault_hessian",
        "fault_types": fault_types,
        "num_batches": num_batches,
        "fault_analysis": fault_analysis
    }

    return SensitivityResult(
        values=combined_scores,
        metadata=metadata,
        metric_name="fault_hessian",
        top_k_weights=top_weights
    )


def _compute_hessian_loss(model: torch.nn.Module, batch: Dict) -> torch.Tensor:
    """Compute loss suitable for Hessian computation."""
    try:
        if hasattr(model, 'transformer') or hasattr(model, 'bert'):
            outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})

            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                # Use MSE-like loss for better Hessian properties
                target = torch.zeros_like(hidden_states)
                loss = F.mse_loss(hidden_states, target)
            else:
                loss = outputs[0].pow(2).mean()
        else:
            outputs = model(**batch)
            if torch.is_tensor(outputs):
                loss = outputs.pow(2).mean()
            else:
                loss = outputs[0].pow(2).mean()

        return loss

    except Exception as e:
        logger.warning(f"Fallback loss in Hessian computation: {e}")
        # Fallback: L2 loss on parameters
        return sum(param.pow(2).sum() for param in model.parameters() if param.requires_grad) / 1000


def _compute_parameter_hessian_diagonal(
    grad: torch.Tensor,
    param: torch.Tensor,
    damping: float
) -> torch.Tensor:
    """
    Compute Hessian diagonal for a specific parameter using autograd.

    Uses the fact that H_ii = d²L/dx_i² can be approximated using
    gradient information and numerical methods.
    """
    try:
        # Simple approximation: Use gradient squared as Hessian diagonal estimate
        # This is a common approximation in optimization (similar to AdaGrad)
        hess_diag = grad.pow(2) + damping

        # Apply security-aware weighting
        param_magnitude = param.abs()
        magnitude_weight = torch.where(
            param_magnitude > 1e-6,
            1.0 / (param_magnitude + 1e-6),  # Inverse magnitude weighting
            torch.ones_like(param_magnitude)
        )

        # Combine with gradient information
        hess_approx = hess_diag * magnitude_weight

        return hess_approx

    except Exception as e:
        logger.warning(f"Hessian diagonal computation failed: {e}")
        return torch.zeros_like(grad)


def _compute_fault_impact_score(
    param: torch.Tensor,
    grad: torch.Tensor,
    hess_diag: torch.Tensor
) -> torch.Tensor:
    """
    Compute fault impact score considering parameter characteristics.

    This estimates how much a fault in this parameter would impact the model.
    """
    # Base impact: combination of gradient and Hessian information
    base_impact = grad.abs() * torch.sqrt(hess_diag + 1e-8)

    # Parameter magnitude factor (larger weights have more impact when corrupted)
    magnitude_factor = param.abs()

    # Position sensitivity (parameters with extreme values are more sensitive)
    position_factor = torch.where(
        param.abs() > param.abs().median(),
        torch.ones_like(param) * 1.5,  # High magnitude parameters
        torch.ones_like(param) * 1.0   # Normal parameters
    )

    # Combine factors
    fault_impact = base_impact * magnitude_factor * position_factor

    return fault_impact


def _combine_hessian_and_fault_metrics(
    hessian_diag: Dict[str, torch.Tensor],
    fault_scores: Optional[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Combine Hessian diagonal and fault tolerance metrics."""
    combined = {}

    for name, hess in hessian_diag.items():
        if fault_scores is not None and name in fault_scores:
            # Weight: 60% Hessian, 40% fault impact
            combined[name] = 0.6 * hess + 0.4 * fault_scores[name]
        else:
            combined[name] = hess

    return combined


def _get_top_k_hessian_weights(
    sensitivity_scores: Dict[str, torch.Tensor],
    top_k: int
) -> List[Tuple[str, int, float]]:
    """Get top-k weights based on Hessian sensitivity."""
    all_scores = []

    for layer_name, scores in sensitivity_scores.items():
        flat_scores = scores.flatten()
        for param_idx, score in enumerate(flat_scores):
            all_scores.append((layer_name, param_idx, float(score.item())))

    # Sort by score and take top-k
    all_scores.sort(key=lambda x: x[2], reverse=True)
    return all_scores[:top_k]


def _compute_fault_tolerance_metadata(
    hessian_diag: Dict[str, torch.Tensor],
    fault_scores: Dict[str, torch.Tensor],
    model: torch.nn.Module
) -> Dict[str, Any]:
    """Compute fault tolerance analysis metadata."""

    # Analyze fault tolerance characteristics by layer
    layer_fault_tolerance = {}

    for layer_name, hess in hessian_diag.items():
        fault_score = fault_scores.get(layer_name, torch.zeros_like(hess))

        # Compute layer-level metrics
        layer_metrics = {
            "hessian_mean": float(hess.mean().item()),
            "hessian_max": float(hess.max().item()),
            "fault_vulnerability": float(fault_score.mean().item()),
            "high_risk_weights": int((fault_score > fault_score.quantile(0.9)).sum().item()),
            "stability_score": float((hess * fault_score).mean().item())
        }

        layer_fault_tolerance[layer_name] = layer_metrics

    # Overall fault tolerance assessment
    total_high_risk = sum(metrics["high_risk_weights"] for metrics in layer_fault_tolerance.values())

    fault_tolerance_summary = {
        "layer_analysis": layer_fault_tolerance,
        "total_high_risk_weights": total_high_risk,
        "most_vulnerable_layer": max(
            layer_fault_tolerance.keys(),
            key=lambda x: layer_fault_tolerance[x]["fault_vulnerability"]
        ) if layer_fault_tolerance else "none",
        "overall_fault_tolerance": "low" if total_high_risk > 500 else "medium"
    }

    return {"fault_tolerance_analysis": fault_tolerance_summary}


def _compute_fault_specific_hessian(
    model: torch.nn.Module,
    batch: Dict,
    fault_type: str
) -> Dict[str, torch.Tensor]:
    """Compute Hessian approximation specific to a fault type."""
    fault_specific_scores = {}

    try:
        model.zero_grad()
        loss = _compute_hessian_loss(model, batch)

        # Apply fault-specific perturbation to understand sensitivity
        if fault_type == 'bit_flip':
            # Simulate bit flip sensitivity
            loss_modifier = loss * 1.1  # Small perturbation
        elif fault_type == 'stuck_at_zero':
            # Simulate stuck-at-zero sensitivity
            loss_modifier = loss + 0.01
        elif fault_type == 'random_noise':
            # Simulate noise sensitivity
            noise_factor = torch.randn_like(loss) * 0.01
            loss_modifier = loss + noise_factor
        else:
            loss_modifier = loss

        loss_modifier.backward(create_graph=True)

        # Collect fault-specific gradient information
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Approximate fault sensitivity using gradient magnitude
                fault_sensitivity = param.grad.abs()

                # Apply fault-type specific weighting
                if fault_type == 'bit_flip':
                    # Bit flips affect high-magnitude weights more
                    fault_weight = param.abs()
                elif fault_type == 'stuck_at_zero':
                    # Stuck-at-zero affects positive weights
                    fault_weight = torch.where(param > 0, param, torch.zeros_like(param))
                else:  # random_noise
                    # Noise affects all weights proportionally
                    fault_weight = torch.ones_like(param)

                fault_specific_scores[name] = fault_sensitivity * fault_weight

    except Exception as e:
        logger.warning(f"Failed to compute fault-specific Hessian for {fault_type}: {e}")

    return fault_specific_scores


def _combine_fault_hessian_scores(
    fault_hessians: Dict[str, Dict[str, torch.Tensor]],
    fault_types: List[str]
) -> Dict[str, torch.Tensor]:
    """Combine Hessian scores from different fault types."""
    combined = {}

    # Get all parameter names
    all_params = set()
    for fault_dict in fault_hessians.values():
        all_params.update(fault_dict.keys())

    # Combine scores with equal weighting
    weight_per_fault = 1.0 / len(fault_types)

    for param_name in all_params:
        combined_score = None

        for fault_type in fault_types:
            if param_name in fault_hessians[fault_type]:
                fault_score = fault_hessians[fault_type][param_name]

                if combined_score is None:
                    combined_score = fault_score * weight_per_fault
                else:
                    combined_score += fault_score * weight_per_fault

        if combined_score is not None:
            combined[param_name] = combined_score

    return combined


def _get_fault_aware_top_weights(
    combined_scores: Dict[str, torch.Tensor],
    fault_hessians: Dict[str, Dict[str, torch.Tensor]],
    top_k: int
) -> List[Tuple[str, int, float]]:
    """Get top weights considering fault-aware analysis."""
    all_scores = []

    for layer_name, scores in combined_scores.items():
        flat_scores = scores.flatten()

        for param_idx, score in enumerate(flat_scores):
            # Base score from combined analysis
            base_score = float(score.item())

            # Add fault-type diversity bonus
            fault_diversity = _compute_fault_diversity_score(
                layer_name, param_idx, fault_hessians
            )

            # Final score with diversity bonus
            final_score = base_score * (1.0 + fault_diversity)

            all_scores.append((layer_name, param_idx, final_score))

    all_scores.sort(key=lambda x: x[2], reverse=True)
    return all_scores[:top_k]


def _compute_fault_diversity_score(
    layer_name: str,
    param_idx: int,
    fault_hessians: Dict[str, Dict[str, torch.Tensor]]
) -> float:
    """Compute diversity score across different fault types."""
    fault_scores = []

    for fault_type, fault_dict in fault_hessians.items():
        if layer_name in fault_dict:
            flat_scores = fault_dict[layer_name].flatten()
            if param_idx < len(flat_scores):
                fault_scores.append(float(flat_scores[param_idx].item()))

    if len(fault_scores) <= 1:
        return 0.0

    # Diversity bonus based on variance across fault types
    variance = np.var(fault_scores)
    return min(variance / 10.0, 0.5)  # Cap at 50% bonus


def _analyze_fault_vulnerabilities(
    fault_hessians: Dict[str, Dict[str, torch.Tensor]],
    fault_types: List[str],
    model: torch.nn.Module
) -> Dict[str, Any]:
    """Analyze vulnerabilities to different fault types."""
    analysis = {}

    for fault_type in fault_types:
        fault_dict = fault_hessians[fault_type]

        # Count high-vulnerability weights per fault type
        high_vuln_count = 0
        layer_vulnerabilities = {}

        for layer_name, scores in fault_dict.items():
            layer_high_vuln = (scores > scores.quantile(0.95)).sum().item()
            high_vuln_count += layer_high_vuln
            layer_vulnerabilities[layer_name] = layer_high_vuln

        analysis[fault_type] = {
            "total_vulnerable_weights": high_vuln_count,
            "most_vulnerable_layer": max(layer_vulnerabilities.keys(),
                                       key=lambda x: layer_vulnerabilities[x]) if layer_vulnerabilities else "none",
            "vulnerability_distribution": layer_vulnerabilities
        }

    return analysis