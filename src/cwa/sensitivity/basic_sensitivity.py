"""Basic sensitivity analysis implementations."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any
from ..core.interfaces import SensitivityResult
import logging

logger = logging.getLogger(__name__)


def compute_basic_gradient_sensitivity(
    model: torch.nn.Module,
    data_loader: Any,
    top_k: int = 100,
    **kwargs
) -> SensitivityResult:
    """
    Compute basic gradient-based sensitivity.
    This is a simplified version for the foundation build.
    """
    model.eval()
    all_gradients = {}

    logger.info("Computing basic gradient sensitivity...")

    # Initialize gradient storage
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_gradients[name] = torch.zeros_like(param)

    num_batches = 0
    for batch in data_loader:
        try:
            # Simple forward pass
            model.zero_grad()

            if hasattr(model, 'transformer'):  # GPT-style model
                outputs = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']})
                # Use a simple loss (mean of hidden states)
                if hasattr(outputs, 'last_hidden_state'):
                    loss = outputs.last_hidden_state.mean()
                else:
                    loss = outputs[0].mean()
            else:
                # Fallback for other model types
                loss = model(**batch).mean()

            loss.backward()

            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    all_gradients[name] += param.grad.abs()

            num_batches += 1

            if num_batches >= 10:  # Limit for foundation build
                break

        except Exception as e:
            logger.warning(f"Skipping batch due to error: {e}")
            continue

    if num_batches == 0:
        logger.error("No successful batches processed")
        return SensitivityResult(
            values={},
            metadata={"error": "No successful batches"},
            metric_name="basic_gradient",
            top_k_weights=[]
        )

    # Average gradients
    for name in all_gradients:
        all_gradients[name] /= num_batches

    # Get top-k weights
    all_scores = []
    for name, grads in all_gradients.items():
        flat_grads = grads.flatten()
        for i, score in enumerate(flat_grads):
            all_scores.append((name, i, score.item()))

    # Sort by score and take top-k
    all_scores.sort(key=lambda x: x[2], reverse=True)
    top_k_weights = all_scores[:top_k]

    logger.info(f"Completed gradient sensitivity analysis. Top score: {top_k_weights[0][2]:.6f}")

    return SensitivityResult(
        values=all_gradients,
        metadata={"num_batches": num_batches, "method": "basic_gradient"},
        metric_name="basic_gradient",
        top_k_weights=top_k_weights
    )