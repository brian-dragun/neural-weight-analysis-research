"""Base analyzer interface for weight analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseWeightAnalyzer(ABC):
    """
    Abstract base class for weight analyzers.

    Provides common interface and utilities for neural network weight analysis.
    """

    def __init__(self, model: torch.nn.Module, model_name: str):
        """
        Initialize base analyzer.

        Args:
            model: Neural network model to analyze
            model_name: Name/identifier for the model
        """
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device

        # Common configuration
        self.activation_threshold = 1e3
        self.early_layer_range = (0, 4)

        # Storage for analysis results
        self.analysis_cache = {}

    @abstractmethod
    def extract_critical_weights(
        self,
        mode: str = "discovery",
        sensitivity_threshold: float = 0.7,
        top_k_percent: float = 0.001,
        layer_focus: str = "early",
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """
        Extract critical weights from the model.

        Args:
            mode: Analysis mode
            sensitivity_threshold: Minimum sensitivity for inclusion
            top_k_percent: Percentage of top weights to extract
            layer_focus: Which layers to focus on
            output_dir: Output directory for results

        Returns:
            Dict containing analysis results
        """
        pass

    def _get_target_layers(self, layer_focus: str) -> List[int]:
        """Identify target layers based on focus specification."""
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

        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in ['layer.', 'h.', 'layers.']):
                if 'mlp' in name and 'down_proj' in name:
                    parts = name.split('.')
                    for part in parts:
                        if part.isdigit():
                            layer_count = max(layer_count, int(part) + 1)
                            break

        return layer_count

    def _extract_layer_number(self, module_name: str) -> int:
        """Extract layer number from module name."""
        parts = module_name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1

    def _get_module_by_layer_and_component(
        self,
        layer_idx: int,
        component: str
    ) -> Optional[nn.Module]:
        """Get module by layer index and component name."""
        for name, module in self.model.named_modules():
            if (f"layer.{layer_idx}.{component}" in name or
                f"h.{layer_idx}.{component}" in name or
                f"layers.{layer_idx}.{component}" in name):
                return module
        return None

    def _flat_to_coordinates(self, flat_idx: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional coordinates."""
        coords = []
        remaining = flat_idx

        for dim_size in reversed(shape):
            coords.append(remaining % dim_size)
            remaining //= dim_size

        return tuple(reversed(coords))

    def _generate_sample_inputs(
        self,
        batch_size: int = 8,
        seq_length: int = 32
    ) -> torch.Tensor:
        """Generate sample inputs for analysis."""
        try:
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                vocab_size = 50257  # Default for GPT-2

            return torch.randint(0, vocab_size, (batch_size, seq_length), device=self.device)
        except:
            return torch.randint(0, 50257, (batch_size, seq_length), device=self.device)

    def _setup_output_directory(self, output_dir: str) -> Path:
        """Setup and validate output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def _compute_sample_loss(self) -> float:
        """Compute loss on a small sample for sensitivity calculation."""
        self.model.eval()

        # Create a simple sample input
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'vocab_size'):
                vocab_size = self.model.config.vocab_size
            else:
                vocab_size = 50257  # Default for GPT-2
        else:
            vocab_size = 50257

        sample_input = torch.randint(0, vocab_size, (1, 32), device=self.device)

        with torch.no_grad():
            try:
                outputs = self.model(sample_input)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -log_probs.mean()
                return loss.item()
            except Exception as e:
                logger.warning(f"Error computing sample loss: {e}")
                return 0.0