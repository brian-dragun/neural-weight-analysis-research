"""
Cross-Architecture Weight Mapping System

Implements sophisticated weight mapping strategies to transfer learned
representations between different neural network architectures while
preserving critical patterns and functionality.

Key Features:
- Multiple mapping strategies (geometric, semantic, functional)
- Weight interpolation and transformation methods
- Architecture compatibility analysis
- Mapping quality assessment
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


class MappingStrategy(Enum):
    """Strategies for mapping weights between architectures."""
    GEOMETRIC_MAPPING = "geometric_mapping"  # Based on tensor shapes and positions
    SEMANTIC_MAPPING = "semantic_mapping"    # Based on functional similarity
    INTERPOLATION_MAPPING = "interpolation_mapping"  # Using interpolation techniques
    ATTENTION_MAPPING = "attention_mapping"  # Using attention mechanisms
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # Through distillation
    PROGRESSIVE_MAPPING = "progressive_mapping"  # Layer-by-layer progressive transfer


class MappingQuality(Enum):
    """Quality levels of weight mappings."""
    PERFECT = "perfect"          # Exact match, no adaptation needed
    HIGH = "high"               # Minor adaptation needed
    MODERATE = "moderate"       # Significant adaptation required
    LOW = "low"                # Major structural differences
    INCOMPATIBLE = "incompatible"  # Cannot be mapped effectively


@dataclass
class WeightMapping:
    """Container for weight mapping information."""
    mapping_id: str
    source_component: str
    target_component: str
    mapping_strategy: MappingStrategy

    # Mapping characteristics
    quality: MappingQuality
    compatibility_score: float  # 0.0 to 1.0
    geometric_similarity: float
    semantic_similarity: float

    # Transformation details
    transformation_matrix: Optional[torch.Tensor] = None
    interpolation_weights: Optional[List[float]] = None
    adaptation_parameters: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    preservation_score: float = 0.0  # How well original functionality is preserved
    transfer_efficiency: float = 0.0  # How efficiently the mapping transfers
    stability_score: float = 0.0     # Stability of the mapping

    # Metadata
    mapping_timestamp: float = 0.0
    computation_cost: float = 0.0
    confidence: float = 0.0


@dataclass
class MappingConfig:
    """Configuration for architecture mapping."""
    # Mapping parameters
    compatibility_threshold: float = 0.6  # Minimum compatibility for mapping
    interpolation_steps: int = 10         # Steps for interpolation mapping
    attention_heads: int = 8              # For attention-based mapping

    # Quality thresholds
    perfect_threshold: float = 0.95
    high_threshold: float = 0.8
    moderate_threshold: float = 0.6
    low_threshold: float = 0.4

    # Transformation parameters
    max_dimension_ratio: float = 4.0      # Max ratio between source/target dimensions
    regularization_strength: float = 0.01
    interpolation_method: str = "linear"   # linear, cubic, spline

    # Performance optimization
    batch_mapping: bool = True
    parallel_computation: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 1024

    # Stability analysis
    stability_iterations: int = 100
    perturbation_strength: float = 0.01


class GeometricMapper:
    """Implements geometric mapping strategies based on tensor shapes."""

    def __init__(self, config: MappingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_geometric_mapping(self, source_param: torch.Tensor,
                               target_param: torch.Tensor,
                               param_name: str) -> WeightMapping:
        """Create geometric mapping between parameters."""
        source_shape = source_param.shape
        target_shape = target_param.shape

        # Compute geometric compatibility
        compatibility = self._compute_geometric_compatibility(source_shape, target_shape)

        # Determine mapping quality
        quality = self._determine_quality(compatibility)

        # Create transformation matrix if applicable
        transformation_matrix = None
        if self._can_create_transformation(source_shape, target_shape):
            transformation_matrix = self._create_transformation_matrix(source_shape, target_shape)

        mapping = WeightMapping(
            mapping_id=f"geometric_{param_name}_{time.time()}",
            source_component=param_name,
            target_component=param_name,
            mapping_strategy=MappingStrategy.GEOMETRIC_MAPPING,
            quality=quality,
            compatibility_score=compatibility,
            geometric_similarity=compatibility,
            semantic_similarity=0.5,  # Placeholder for geometric mapping
            transformation_matrix=transformation_matrix,
            preservation_score=compatibility,
            transfer_efficiency=self._compute_transfer_efficiency(source_shape, target_shape),
            stability_score=compatibility * 0.9,  # Geometric mappings are generally stable
            mapping_timestamp=time.time(),
            confidence=compatibility
        )

        return mapping

    def _compute_geometric_compatibility(self, source_shape: Tuple[int, ...],
                                       target_shape: Tuple[int, ...]) -> float:
        """Compute geometric compatibility between tensor shapes."""
        if source_shape == target_shape:
            return 1.0  # Perfect match

        # Check dimensionality compatibility
        if len(source_shape) != len(target_shape):
            if abs(len(source_shape) - len(target_shape)) == 1:
                # Can potentially squeeze/unsqueeze
                compatibility = 0.7
            else:
                compatibility = 0.3
        else:
            # Same dimensionality, check size compatibility
            dimension_ratios = []
            for s_dim, t_dim in zip(source_shape, target_shape):
                if s_dim == 0 or t_dim == 0:
                    ratio = 0.0
                else:
                    ratio = min(s_dim, t_dim) / max(s_dim, t_dim)
                dimension_ratios.append(ratio)

            compatibility = np.mean(dimension_ratios)

            # Penalize large dimension mismatches
            max_ratio = max(max(source_shape) / max(target_shape),
                          max(target_shape) / max(source_shape))

            if max_ratio > self.config.max_dimension_ratio:
                compatibility *= 0.5

        return float(compatibility)

    def _determine_quality(self, compatibility: float) -> MappingQuality:
        """Determine mapping quality based on compatibility score."""
        if compatibility >= self.config.perfect_threshold:
            return MappingQuality.PERFECT
        elif compatibility >= self.config.high_threshold:
            return MappingQuality.HIGH
        elif compatibility >= self.config.moderate_threshold:
            return MappingQuality.MODERATE
        elif compatibility >= self.config.low_threshold:
            return MappingQuality.LOW
        else:
            return MappingQuality.INCOMPATIBLE

    def _can_create_transformation(self, source_shape: Tuple[int, ...],
                                 target_shape: Tuple[int, ...]) -> bool:
        """Check if transformation matrix can be created."""
        # For now, support 2D matrices (linear layers)
        return (len(source_shape) == 2 and len(target_shape) == 2 and
                max(source_shape) <= 10000 and max(target_shape) <= 10000)

    def _create_transformation_matrix(self, source_shape: Tuple[int, ...],
                                    target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Create transformation matrix for geometric mapping."""
        if len(source_shape) == 2 and len(target_shape) == 2:
            s_rows, s_cols = source_shape
            t_rows, t_cols = target_shape

            # Create transformation using interpolation or projection
            if s_rows == t_rows and s_cols == t_cols:
                return torch.eye(s_rows)  # Identity transformation

            elif s_rows <= t_rows and s_cols <= t_cols:
                # Expansion case - use interpolation
                transform = torch.zeros(t_rows, s_rows)
                row_indices = torch.linspace(0, s_rows - 1, t_rows).long()
                for i, src_idx in enumerate(row_indices):
                    transform[i, src_idx] = 1.0
                return transform

            else:
                # Contraction case - use averaging
                transform = torch.zeros(t_rows, s_rows)
                for i in range(t_rows):
                    start_idx = int(i * s_rows / t_rows)
                    end_idx = int((i + 1) * s_rows / t_rows)
                    transform[i, start_idx:end_idx] = 1.0 / (end_idx - start_idx)
                return transform

        return torch.eye(min(source_shape[0], target_shape[0]))

    def _compute_transfer_efficiency(self, source_shape: Tuple[int, ...],
                                   target_shape: Tuple[int, ...]) -> float:
        """Compute transfer efficiency for geometric mapping."""
        source_size = np.prod(source_shape)
        target_size = np.prod(target_shape)

        # Efficiency based on parameter utilization
        utilization = min(source_size, target_size) / max(source_size, target_size)

        # Adjust for shape compatibility
        shape_compatibility = self._compute_geometric_compatibility(source_shape, target_shape)

        efficiency = (utilization + shape_compatibility) / 2.0
        return float(efficiency)


class SemanticMapper:
    """Implements semantic mapping based on functional similarity."""

    def __init__(self, config: MappingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_semantic_mapping(self, source_model: nn.Module, target_model: nn.Module,
                              source_name: str, target_name: str) -> WeightMapping:
        """Create semantic mapping between model components."""
        source_module = self._get_module_by_name(source_model, source_name)
        target_module = self._get_module_by_name(target_model, target_name)

        if source_module is None or target_module is None:
            return self._create_incompatible_mapping(source_name, target_name)

        # Compute semantic similarity
        semantic_similarity = self._compute_semantic_similarity(source_module, target_module)

        # Analyze functional compatibility
        functional_compatibility = self._analyze_functional_compatibility(source_module, target_module)

        # Overall compatibility
        compatibility = (semantic_similarity + functional_compatibility) / 2.0

        quality = self._determine_quality(compatibility)

        mapping = WeightMapping(
            mapping_id=f"semantic_{source_name}_{target_name}_{time.time()}",
            source_component=source_name,
            target_component=target_name,
            mapping_strategy=MappingStrategy.SEMANTIC_MAPPING,
            quality=quality,
            compatibility_score=compatibility,
            geometric_similarity=functional_compatibility,
            semantic_similarity=semantic_similarity,
            preservation_score=semantic_similarity,
            transfer_efficiency=compatibility * 0.8,  # Semantic mappings may need adaptation
            stability_score=semantic_similarity * 0.85,
            mapping_timestamp=time.time(),
            confidence=compatibility
        )

        return mapping

    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by name from model."""
        try:
            for module_name, module in model.named_modules():
                if module_name == name:
                    return module
        except Exception as e:
            self.logger.warning(f"Failed to get module {name}: {e}")
        return None

    def _compute_semantic_similarity(self, source_module: nn.Module,
                                   target_module: nn.Module) -> float:
        """Compute semantic similarity between modules."""
        # Check module type similarity
        if type(source_module) == type(target_module):
            type_similarity = 1.0
        elif self._are_compatible_types(source_module, target_module):
            type_similarity = 0.8
        else:
            type_similarity = 0.3

        # Check parameter similarity
        param_similarity = self._compute_parameter_similarity(source_module, target_module)

        # Combine similarities
        semantic_similarity = (type_similarity + param_similarity) / 2.0
        return float(semantic_similarity)

    def _are_compatible_types(self, module1: nn.Module, module2: nn.Module) -> bool:
        """Check if two module types are compatible."""
        compatible_groups = [
            {nn.Linear},
            {nn.Conv1d, nn.Conv2d, nn.Conv3d},
            {nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU},
            {nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d},
            {nn.LSTM, nn.GRU}
        ]

        type1 = type(module1)
        type2 = type(module2)

        for group in compatible_groups:
            if type1 in group and type2 in group:
                return True

        return False

    def _compute_parameter_similarity(self, module1: nn.Module, module2: nn.Module) -> float:
        """Compute parameter similarity between modules."""
        params1 = list(module1.parameters())
        params2 = list(module2.parameters())

        if not params1 or not params2:
            return 0.5  # Neutral similarity for modules without parameters

        # Compare parameter shapes
        shape_similarities = []
        for p1, p2 in zip(params1, params2):
            shape_sim = self._compute_shape_similarity(p1.shape, p2.shape)
            shape_similarities.append(shape_sim)

        # Pad with zeros if different number of parameters
        if len(params1) != len(params2):
            diff = abs(len(params1) - len(params2))
            shape_similarities.extend([0.0] * diff)

        return np.mean(shape_similarities) if shape_similarities else 0.0

    def _compute_shape_similarity(self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> float:
        """Compute similarity between parameter shapes."""
        if shape1 == shape2:
            return 1.0

        if len(shape1) != len(shape2):
            return 0.3

        ratios = []
        for dim1, dim2 in zip(shape1, shape2):
            if dim1 == 0 or dim2 == 0:
                ratios.append(0.0)
            else:
                ratio = min(dim1, dim2) / max(dim1, dim2)
                ratios.append(ratio)

        return np.mean(ratios)

    def _analyze_functional_compatibility(self, module1: nn.Module, module2: nn.Module) -> float:
        """Analyze functional compatibility between modules."""
        # This is a simplified analysis
        # In practice, would involve analyzing activation patterns, gradients, etc.

        if type(module1) == type(module2):
            return 1.0

        if self._are_compatible_types(module1, module2):
            return 0.8

        # Check if both are trainable
        trainable1 = any(p.requires_grad for p in module1.parameters())
        trainable2 = any(p.requires_grad for p in module2.parameters())

        if trainable1 == trainable2:
            return 0.5
        else:
            return 0.3

    def _determine_quality(self, compatibility: float) -> MappingQuality:
        """Determine mapping quality based on compatibility."""
        if compatibility >= self.config.perfect_threshold:
            return MappingQuality.PERFECT
        elif compatibility >= self.config.high_threshold:
            return MappingQuality.HIGH
        elif compatibility >= self.config.moderate_threshold:
            return MappingQuality.MODERATE
        elif compatibility >= self.config.low_threshold:
            return MappingQuality.LOW
        else:
            return MappingQuality.INCOMPATIBLE

    def _create_incompatible_mapping(self, source_name: str, target_name: str) -> WeightMapping:
        """Create mapping for incompatible components."""
        return WeightMapping(
            mapping_id=f"incompatible_{source_name}_{target_name}_{time.time()}",
            source_component=source_name,
            target_component=target_name,
            mapping_strategy=MappingStrategy.SEMANTIC_MAPPING,
            quality=MappingQuality.INCOMPATIBLE,
            compatibility_score=0.0,
            geometric_similarity=0.0,
            semantic_similarity=0.0,
            preservation_score=0.0,
            transfer_efficiency=0.0,
            stability_score=0.0,
            mapping_timestamp=time.time(),
            confidence=0.0
        )


class InterpolationMapper:
    """Implements interpolation-based mapping strategies."""

    def __init__(self, config: MappingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_interpolation_mapping(self, source_param: torch.Tensor,
                                   target_param: torch.Tensor,
                                   param_name: str) -> WeightMapping:
        """Create interpolation mapping between parameters."""
        # Compute interpolation weights
        interpolation_weights = self._compute_interpolation_weights(source_param, target_param)

        # Assess interpolation quality
        quality_score = self._assess_interpolation_quality(source_param, target_param)

        quality = self._determine_quality(quality_score)

        mapping = WeightMapping(
            mapping_id=f"interpolation_{param_name}_{time.time()}",
            source_component=param_name,
            target_component=param_name,
            mapping_strategy=MappingStrategy.INTERPOLATION_MAPPING,
            quality=quality,
            compatibility_score=quality_score,
            geometric_similarity=quality_score,
            semantic_similarity=quality_score * 0.9,  # Slightly lower for interpolation
            interpolation_weights=interpolation_weights,
            preservation_score=quality_score,
            transfer_efficiency=quality_score * 0.85,
            stability_score=quality_score * 0.8,  # Interpolation can be less stable
            mapping_timestamp=time.time(),
            confidence=quality_score
        )

        return mapping

    def _compute_interpolation_weights(self, source_param: torch.Tensor,
                                     target_param: torch.Tensor) -> List[float]:
        """Compute interpolation weights for mapping."""
        if source_param.shape == target_param.shape:
            # Direct interpolation possible
            return [0.5, 0.5]  # Simple average

        # For different shapes, compute optimal interpolation weights
        # This is a simplified implementation
        source_norm = torch.norm(source_param).item()
        target_norm = torch.norm(target_param).item()

        if source_norm + target_norm == 0:
            return [0.5, 0.5]

        # Weight based on magnitudes
        total_norm = source_norm + target_norm
        source_weight = source_norm / total_norm
        target_weight = target_norm / total_norm

        return [float(source_weight), float(target_weight)]

    def _assess_interpolation_quality(self, source_param: torch.Tensor,
                                    target_param: torch.Tensor) -> float:
        """Assess quality of interpolation mapping."""
        # Shape compatibility
        if source_param.shape == target_param.shape:
            shape_compatibility = 1.0
        else:
            # Compute shape compatibility for different shapes
            source_size = source_param.numel()
            target_size = target_param.numel()
            shape_compatibility = min(source_size, target_size) / max(source_size, target_size)

        # Statistical similarity
        if source_param.shape == target_param.shape:
            source_stats = self._compute_tensor_stats(source_param)
            target_stats = self._compute_tensor_stats(target_param)
            stat_similarity = self._compute_stat_similarity(source_stats, target_stats)
        else:
            stat_similarity = 0.5  # Neutral for different shapes

        # Overall quality
        quality = (shape_compatibility + stat_similarity) / 2.0
        return float(quality)

    def _compute_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of tensor."""
        flat_tensor = tensor.flatten()
        return {
            'mean': float(torch.mean(flat_tensor)),
            'std': float(torch.std(flat_tensor)),
            'min': float(torch.min(flat_tensor)),
            'max': float(torch.max(flat_tensor))
        }

    def _compute_stat_similarity(self, stats1: Dict[str, float],
                               stats2: Dict[str, float]) -> float:
        """Compute similarity between tensor statistics."""
        similarities = []

        for key in ['mean', 'std', 'min', 'max']:
            if key in stats1 and key in stats2:
                val1, val2 = stats1[key], stats2[key]
                if abs(val1) + abs(val2) < 1e-10:
                    similarities.append(1.0)
                else:
                    similarity = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2))
                    similarities.append(max(0.0, similarity))

        return np.mean(similarities) if similarities else 0.0

    def _determine_quality(self, quality_score: float) -> MappingQuality:
        """Determine mapping quality."""
        if quality_score >= self.config.perfect_threshold:
            return MappingQuality.PERFECT
        elif quality_score >= self.config.high_threshold:
            return MappingQuality.HIGH
        elif quality_score >= self.config.moderate_threshold:
            return MappingQuality.MODERATE
        elif quality_score >= self.config.low_threshold:
            return MappingQuality.LOW
        else:
            return MappingQuality.INCOMPATIBLE


class ArchitectureMapper:
    """
    Main mapper for cross-architecture weight mapping.

    Provides comprehensive mapping capabilities between different neural
    network architectures using multiple strategies and quality assessment.
    """

    def __init__(self, config: Optional[MappingConfig] = None):
        self.config = config or MappingConfig()
        self.geometric_mapper = GeometricMapper(self.config)
        self.semantic_mapper = SemanticMapper(self.config)
        self.interpolation_mapper = InterpolationMapper(self.config)
        self.mapping_history = []
        self.logger = logging.getLogger(__name__)

    def create_architecture_mapping(self, source_model: nn.Module, target_model: nn.Module,
                                  mapping_strategy: MappingStrategy = MappingStrategy.SEMANTIC_MAPPING) -> List[WeightMapping]:
        """Create comprehensive mapping between architectures."""
        try:
            start_time = time.time()

            self.logger.info(f"Creating architecture mapping using {mapping_strategy.value}")

            mappings = []

            if mapping_strategy == MappingStrategy.GEOMETRIC_MAPPING:
                mappings = self._create_geometric_mappings(source_model, target_model)
            elif mapping_strategy == MappingStrategy.SEMANTIC_MAPPING:
                mappings = self._create_semantic_mappings(source_model, target_model)
            elif mapping_strategy == MappingStrategy.INTERPOLATION_MAPPING:
                mappings = self._create_interpolation_mappings(source_model, target_model)
            else:
                # Default to combined approach
                mappings = self._create_combined_mappings(source_model, target_model)

            # Filter mappings by quality
            quality_mappings = [m for m in mappings if m.compatibility_score >= self.config.compatibility_threshold]

            # Store mapping results
            mapping_result = {
                'timestamp': time.time(),
                'mapping_strategy': mapping_strategy.value,
                'total_mappings': len(mappings),
                'quality_mappings': len(quality_mappings),
                'mappings': quality_mappings,
                'computation_time': time.time() - start_time
            }

            self.mapping_history.append(mapping_result)

            self.logger.info(f"Created {len(quality_mappings)} quality mappings in {time.time() - start_time:.2f}s")

            return quality_mappings

        except Exception as e:
            self.logger.error(f"Architecture mapping failed: {e}")
            return []

    def _create_geometric_mappings(self, source_model: nn.Module,
                                 target_model: nn.Module) -> List[WeightMapping]:
        """Create mappings using geometric strategy."""
        mappings = []

        source_params = dict(source_model.named_parameters())
        target_params = dict(target_model.named_parameters())

        # Map parameters with same names first
        common_names = set(source_params.keys()).intersection(set(target_params.keys()))

        for param_name in common_names:
            source_param = source_params[param_name]
            target_param = target_params[param_name]

            mapping = self.geometric_mapper.create_geometric_mapping(
                source_param, target_param, param_name
            )
            mappings.append(mapping)

        return mappings

    def _create_semantic_mappings(self, source_model: nn.Module,
                                target_model: nn.Module) -> List[WeightMapping]:
        """Create mappings using semantic strategy."""
        mappings = []

        source_modules = dict(source_model.named_modules())
        target_modules = dict(target_model.named_modules())

        # Find semantically similar modules
        for source_name, source_module in source_modules.items():
            if len(list(source_module.children())) == 0:  # Leaf modules only
                best_target = None
                best_similarity = 0.0

                for target_name, target_module in target_modules.items():
                    if len(list(target_module.children())) == 0:  # Leaf modules only
                        # Quick compatibility check
                        if self.semantic_mapper._are_compatible_types(source_module, target_module):
                            similarity = self.semantic_mapper._compute_semantic_similarity(
                                source_module, target_module
                            )

                            if similarity > best_similarity and similarity >= self.config.compatibility_threshold:
                                best_similarity = similarity
                                best_target = target_name

                if best_target:
                    mapping = self.semantic_mapper.create_semantic_mapping(
                        source_model, target_model, source_name, best_target
                    )
                    mappings.append(mapping)

        return mappings

    def _create_interpolation_mappings(self, source_model: nn.Module,
                                     target_model: nn.Module) -> List[WeightMapping]:
        """Create mappings using interpolation strategy."""
        mappings = []

        source_params = dict(source_model.named_parameters())
        target_params = dict(target_model.named_parameters())

        # Create interpolation mappings for all parameter pairs
        for source_name, source_param in source_params.items():
            for target_name, target_param in target_params.items():
                if source_name == target_name:  # Same parameter name
                    mapping = self.interpolation_mapper.create_interpolation_mapping(
                        source_param, target_param, source_name
                    )
                    mappings.append(mapping)
                    break

        return mappings

    def _create_combined_mappings(self, source_model: nn.Module,
                                target_model: nn.Module) -> List[WeightMapping]:
        """Create mappings using combined strategies."""
        # Start with semantic mappings
        semantic_mappings = self._create_semantic_mappings(source_model, target_model)

        # Add geometric mappings for unmapped components
        geometric_mappings = self._create_geometric_mappings(source_model, target_model)

        # Combine and deduplicate
        all_mappings = semantic_mappings + geometric_mappings
        unique_mappings = self._deduplicate_mappings(all_mappings)

        return unique_mappings

    def _deduplicate_mappings(self, mappings: List[WeightMapping]) -> List[WeightMapping]:
        """Remove duplicate mappings, keeping the best quality ones."""
        mapping_dict = {}

        for mapping in mappings:
            key = (mapping.source_component, mapping.target_component)

            if key not in mapping_dict or mapping.compatibility_score > mapping_dict[key].compatibility_score:
                mapping_dict[key] = mapping

        return list(mapping_dict.values())

    def apply_weight_mapping(self, source_model: nn.Module, target_model: nn.Module,
                           mappings: List[WeightMapping]) -> nn.Module:
        """Apply weight mappings to transfer weights from source to target."""
        try:
            import copy
            mapped_model = copy.deepcopy(target_model)

            for mapping in mappings:
                if mapping.quality == MappingQuality.INCOMPATIBLE:
                    continue

                # Get source and target parameters
                source_param = self._get_parameter_by_name(source_model, mapping.source_component)
                target_param = self._get_parameter_by_name(mapped_model, mapping.target_component)

                if source_param is None or target_param is None:
                    self.logger.warning(f"Could not find parameters for mapping: {mapping.source_component} -> {mapping.target_component}")
                    continue

                # Apply the mapping
                if mapping.transformation_matrix is not None:
                    # Use transformation matrix
                    transformed_weights = self._apply_transformation(source_param.data, mapping.transformation_matrix)
                    target_param.data = transformed_weights
                elif mapping.interpolation_weights:
                    # Use interpolation
                    if len(mapping.interpolation_weights) >= 2:
                        alpha = mapping.interpolation_weights[0]
                        if source_param.shape == target_param.shape:
                            target_param.data = alpha * source_param.data + (1 - alpha) * target_param.data
                else:
                    # Direct copy if shapes match
                    if source_param.shape == target_param.shape:
                        target_param.data = source_param.data.clone()

            return mapped_model

        except Exception as e:
            self.logger.error(f"Failed to apply weight mapping: {e}")
            return target_model

    def _get_parameter_by_name(self, model: nn.Module, param_name: str) -> Optional[torch.Tensor]:
        """Get parameter by name from model."""
        for name, param in model.named_parameters():
            if name == param_name:
                return param
        return None

    def _apply_transformation(self, source_weights: torch.Tensor,
                            transformation_matrix: torch.Tensor) -> torch.Tensor:
        """Apply transformation matrix to source weights."""
        try:
            if len(source_weights.shape) == 2:
                # Matrix multiplication for 2D weights
                transformed = torch.mm(transformation_matrix, source_weights)
            else:
                # For higher dimensional weights, apply to flattened version
                original_shape = source_weights.shape
                flattened = source_weights.view(source_weights.shape[0], -1)
                transformed = torch.mm(transformation_matrix, flattened)
                transformed = transformed.view(-1, *original_shape[1:])

            return transformed

        except Exception as e:
            self.logger.warning(f"Transformation application failed: {e}")
            return source_weights

    def evaluate_mapping_quality(self, mappings: List[WeightMapping]) -> Dict[str, Any]:
        """Evaluate overall quality of architecture mapping."""
        if not mappings:
            return {"message": "No mappings to evaluate"}

        # Quality distribution
        quality_counts = {}
        for mapping in mappings:
            quality = mapping.quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # Aggregate metrics
        compatibility_scores = [m.compatibility_score for m in mappings]
        efficiency_scores = [m.transfer_efficiency for m in mappings]
        stability_scores = [m.stability_score for m in mappings]

        return {
            "total_mappings": len(mappings),
            "quality_distribution": quality_counts,
            "average_compatibility": np.mean(compatibility_scores),
            "average_efficiency": np.mean(efficiency_scores),
            "average_stability": np.mean(stability_scores),
            "high_quality_mappings": len([m for m in mappings if m.quality in [MappingQuality.PERFECT, MappingQuality.HIGH]]),
            "mapping_strategies": list(set(m.mapping_strategy.value for m in mappings))
        }

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get summary of mapping history."""
        if not self.mapping_history:
            return {"message": "No mapping history available"}

        recent_mappings = self.mapping_history[-10:]

        total_mappings = sum(m['total_mappings'] for m in recent_mappings)
        quality_mappings = sum(m['quality_mappings'] for m in recent_mappings)
        computation_times = [m['computation_time'] for m in recent_mappings]

        return {
            "total_mapping_sessions": len(self.mapping_history),
            "recent_sessions": len(recent_mappings),
            "total_mappings_created": total_mappings,
            "quality_mappings_created": quality_mappings,
            "quality_ratio": quality_mappings / total_mappings if total_mappings > 0 else 0,
            "average_computation_time": np.mean(computation_times),
            "config": {
                "compatibility_threshold": self.config.compatibility_threshold,
                "perfect_threshold": self.config.perfect_threshold,
                "max_dimension_ratio": self.config.max_dimension_ratio
            }
        }

    def reset_mapping_history(self) -> None:
        """Reset mapping history."""
        self.mapping_history.clear()
        self.logger.info("Mapping history reset")