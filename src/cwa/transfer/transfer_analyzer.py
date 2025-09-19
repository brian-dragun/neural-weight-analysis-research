"""
Main Transfer Analysis Engine

Implements comprehensive transfer learning analysis to understand how critical
weight patterns and vulnerabilities transfer across different architectures,
domains, and tasks.

Key Features:
- Multi-architecture transfer pattern detection
- Critical weight transferability analysis
- Transfer success prediction
- Cross-domain vulnerability migration
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict
import hashlib


class TransferType(Enum):
    """Types of transfer learning scenarios."""
    ARCHITECTURE_TRANSFER = "architecture_transfer"  # Same domain, different architecture
    DOMAIN_TRANSFER = "domain_transfer"  # Same architecture, different domain
    TASK_TRANSFER = "task_transfer"  # Different task within same domain
    CROSS_MODAL_TRANSFER = "cross_modal_transfer"  # Different modalities
    HIERARCHICAL_TRANSFER = "hierarchical_transfer"  # Between model layers/depths


class TransferDirection(Enum):
    """Direction of transfer analysis."""
    SOURCE_TO_TARGET = "source_to_target"
    BIDIRECTIONAL = "bidirectional"
    TARGET_TO_SOURCE = "target_to_source"


@dataclass
class TransferPattern:
    """Container for transfer pattern analysis results."""
    pattern_id: str
    transfer_type: TransferType
    source_signature: str  # Fingerprint of source pattern
    target_signature: str  # Fingerprint of target pattern

    # Transfer characteristics
    transferability_score: float  # 0.0 to 1.0
    pattern_similarity: float  # Structural similarity
    semantic_alignment: float  # Semantic meaning preservation

    # Performance metrics
    transfer_efficiency: float  # How well the pattern transfers
    adaptation_complexity: float  # How much adaptation is needed
    stability_retention: float  # How stable the pattern remains

    # Pattern details
    source_components: List[str] = field(default_factory=list)
    target_components: List[str] = field(default_factory=list)
    mapping_strategy: str = ""

    # Analysis metadata
    confidence_score: float = 0.0
    analysis_timestamp: float = 0.0
    computational_cost: float = 0.0


@dataclass
class TransferConfig:
    """Configuration for transfer analysis."""
    # Analysis scope
    max_source_components: int = 50  # Limit for computational efficiency
    max_target_components: int = 50
    pattern_depth: int = 3  # How deep to analyze patterns

    # Similarity thresholds
    similarity_threshold: float = 0.7  # Minimum similarity for pattern matching
    transferability_threshold: float = 0.5  # Minimum score for viable transfer
    alignment_threshold: float = 0.6  # Semantic alignment threshold

    # Transfer metrics
    efficiency_weight: float = 0.4  # Weight for efficiency in scoring
    stability_weight: float = 0.3  # Weight for stability in scoring
    complexity_weight: float = 0.3  # Weight for complexity in scoring

    # Performance optimization
    use_caching: bool = True
    parallel_analysis: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 512

    # Pattern matching
    pattern_window_size: int = 10  # Window for pattern extraction
    overlap_tolerance: float = 0.1  # Allowed overlap in pattern matching


class PatternExtractor:
    """Extracts transferable patterns from neural networks."""

    def __init__(self, config: TransferConfig):
        self.config = config
        self.pattern_cache = {}
        self.logger = logging.getLogger(__name__)

    def extract_patterns(self, model: nn.Module, model_id: str) -> Dict[str, Any]:
        """Extract transferable patterns from a model."""
        if model_id in self.pattern_cache:
            return self.pattern_cache[model_id]

        patterns = {
            'structural_patterns': self._extract_structural_patterns(model),
            'weight_patterns': self._extract_weight_patterns(model),
            'activation_patterns': self._extract_activation_patterns(model),
            'gradient_patterns': self._extract_gradient_patterns(model),
            'connectivity_patterns': self._extract_connectivity_patterns(model)
        }

        if self.config.use_caching:
            self.pattern_cache[model_id] = patterns

        return patterns

    def _extract_structural_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract structural patterns from model architecture."""
        patterns = {
            'layer_types': [],
            'layer_sizes': [],
            'connections': [],
            'depth_profile': {}
        }

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = type(module).__name__
                patterns['layer_types'].append(module_type)

                # Extract size information
                if hasattr(module, 'weight'):
                    patterns['layer_sizes'].append(tuple(module.weight.shape))

                # Extract depth information
                depth = name.count('.')
                if depth not in patterns['depth_profile']:
                    patterns['depth_profile'][depth] = 0
                patterns['depth_profile'][depth] += 1

        return patterns

    def _extract_weight_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract weight-based patterns."""
        patterns = {
            'weight_distributions': {},
            'magnitude_patterns': {},
            'sparsity_patterns': {},
            'correlation_patterns': {}
        }

        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                weight_data = param.data.cpu().numpy()

                # Distribution characteristics
                patterns['weight_distributions'][name] = {
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'skewness': float(self._compute_skewness(weight_data)),
                    'kurtosis': float(self._compute_kurtosis(weight_data))
                }

                # Magnitude patterns
                patterns['magnitude_patterns'][name] = {
                    'l1_norm': float(np.linalg.norm(weight_data, ord=1)),
                    'l2_norm': float(np.linalg.norm(weight_data, ord=2)),
                    'max_magnitude': float(np.max(np.abs(weight_data))),
                    'magnitude_variance': float(np.var(np.abs(weight_data)))
                }

                # Sparsity patterns
                zero_threshold = 1e-6
                sparsity_ratio = np.sum(np.abs(weight_data) < zero_threshold) / weight_data.size
                patterns['sparsity_patterns'][name] = {
                    'sparsity_ratio': float(sparsity_ratio),
                    'structured_sparsity': self._compute_structured_sparsity(weight_data)
                }

        return patterns

    def _extract_activation_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract activation-based patterns."""
        patterns = {
            'activation_types': [],
            'activation_bounds': {},
            'nonlinearity_patterns': {}
        }

        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.GELU)):
                activation_type = type(module).__name__
                patterns['activation_types'].append(activation_type)

                # Extract activation-specific parameters
                if hasattr(module, 'negative_slope'):  # LeakyReLU
                    patterns['nonlinearity_patterns'][name] = {
                        'type': activation_type,
                        'negative_slope': float(module.negative_slope)
                    }
                else:
                    patterns['nonlinearity_patterns'][name] = {
                        'type': activation_type
                    }

        return patterns

    def _extract_gradient_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract gradient-based patterns."""
        patterns = {
            'gradient_flow': {},
            'parameter_sensitivity': {},
            'optimization_patterns': {}
        }

        # This would require actual gradient computation
        # For now, return placeholder structure
        for name, param in model.named_parameters():
            if param.requires_grad:
                patterns['parameter_sensitivity'][name] = {
                    'requires_grad': True,
                    'param_shape': list(param.shape),
                    'param_count': param.numel()
                }

        return patterns

    def _extract_connectivity_patterns(self, model: nn.Module) -> Dict[str, Any]:
        """Extract connectivity patterns between layers."""
        patterns = {
            'layer_connectivity': {},
            'skip_connections': [],
            'branching_patterns': {},
            'layer_hierarchy': {}
        }

        # Analyze module hierarchy
        for name, module in model.named_modules():
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
            if parent_name not in patterns['layer_hierarchy']:
                patterns['layer_hierarchy'][parent_name] = []
            patterns['layer_hierarchy'][parent_name].append(name)

        return patterns

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        try:
            from scipy.stats import skew
            return skew(data.flatten())
        except ImportError:
            # Fallback manual computation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            normalized = (data - mean) / std
            return np.mean(normalized**3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data.flatten())
        except ImportError:
            # Fallback manual computation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            normalized = (data - mean) / std
            return np.mean(normalized**4) - 3.0

    def _compute_structured_sparsity(self, weight_data: np.ndarray) -> Dict[str, float]:
        """Compute structured sparsity metrics."""
        # Block sparsity (simplified)
        block_size = min(4, min(weight_data.shape))

        structured_metrics = {
            'row_sparsity': 0.0,
            'column_sparsity': 0.0,
            'block_sparsity': 0.0
        }

        if weight_data.ndim >= 2:
            # Row sparsity
            row_norms = np.linalg.norm(weight_data, axis=1)
            structured_metrics['row_sparsity'] = float(np.sum(row_norms < 1e-6) / len(row_norms))

            # Column sparsity
            col_norms = np.linalg.norm(weight_data, axis=0)
            structured_metrics['column_sparsity'] = float(np.sum(col_norms < 1e-6) / len(col_norms))

        return structured_metrics


class PatternMatcher:
    """Matches patterns between source and target models."""

    def __init__(self, config: TransferConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def match_patterns(self, source_patterns: Dict[str, Any],
                      target_patterns: Dict[str, Any]) -> List[TransferPattern]:
        """Match patterns between source and target."""
        matches = []

        # Match structural patterns
        structural_matches = self._match_structural_patterns(
            source_patterns.get('structural_patterns', {}),
            target_patterns.get('structural_patterns', {})
        )
        matches.extend(structural_matches)

        # Match weight patterns
        weight_matches = self._match_weight_patterns(
            source_patterns.get('weight_patterns', {}),
            target_patterns.get('weight_patterns', {})
        )
        matches.extend(weight_matches)

        # Match activation patterns
        activation_matches = self._match_activation_patterns(
            source_patterns.get('activation_patterns', {}),
            target_patterns.get('activation_patterns', {})
        )
        matches.extend(activation_matches)

        return matches

    def _match_structural_patterns(self, source_struct: Dict[str, Any],
                                 target_struct: Dict[str, Any]) -> List[TransferPattern]:
        """Match structural patterns."""
        matches = []

        # Compare layer type sequences
        source_types = source_struct.get('layer_types', [])
        target_types = target_struct.get('layer_types', [])

        if source_types and target_types:
            similarity = self._compute_sequence_similarity(source_types, target_types)

            if similarity >= self.config.similarity_threshold:
                pattern = TransferPattern(
                    pattern_id=self._generate_pattern_id('structural', source_types, target_types),
                    transfer_type=TransferType.ARCHITECTURE_TRANSFER,
                    source_signature=str(hash(tuple(source_types))),
                    target_signature=str(hash(tuple(target_types))),
                    transferability_score=similarity,
                    pattern_similarity=similarity,
                    semantic_alignment=self._compute_semantic_alignment(source_types, target_types),
                    transfer_efficiency=similarity * 0.8,  # Structural patterns transfer well
                    adaptation_complexity=1.0 - similarity,
                    stability_retention=similarity,
                    source_components=source_types,
                    target_components=target_types,
                    mapping_strategy="structural_alignment",
                    confidence_score=similarity,
                    analysis_timestamp=time.time()
                )
                matches.append(pattern)

        return matches

    def _match_weight_patterns(self, source_weights: Dict[str, Any],
                             target_weights: Dict[str, Any]) -> List[TransferPattern]:
        """Match weight patterns."""
        matches = []

        source_distributions = source_weights.get('weight_distributions', {})
        target_distributions = target_weights.get('weight_distributions', {})

        # Find compatible weight layers
        for source_layer, source_dist in source_distributions.items():
            best_match = None
            best_similarity = 0.0

            for target_layer, target_dist in target_distributions.items():
                similarity = self._compute_distribution_similarity(source_dist, target_dist)

                if similarity > best_similarity and similarity >= self.config.similarity_threshold:
                    best_similarity = similarity
                    best_match = target_layer

            if best_match:
                pattern = TransferPattern(
                    pattern_id=self._generate_pattern_id('weight', source_layer, best_match),
                    transfer_type=TransferType.ARCHITECTURE_TRANSFER,
                    source_signature=str(hash(str(source_dist))),
                    target_signature=str(hash(str(target_distributions[best_match]))),
                    transferability_score=best_similarity,
                    pattern_similarity=best_similarity,
                    semantic_alignment=self._compute_weight_semantic_alignment(source_dist, target_distributions[best_match]),
                    transfer_efficiency=best_similarity * 0.6,  # Weight patterns may need adaptation
                    adaptation_complexity=1.2 - best_similarity,
                    stability_retention=best_similarity * 0.9,
                    source_components=[source_layer],
                    target_components=[best_match],
                    mapping_strategy="distribution_matching",
                    confidence_score=best_similarity,
                    analysis_timestamp=time.time()
                )
                matches.append(pattern)

        return matches

    def _match_activation_patterns(self, source_activations: Dict[str, Any],
                                 target_activations: Dict[str, Any]) -> List[TransferPattern]:
        """Match activation patterns."""
        matches = []

        source_types = source_activations.get('activation_types', [])
        target_types = target_activations.get('activation_types', [])

        if source_types and target_types:
            similarity = self._compute_sequence_similarity(source_types, target_types)

            if similarity >= self.config.similarity_threshold:
                pattern = TransferPattern(
                    pattern_id=self._generate_pattern_id('activation', source_types, target_types),
                    transfer_type=TransferType.ARCHITECTURE_TRANSFER,
                    source_signature=str(hash(tuple(source_types))),
                    target_signature=str(hash(tuple(target_types))),
                    transferability_score=similarity,
                    pattern_similarity=similarity,
                    semantic_alignment=1.0,  # Activation functions have clear semantic meaning
                    transfer_efficiency=similarity,
                    adaptation_complexity=0.5 - similarity * 0.3,  # Activations are easy to adapt
                    stability_retention=similarity,
                    source_components=source_types,
                    target_components=target_types,
                    mapping_strategy="activation_mapping",
                    confidence_score=similarity,
                    analysis_timestamp=time.time()
                )
                matches.append(pattern)

        return matches

    def _compute_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Compute similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0

        # Use Jaccard similarity for sets
        set1 = set(seq1)
        set2 = set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def _compute_distribution_similarity(self, dist1: Dict[str, float],
                                       dist2: Dict[str, float]) -> float:
        """Compute similarity between weight distributions."""
        keys = ['mean', 'std', 'skewness', 'kurtosis']

        similarities = []
        for key in keys:
            if key in dist1 and key in dist2:
                # Normalize and compute similarity
                val1 = dist1[key]
                val2 = dist2[key]

                # Avoid division by zero
                if abs(val1) + abs(val2) < 1e-10:
                    similarities.append(1.0)
                else:
                    similarity = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2))
                    similarities.append(max(0.0, similarity))

        return np.mean(similarities) if similarities else 0.0

    def _compute_semantic_alignment(self, source_components: List[str],
                                  target_components: List[str]) -> float:
        """Compute semantic alignment between components."""
        # Simplified semantic alignment based on component name similarity
        if not source_components or not target_components:
            return 0.0

        alignments = []
        for src_comp in source_components:
            best_alignment = 0.0
            for tgt_comp in target_components:
                # Simple string similarity
                alignment = self._string_similarity(src_comp, tgt_comp)
                best_alignment = max(best_alignment, alignment)
            alignments.append(best_alignment)

        return np.mean(alignments)

    def _compute_weight_semantic_alignment(self, source_dist: Dict[str, float],
                                         target_dist: Dict[str, float]) -> float:
        """Compute semantic alignment for weight distributions."""
        # Weight distributions have implicit semantic meaning
        # Similar distributions suggest similar learned representations
        return self._compute_distribution_similarity(source_dist, target_dist)

    def _string_similarity(self, str1: str, str2: str) -> float:
        """Compute string similarity using simple overlap metric."""
        if not str1 or not str2:
            return 0.0

        # Convert to lowercase and split into tokens
        tokens1 = set(str1.lower().split('_'))
        tokens2 = set(str2.lower().split('_'))

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def _generate_pattern_id(self, pattern_type: str, source: Any, target: Any) -> str:
        """Generate unique pattern ID."""
        content = f"{pattern_type}_{str(source)}_{str(target)}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class TransferAnalyzer:
    """
    Main analyzer for cross-architecture transfer analysis.

    Provides comprehensive analysis of how patterns, vulnerabilities,
    and critical configurations transfer between different neural architectures.
    """

    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
        self.pattern_extractor = PatternExtractor(self.config)
        self.pattern_matcher = PatternMatcher(self.config)
        self.analysis_history = []
        self.model_patterns = {}  # Cache for extracted patterns
        self.logger = logging.getLogger(__name__)

    def analyze_transfer_potential(self, source_model: nn.Module, target_model: nn.Module,
                                 source_id: str = "source", target_id: str = "target") -> List[TransferPattern]:
        """Analyze transfer potential between source and target models."""
        try:
            start_time = time.time()

            self.logger.info(f"Analyzing transfer potential: {source_id} -> {target_id}")

            # Extract patterns from both models
            source_patterns = self.pattern_extractor.extract_patterns(source_model, source_id)
            target_patterns = self.pattern_extractor.extract_patterns(target_model, target_id)

            # Store patterns for future analysis
            self.model_patterns[source_id] = source_patterns
            self.model_patterns[target_id] = target_patterns

            # Match patterns between models
            transfer_patterns = self.pattern_matcher.match_patterns(source_patterns, target_patterns)

            # Compute overall transfer metrics
            if transfer_patterns:
                self._compute_transfer_metrics(transfer_patterns)

            # Store analysis results
            analysis_result = {
                'timestamp': time.time(),
                'source_id': source_id,
                'target_id': target_id,
                'transfer_patterns': transfer_patterns,
                'source_patterns': source_patterns,
                'target_patterns': target_patterns,
                'computation_time': time.time() - start_time
            }

            self.analysis_history.append(analysis_result)

            self.logger.info(f"Found {len(transfer_patterns)} transferable patterns in {time.time() - start_time:.2f}s")

            return transfer_patterns

        except Exception as e:
            self.logger.error(f"Transfer analysis failed: {e}")
            return []

    def analyze_multi_architecture_transfer(self, models: Dict[str, nn.Module]) -> Dict[str, List[TransferPattern]]:
        """Analyze transfer patterns across multiple architectures."""
        transfer_matrix = {}
        model_ids = list(models.keys())

        # Analyze all pairwise transfers
        for i, source_id in enumerate(model_ids):
            for j, target_id in enumerate(model_ids):
                if i != j:  # Don't analyze self-transfer
                    transfer_key = f"{source_id}_to_{target_id}"
                    patterns = self.analyze_transfer_potential(
                        models[source_id], models[target_id], source_id, target_id
                    )
                    transfer_matrix[transfer_key] = patterns

        return transfer_matrix

    def predict_transfer_success(self, source_model: nn.Module, target_model: nn.Module,
                               transfer_scenario: TransferType = TransferType.ARCHITECTURE_TRANSFER) -> Dict[str, float]:
        """Predict the success of transfer learning between models."""
        patterns = self.analyze_transfer_potential(source_model, target_model)

        if not patterns:
            return {
                'transfer_success_probability': 0.0,
                'expected_performance_retention': 0.0,
                'adaptation_difficulty': 1.0,
                'confidence': 0.0
            }

        # Aggregate pattern metrics
        transferability_scores = [p.transferability_score for p in patterns]
        efficiency_scores = [p.transfer_efficiency for p in patterns]
        stability_scores = [p.stability_retention for p in patterns]
        complexity_scores = [p.adaptation_complexity for p in patterns]

        # Weighted combination
        success_probability = (
            np.mean(transferability_scores) * 0.4 +
            np.mean(efficiency_scores) * 0.3 +
            np.mean(stability_scores) * 0.3
        )

        performance_retention = np.mean(efficiency_scores)
        adaptation_difficulty = np.mean(complexity_scores)
        confidence = min(1.0, len(patterns) / 10.0)  # More patterns = higher confidence

        return {
            'transfer_success_probability': float(success_probability),
            'expected_performance_retention': float(performance_retention),
            'adaptation_difficulty': float(adaptation_difficulty),
            'confidence': float(confidence),
            'num_transferable_patterns': len(patterns)
        }

    def _compute_transfer_metrics(self, patterns: List[TransferPattern]) -> None:
        """Compute and update transfer metrics for patterns."""
        for pattern in patterns:
            # Update computational cost
            pattern.computational_cost = 0.1  # Placeholder

            # Refine transferability score based on multiple factors
            transferability = (
                pattern.pattern_similarity * self.config.efficiency_weight +
                pattern.semantic_alignment * self.config.stability_weight +
                (1.0 - pattern.adaptation_complexity) * self.config.complexity_weight
            )
            pattern.transferability_score = min(1.0, transferability)

            # Update confidence based on pattern characteristics
            pattern.confidence_score = min(1.0, pattern.transferability_score * 1.2)

    def get_transfer_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of transfer analyses."""
        if not self.analysis_history:
            return {"message": "No transfer analysis history available"}

        recent_analyses = self.analysis_history[-10:]

        # Aggregate statistics
        total_patterns = sum(len(a['transfer_patterns']) for a in recent_analyses)
        transferability_scores = []
        efficiency_scores = []

        for analysis in recent_analyses:
            for pattern in analysis['transfer_patterns']:
                transferability_scores.append(pattern.transferability_score)
                efficiency_scores.append(pattern.transfer_efficiency)

        # Transfer type distribution
        transfer_types = {}
        for analysis in recent_analyses:
            for pattern in analysis['transfer_patterns']:
                transfer_type = pattern.transfer_type.value
                transfer_types[transfer_type] = transfer_types.get(transfer_type, 0) + 1

        return {
            "total_analyses": len(self.analysis_history),
            "recent_analyses": len(recent_analyses),
            "total_patterns_found": total_patterns,
            "average_patterns_per_analysis": total_patterns / len(recent_analyses) if recent_analyses else 0,
            "average_transferability": np.mean(transferability_scores) if transferability_scores else 0,
            "average_efficiency": np.mean(efficiency_scores) if efficiency_scores else 0,
            "transfer_type_distribution": transfer_types,
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "transferability_threshold": self.config.transferability_threshold,
                "max_components": self.config.max_source_components,
                "pattern_depth": self.config.pattern_depth
            }
        }

    def export_transfer_knowledge(self, filepath: str) -> None:
        """Export accumulated transfer knowledge to file."""
        transfer_knowledge = {
            'config': self.config.__dict__,
            'model_patterns': self.model_patterns,
            'analysis_history': []
        }

        # Convert TransferPattern objects to serializable format
        for analysis in self.analysis_history:
            serializable_analysis = {
                'timestamp': analysis['timestamp'],
                'source_id': analysis['source_id'],
                'target_id': analysis['target_id'],
                'computation_time': analysis['computation_time'],
                'transfer_patterns': []
            }

            for pattern in analysis['transfer_patterns']:
                pattern_dict = {
                    'pattern_id': pattern.pattern_id,
                    'transfer_type': pattern.transfer_type.value,
                    'transferability_score': pattern.transferability_score,
                    'pattern_similarity': pattern.pattern_similarity,
                    'semantic_alignment': pattern.semantic_alignment,
                    'transfer_efficiency': pattern.transfer_efficiency,
                    'adaptation_complexity': pattern.adaptation_complexity,
                    'stability_retention': pattern.stability_retention,
                    'mapping_strategy': pattern.mapping_strategy,
                    'confidence_score': pattern.confidence_score
                }
                serializable_analysis['transfer_patterns'].append(pattern_dict)

            transfer_knowledge['analysis_history'].append(serializable_analysis)

        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(transfer_knowledge, f, indent=2)

        self.logger.info(f"Transfer knowledge exported to {filepath}")

    def reset_analysis_history(self) -> None:
        """Reset analysis history and caches."""
        self.analysis_history.clear()
        self.model_patterns.clear()
        self.pattern_extractor.pattern_cache.clear()
        self.logger.info("Transfer analysis history reset")