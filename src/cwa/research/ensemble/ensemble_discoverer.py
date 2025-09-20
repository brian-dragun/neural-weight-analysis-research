"""Ensemble-based super weight discovery coordinator."""

from typing import Dict, List, Tuple, Any
import torch
import numpy as np
import logging
from dataclasses import dataclass

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


class EnsembleWeightDiscoverer:
    """
    Coordinates multiple discovery methods for robust super weight detection.

    Combines results from different discovery approaches using weighted voting
    to improve confidence and reduce false positives.
    """

    def __init__(self):
        """Initialize ensemble discoverer with default configuration."""
        self.ensemble_config = {
            "methods": [
                "activation_outliers",
                "causal_intervention",
                "information_bottleneck",
                "spectral_anomaly"
            ],
            "weights": {
                "activation_outliers": 0.3,
                "causal_intervention": 0.3,
                "information_bottleneck": 0.2,
                "spectral_anomaly": 0.2
            },
            "confidence_threshold": 0.7,
            "agreement_threshold": 0.6  # Require 60% of methods to agree
        }

    def discover_super_weights_ensemble(
        self,
        activation_data: Dict[str, Any],
        sensitivity_data: Dict[str, Any],
        top_k_percent: float,
        target_layers: List[str],
        sample_inputs: torch.Tensor
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

        # Import discovery methods
        from .discovery_methods import (
            ActivationOutlierMethod,
            CausalInterventionMethod,
            InformationBottleneckMethod,
            SpectralAnomalyMethod
        )

        # Method 1: Activation outliers
        logger.info("Running Method 1: Activation outlier detection")
        activation_method = ActivationOutlierMethod()
        activation_candidates = activation_method.discover(
            activation_data, sensitivity_data, top_k_percent
        )

        # Method 2: Causal intervention analysis
        logger.info("Running Method 2: Causal intervention analysis")
        causal_method = CausalInterventionMethod()
        causal_candidates = causal_method.discover(
            sample_inputs, target_layers, top_k_percent
        )

        # Method 3: Information bottleneck detection
        logger.info("Running Method 3: Information bottleneck detection")
        bottleneck_method = InformationBottleneckMethod()
        bottleneck_candidates = bottleneck_method.discover(
            sample_inputs, target_layers, top_k_percent
        )

        # Method 4: Spectral anomaly detection
        logger.info("Running Method 4: Spectral anomaly detection")
        spectral_method = SpectralAnomalyMethod()
        spectral_candidates = spectral_method.discover(
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