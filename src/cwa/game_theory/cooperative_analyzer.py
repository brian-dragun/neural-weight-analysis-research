"""
Cooperative Game Analysis for Neural Weight Coalitions

Implements cooperative game theory to analyze coalition formation
among neural network weights and identify stable cooperative structures.

Key Features:
- Shapley value computation for weight importance
- Coalition structure analysis
- Core solution concepts
- Stability assessment of weight coalitions
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict


class CoalitionType(Enum):
    """Types of coalitions in cooperative games."""
    GRAND_COALITION = "grand_coalition"
    BLOCKING_COALITION = "blocking_coalition"
    STABLE_COALITION = "stable_coalition"
    MINIMAL_COALITION = "minimal_coalition"
    MAXIMAL_COALITION = "maximal_coalition"


class SolutionConcept(Enum):
    """Solution concepts for cooperative games."""
    CORE = "core"
    SHAPLEY_VALUE = "shapley_value"
    NUCLEOLUS = "nucleolus"
    STABLE_SET = "stable_set"
    BARGAINING_SET = "bargaining_set"


@dataclass
class CoalitionStructure:
    """Represents a coalition structure in cooperative game."""
    coalitions: List[FrozenSet[str]]  # Partition of players
    coalition_values: Dict[FrozenSet[str], float]  # Coalition value function
    stability_score: float  # 0.0 to 1.0
    efficiency_score: float  # Total value / grand coalition value

    # Shapley values for individual players
    shapley_values: Dict[str, float] = field(default_factory=dict)

    # Core analysis
    is_in_core: bool = False
    core_constraints_satisfied: int = 0
    total_core_constraints: int = 0

    # Coalition formation details
    formation_process: List[str] = field(default_factory=list)
    blocking_coalitions: List[FrozenSet[str]] = field(default_factory=list)


@dataclass
class CooperativeGameConfig:
    """Configuration for cooperative game analysis."""
    # Coalition analysis
    max_coalition_size: int = 10  # Limit for computational tractability
    min_coalition_size: int = 2
    sample_coalitions: bool = True  # Sample instead of exhaustive enumeration
    coalition_sample_size: int = 100

    # Shapley value computation
    compute_shapley_values: bool = True
    shapley_sample_size: int = 1000  # For approximate computation
    exact_shapley_threshold: int = 8  # Use exact computation for <= 8 players

    # Core analysis
    analyze_core: bool = True
    core_epsilon: float = 1e-6  # Tolerance for core membership

    # Performance optimization
    parallel_computation: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 256

    # Stability analysis
    stability_iterations: int = 100
    convergence_threshold: float = 1e-4


class CoalitionValueFunction:
    """Computes coalition values for cooperative game analysis."""

    def __init__(self, model: nn.Module, config: CooperativeGameConfig):
        self.model = model
        self.config = config
        self.baseline_performance = None
        self._value_cache = {}
        self.logger = logging.getLogger(__name__)

    def compute_coalition_value(self, coalition: FrozenSet[str],
                              inputs: torch.Tensor) -> float:
        """Compute the value of a coalition of players (weight groups)."""
        if coalition in self._value_cache:
            return self._value_cache[coalition]

        try:
            # Create modified model with only coalition members active
            modified_model = self._create_coalition_model(coalition)

            # Compute performance with coalition
            with torch.no_grad():
                outputs = modified_model(inputs)
                performance = self._compute_performance_metric(outputs, inputs)

            # Convert to coalition value
            value = self._performance_to_value(performance)

            self._value_cache[coalition] = value
            return value

        except Exception as e:
            self.logger.warning(f"Coalition value computation failed for {coalition}: {e}")
            return 0.0

    def _create_coalition_model(self, coalition: FrozenSet[str]) -> nn.Module:
        """Create model with only coalition members active."""
        import copy
        model_copy = copy.deepcopy(self.model)

        # Identify which parameters belong to coalition members
        coalition_params = set()
        for name, param in model_copy.named_parameters():
            if param.requires_grad:
                # Check if parameter belongs to any coalition member
                for player in coalition:
                    if player in name or name.startswith(player):
                        coalition_params.add(name)
                        break

        # Zero out non-coalition parameters
        for name, param in model_copy.named_parameters():
            if param.requires_grad and name not in coalition_params:
                param.data.zero_()

        return model_copy

    def _compute_performance_metric(self, outputs: torch.Tensor,
                                  inputs: torch.Tensor) -> float:
        """Compute performance metric from model outputs."""
        # Use output magnitude and stability as performance indicators
        magnitude = torch.norm(outputs).item()
        variance = torch.var(outputs).item()

        # Combine metrics (higher magnitude, lower variance = better)
        performance = magnitude / (1.0 + variance)
        return performance

    def _performance_to_value(self, performance: float) -> float:
        """Convert performance metric to coalition value."""
        if self.baseline_performance is None:
            self.baseline_performance = performance

        # Relative improvement as value
        value = performance / (self.baseline_performance + 1e-8)
        return max(0.0, value)  # Ensure non-negative values

    def compute_all_coalition_values(self, players: List[str],
                                   inputs: torch.Tensor) -> Dict[FrozenSet[str], float]:
        """Compute values for all relevant coalitions."""
        coalition_values = {}

        if self.config.sample_coalitions and len(players) > self.config.max_coalition_size:
            # Sample coalitions for large player sets
            coalitions = self._sample_coalitions(players)
        else:
            # Generate all possible coalitions
            coalitions = self._generate_all_coalitions(players)

        # Compute values in parallel if enabled
        if self.config.parallel_computation:
            coalition_values = self._compute_values_parallel(coalitions, inputs)
        else:
            for coalition in coalitions:
                coalition_values[coalition] = self.compute_coalition_value(coalition, inputs)

        return coalition_values

    def _sample_coalitions(self, players: List[str]) -> List[FrozenSet[str]]:
        """Sample coalitions for computational efficiency."""
        coalitions = []

        # Always include singleton coalitions and grand coalition
        for player in players:
            coalitions.append(frozenset([player]))
        coalitions.append(frozenset(players))

        # Sample random coalitions
        for _ in range(self.config.coalition_sample_size):
            size = np.random.randint(self.config.min_coalition_size,
                                   min(len(players), self.config.max_coalition_size) + 1)
            coalition = frozenset(np.random.choice(players, size, replace=False))
            coalitions.append(coalition)

        return list(set(coalitions))  # Remove duplicates

    def _generate_all_coalitions(self, players: List[str]) -> List[FrozenSet[str]]:
        """Generate all possible coalitions."""
        coalitions = []
        for size in range(1, len(players) + 1):
            for coalition in itertools.combinations(players, size):
                coalitions.append(frozenset(coalition))
        return coalitions

    def _compute_values_parallel(self, coalitions: List[FrozenSet[str]],
                               inputs: torch.Tensor) -> Dict[FrozenSet[str], float]:
        """Compute coalition values in parallel."""
        coalition_values = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            future_to_coalition = {
                executor.submit(self.compute_coalition_value, coalition, inputs): coalition
                for coalition in coalitions
            }

            # Collect results
            for future in as_completed(future_to_coalition):
                coalition = future_to_coalition[future]
                try:
                    value = future.result()
                    coalition_values[coalition] = value
                except Exception as e:
                    self.logger.warning(f"Failed to compute value for {coalition}: {e}")
                    coalition_values[coalition] = 0.0

        return coalition_values


class ShapleyValueCalculator:
    """Calculates Shapley values for cooperative games."""

    def __init__(self, config: CooperativeGameConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def compute_shapley_values(self, players: List[str],
                             coalition_values: Dict[FrozenSet[str], float]) -> Dict[str, float]:
        """Compute Shapley values for all players."""
        if len(players) <= self.config.exact_shapley_threshold:
            return self._compute_exact_shapley_values(players, coalition_values)
        else:
            return self._compute_approximate_shapley_values(players, coalition_values)

    def _compute_exact_shapley_values(self, players: List[str],
                                    coalition_values: Dict[FrozenSet[str], float]) -> Dict[str, float]:
        """Compute exact Shapley values using the formula."""
        shapley_values = {}
        n = len(players)

        for player in players:
            shapley_value = 0.0

            # Sum over all coalitions not containing the player
            for size in range(n):
                for coalition in itertools.combinations([p for p in players if p != player], size):
                    coalition_set = frozenset(coalition)
                    coalition_with_player = coalition_set | {player}

                    # Marginal contribution
                    v_with = coalition_values.get(coalition_with_player, 0.0)
                    v_without = coalition_values.get(coalition_set, 0.0)
                    marginal_contribution = v_with - v_without

                    # Shapley weight
                    weight = (np.math.factorial(size) * np.math.factorial(n - size - 1)) / np.math.factorial(n)

                    shapley_value += weight * marginal_contribution

            shapley_values[player] = shapley_value

        return shapley_values

    def _compute_approximate_shapley_values(self, players: List[str],
                                          coalition_values: Dict[FrozenSet[str], float]) -> Dict[str, float]:
        """Compute approximate Shapley values using sampling."""
        shapley_values = {player: 0.0 for player in players}

        for _ in range(self.config.shapley_sample_size):
            # Random permutation of players
            permutation = np.random.permutation(players)

            # Compute marginal contributions
            coalition = set()
            for player in permutation:
                coalition_without = frozenset(coalition)
                coalition_with = frozenset(coalition | {player})

                v_with = coalition_values.get(coalition_with, 0.0)
                v_without = coalition_values.get(coalition_without, 0.0)

                marginal_contribution = v_with - v_without
                shapley_values[player] += marginal_contribution

                coalition.add(player)

        # Average over samples
        for player in players:
            shapley_values[player] /= self.config.shapley_sample_size

        return shapley_values


class CoreAnalyzer:
    """Analyzes core solutions in cooperative games."""

    def __init__(self, config: CooperativeGameConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def analyze_core(self, players: List[str],
                    coalition_values: Dict[FrozenSet[str], float],
                    allocation: Dict[str, float]) -> Tuple[bool, int, int]:
        """Analyze if allocation is in the core."""
        constraints_satisfied = 0
        total_constraints = 0

        # Check all coalition constraints
        for coalition, value in coalition_values.items():
            if len(coalition) == 0:
                continue

            total_constraints += 1

            # Sum of allocations to coalition members
            coalition_allocation = sum(allocation.get(player, 0.0) for player in coalition)

            # Core constraint: allocation >= coalition value
            if coalition_allocation >= value - self.config.core_epsilon:
                constraints_satisfied += 1

        is_in_core = (constraints_satisfied == total_constraints)
        return is_in_core, constraints_satisfied, total_constraints

    def find_core_allocations(self, players: List[str],
                            coalition_values: Dict[FrozenSet[str], float]) -> List[Dict[str, float]]:
        """Find allocations in the core (simplified approach)."""
        # This is a simplified implementation
        # In practice, this requires solving a linear program

        grand_coalition = frozenset(players)
        total_value = coalition_values.get(grand_coalition, 0.0)

        # Equal allocation as starting point
        equal_allocation = {player: total_value / len(players) for player in players}

        is_core, _, _ = self.analyze_core(players, coalition_values, equal_allocation)

        if is_core:
            return [equal_allocation]
        else:
            return []  # Core might be empty


class CooperativeGameAnalyzer:
    """
    Main analyzer for cooperative game analysis of neural weights.

    Implements coalition formation analysis to understand cooperative
    structures among neural network components.
    """

    def __init__(self, config: Optional[CooperativeGameConfig] = None):
        self.config = config or CooperativeGameConfig()
        self.value_function = None
        self.shapley_calculator = ShapleyValueCalculator(self.config)
        self.core_analyzer = CoreAnalyzer(self.config)
        self.analysis_history = []
        self.logger = logging.getLogger(__name__)

    def analyze_cooperative_structure(self, model: nn.Module,
                                    inputs: torch.Tensor) -> Optional[CoalitionStructure]:
        """Perform comprehensive cooperative game analysis."""
        try:
            start_time = time.time()

            # Initialize value function
            self.value_function = CoalitionValueFunction(model, self.config)

            # Identify players (weight groups)
            players = self._identify_players(model)

            if len(players) > self.config.max_coalition_size:
                self.logger.warning(f"Too many players ({len(players)}), sampling {self.config.max_coalition_size}")
                players = players[:self.config.max_coalition_size]

            self.logger.info(f"Analyzing cooperative game with {len(players)} players")

            # Compute coalition values
            coalition_values = self.value_function.compute_all_coalition_values(players, inputs)

            # Compute Shapley values
            shapley_values = {}
            if self.config.compute_shapley_values:
                shapley_values = self.shapley_calculator.compute_shapley_values(players, coalition_values)

            # Analyze coalition structure
            coalition_structure = self._analyze_coalition_structure(players, coalition_values, shapley_values)

            # Store analysis
            self.analysis_history.append({
                'timestamp': time.time(),
                'coalition_structure': coalition_structure,
                'num_players': len(players),
                'computation_time': time.time() - start_time
            })

            self.logger.info(f"Cooperative analysis completed in {time.time() - start_time:.2f}s")

            return coalition_structure

        except Exception as e:
            self.logger.error(f"Cooperative game analysis failed: {e}")
            return None

    def _identify_players(self, model: nn.Module) -> List[str]:
        """Identify players for cooperative game."""
        players = []

        # Group parameters by layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d)):
                players.append(f"layer_{name}")

        return players[:self.config.max_coalition_size]

    def _analyze_coalition_structure(self, players: List[str],
                                   coalition_values: Dict[FrozenSet[str], float],
                                   shapley_values: Dict[str, float]) -> CoalitionStructure:
        """Analyze the coalition structure of the game."""
        # Find stable coalitions (simplified approach)
        stable_coalitions = self._find_stable_coalitions(players, coalition_values)

        # Compute efficiency
        grand_coalition = frozenset(players)
        grand_coalition_value = coalition_values.get(grand_coalition, 0.0)
        total_stable_value = sum(coalition_values.get(coalition, 0.0) for coalition in stable_coalitions)
        efficiency_score = total_stable_value / (grand_coalition_value + 1e-8)

        # Core analysis
        is_in_core = False
        core_satisfied = 0
        total_constraints = 0

        if self.config.analyze_core and shapley_values:
            is_in_core, core_satisfied, total_constraints = self.core_analyzer.analyze_core(
                players, coalition_values, shapley_values
            )

        # Compute stability score
        stability_score = self._compute_stability_score(stable_coalitions, coalition_values)

        coalition_structure = CoalitionStructure(
            coalitions=stable_coalitions,
            coalition_values=coalition_values,
            stability_score=stability_score,
            efficiency_score=efficiency_score,
            shapley_values=shapley_values,
            is_in_core=is_in_core,
            core_constraints_satisfied=core_satisfied,
            total_core_constraints=total_constraints
        )

        return coalition_structure

    def _find_stable_coalitions(self, players: List[str],
                              coalition_values: Dict[FrozenSet[str], float]) -> List[FrozenSet[str]]:
        """Find stable coalitions using a greedy approach."""
        remaining_players = set(players)
        stable_coalitions = []

        while remaining_players:
            # Find best coalition among remaining players
            best_coalition = None
            best_value_per_player = 0.0

            for size in range(1, min(len(remaining_players), self.config.max_coalition_size) + 1):
                for coalition in itertools.combinations(remaining_players, size):
                    coalition_set = frozenset(coalition)
                    value = coalition_values.get(coalition_set, 0.0)
                    value_per_player = value / len(coalition) if len(coalition) > 0 else 0.0

                    if value_per_player > best_value_per_player:
                        best_value_per_player = value_per_player
                        best_coalition = coalition_set

            if best_coalition:
                stable_coalitions.append(best_coalition)
                remaining_players -= best_coalition
            else:
                # Add remaining players as singletons
                for player in remaining_players:
                    stable_coalitions.append(frozenset([player]))
                break

        return stable_coalitions

    def _compute_stability_score(self, coalitions: List[FrozenSet[str]],
                               coalition_values: Dict[FrozenSet[str], float]) -> float:
        """Compute stability score of coalition structure."""
        if not coalitions:
            return 0.0

        # Check if any coalition has incentive to deviate
        total_deviations = 0
        possible_deviations = 0

        for coalition in coalitions:
            if len(coalition) <= 1:
                continue

            coalition_value = coalition_values.get(coalition, 0.0)
            value_per_player = coalition_value / len(coalition)

            # Check if sub-coalitions would prefer to deviate
            for sub_size in range(1, len(coalition)):
                for sub_coalition in itertools.combinations(coalition, sub_size):
                    sub_coalition_set = frozenset(sub_coalition)
                    sub_value = coalition_values.get(sub_coalition_set, 0.0)
                    sub_value_per_player = sub_value / len(sub_coalition)

                    possible_deviations += 1
                    if sub_value_per_player > value_per_player:
                        total_deviations += 1

        if possible_deviations == 0:
            return 1.0

        stability_score = 1.0 - (total_deviations / possible_deviations)
        return max(0.0, stability_score)

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of cooperative game analysis history."""
        if not self.analysis_history:
            return {"message": "No analysis history available"}

        recent_analyses = self.analysis_history[-10:]

        stability_scores = [a['coalition_structure'].stability_score for a in recent_analyses]
        efficiency_scores = [a['coalition_structure'].efficiency_score for a in recent_analyses]
        computation_times = [a['computation_time'] for a in recent_analyses]

        return {
            "total_analyses": len(self.analysis_history),
            "recent_analyses": len(recent_analyses),
            "average_stability": np.mean(stability_scores),
            "min_stability": np.min(stability_scores),
            "average_efficiency": np.mean(efficiency_scores),
            "max_efficiency": np.max(efficiency_scores),
            "average_computation_time": np.mean(computation_times),
            "config": {
                "max_coalition_size": self.config.max_coalition_size,
                "sample_coalitions": self.config.sample_coalitions,
                "coalition_sample_size": self.config.coalition_sample_size,
                "compute_shapley_values": self.config.compute_shapley_values
            }
        }

    def reset_analysis_history(self) -> None:
        """Reset analysis history."""
        self.analysis_history.clear()
        self.logger.info("Cooperative analysis history reset")