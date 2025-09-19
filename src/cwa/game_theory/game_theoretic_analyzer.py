"""
Game-Theoretic Weight Analysis for Nash Equilibrium Discovery

Implements game-theoretic analysis to discover critical weight configurations
through Nash equilibrium computation and strategic weight optimization.

Key Features:
- Nash equilibrium computation for weight interactions
- Strategic weight configuration analysis
- Multi-player game modeling of neural components
- Equilibrium stability assessment
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class GameType(Enum):
    """Types of games that can be analyzed."""
    ZERO_SUM = "zero_sum"
    NON_ZERO_SUM = "non_zero_sum"
    COOPERATIVE = "cooperative"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class EquilibriumType(Enum):
    """Types of Nash equilibria."""
    PURE_STRATEGY = "pure_strategy"
    MIXED_STRATEGY = "mixed_strategy"
    CORRELATED = "correlated"
    TREMBLING_HAND = "trembling_hand"


@dataclass
class NashEquilibrium:
    """Container for Nash equilibrium analysis results."""
    equilibrium_type: EquilibriumType
    strategies: Dict[str, np.ndarray]  # Player strategies
    payoffs: Dict[str, float]  # Expected payoffs
    stability_score: float  # 0.0 to 1.0
    convergence_iterations: int
    is_stable: bool
    vulnerability_score: float  # Higher = more vulnerable
    critical_weights: List[str]  # Most influential weight parameters

    # Additional equilibrium properties
    regret_bounds: Dict[str, float] = field(default_factory=dict)
    epsilon_equilibrium: float = 0.0
    computational_complexity: float = 0.0


@dataclass
class GameConfiguration:
    """Configuration for game-theoretic analysis."""
    # Game setup
    max_players: int = 50  # Limit for computational tractability
    strategy_space_size: int = 10  # Discrete strategy options per player
    convergence_threshold: float = 1e-6
    max_iterations: int = 1000

    # Equilibrium computation
    use_mixed_strategies: bool = True
    epsilon_equilibrium_threshold: float = 1e-3
    stability_analysis: bool = True

    # Performance optimization
    parallel_computation: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 512

    # Vulnerability analysis
    vulnerability_threshold: float = 0.7
    critical_weight_ratio: float = 0.1  # Top 10% most critical


class PayoffMatrix:
    """Manages payoff computations for game-theoretic analysis."""

    def __init__(self, model: nn.Module, config: GameConfiguration):
        self.model = model
        self.config = config
        self.baseline_performance = None
        self._cache = {}
        self.logger = logging.getLogger(__name__)

    def compute_payoff(self, player: str, strategy: np.ndarray,
                      other_strategies: Dict[str, np.ndarray],
                      inputs: torch.Tensor) -> float:
        """Compute payoff for a player given strategies."""
        cache_key = self._get_cache_key(player, strategy, other_strategies)

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Apply strategy to model weights
            modified_model = self._apply_strategies(strategy, other_strategies)

            # Compute performance
            with torch.no_grad():
                outputs = modified_model(inputs)
                performance = self._compute_performance_metric(outputs, inputs)

            # Convert to payoff (higher is better for the player)
            payoff = self._performance_to_payoff(performance, player)

            self._cache[cache_key] = payoff
            return payoff

        except Exception as e:
            self.logger.warning(f"Payoff computation failed for {player}: {e}")
            return 0.0

    def _apply_strategies(self, strategy: np.ndarray,
                         other_strategies: Dict[str, np.ndarray]) -> nn.Module:
        """Apply player strategies to model weights."""
        # Create a copy of the model
        model_copy = self._deep_copy_model()

        # Apply weight modifications based on strategies
        param_idx = 0
        for name, param in model_copy.named_parameters():
            if param.requires_grad:
                param_size = param.numel()

                # Extract strategy for this parameter
                if param_idx < len(strategy):
                    weight_modification = strategy[param_idx]
                    # Apply modification (e.g., scaling)
                    param.data *= (1.0 + weight_modification * 0.1)

                param_idx += 1

        return model_copy

    def _deep_copy_model(self) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(self.model)

    def _compute_performance_metric(self, outputs: torch.Tensor,
                                  inputs: torch.Tensor) -> float:
        """Compute performance metric from model outputs."""
        # Use output variance as a stability metric
        variance = torch.var(outputs).item()

        # Use activation magnitude as a performance indicator
        magnitude = torch.norm(outputs).item()

        # Combine metrics (lower variance + moderate magnitude = better)
        performance = magnitude / (1.0 + variance)
        return performance

    def _performance_to_payoff(self, performance: float, player: str) -> float:
        """Convert performance metric to player payoff."""
        if self.baseline_performance is None:
            self.baseline_performance = performance

        # Relative improvement/degradation
        relative_performance = performance / (self.baseline_performance + 1e-8)

        # Convert to payoff (centered around 0)
        payoff = relative_performance - 1.0
        return payoff

    def _get_cache_key(self, player: str, strategy: np.ndarray,
                      other_strategies: Dict[str, np.ndarray]) -> str:
        """Generate cache key for payoff computation."""
        strategy_hash = hash(tuple(strategy))
        others_hash = hash(tuple(sorted([(k, hash(tuple(v)))
                                       for k, v in other_strategies.items()])))
        return f"{player}_{strategy_hash}_{others_hash}"


class NashEquilibriumSolver:
    """Solves for Nash equilibria in neural weight games."""

    def __init__(self, config: GameConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def find_nash_equilibrium(self, players: List[str], payoff_matrix: PayoffMatrix,
                            inputs: torch.Tensor) -> Optional[NashEquilibrium]:
        """Find Nash equilibrium using iterative best response."""
        try:
            start_time = time.time()

            # Initialize random strategies
            strategies = {
                player: np.random.uniform(-1, 1, self.config.strategy_space_size)
                for player in players
            }

            convergence_history = []

            for iteration in range(self.config.max_iterations):
                old_strategies = {k: v.copy() for k, v in strategies.items()}

                # Update each player's strategy via best response
                for player in players:
                    other_strategies = {k: v for k, v in strategies.items() if k != player}
                    best_strategy = self._compute_best_response(
                        player, other_strategies, payoff_matrix, inputs
                    )
                    strategies[player] = best_strategy

                # Check convergence
                max_change = max(
                    np.linalg.norm(strategies[player] - old_strategies[player])
                    for player in players
                )
                convergence_history.append(max_change)

                if max_change < self.config.convergence_threshold:
                    self.logger.info(f"Nash equilibrium converged after {iteration} iterations")
                    break

            # Compute final payoffs and analyze equilibrium
            final_payoffs = {}
            for player in players:
                other_strategies = {k: v for k, v in strategies.items() if k != player}
                payoff = payoff_matrix.compute_payoff(player, strategies[player],
                                                    other_strategies, inputs)
                final_payoffs[player] = payoff

            # Analyze equilibrium properties
            stability_score = self._compute_stability_score(strategies, payoff_matrix, inputs)
            vulnerability_score = self._compute_vulnerability_score(final_payoffs)
            critical_weights = self._identify_critical_weights(strategies)

            equilibrium = NashEquilibrium(
                equilibrium_type=EquilibriumType.MIXED_STRATEGY,
                strategies=strategies,
                payoffs=final_payoffs,
                stability_score=stability_score,
                convergence_iterations=iteration + 1,
                is_stable=stability_score > 0.5,
                vulnerability_score=vulnerability_score,
                critical_weights=critical_weights,
                computational_complexity=time.time() - start_time
            )

            return equilibrium

        except Exception as e:
            self.logger.error(f"Nash equilibrium computation failed: {e}")
            return None

    def _compute_best_response(self, player: str, other_strategies: Dict[str, np.ndarray],
                             payoff_matrix: PayoffMatrix, inputs: torch.Tensor) -> np.ndarray:
        """Compute best response strategy for a player."""
        best_strategy = None
        best_payoff = float('-inf')

        # Try multiple random strategies and gradient-based optimization
        for _ in range(10):  # Limited search for performance
            candidate_strategy = np.random.uniform(-1, 1, self.config.strategy_space_size)
            payoff = payoff_matrix.compute_payoff(player, candidate_strategy,
                                                other_strategies, inputs)

            if payoff > best_payoff:
                best_payoff = payoff
                best_strategy = candidate_strategy

        return best_strategy if best_strategy is not None else np.zeros(self.config.strategy_space_size)

    def _compute_stability_score(self, strategies: Dict[str, np.ndarray],
                               payoff_matrix: PayoffMatrix, inputs: torch.Tensor) -> float:
        """Compute stability score of the equilibrium."""
        total_regret = 0.0
        num_players = len(strategies)

        for player, strategy in strategies.items():
            other_strategies = {k: v for k, v in strategies.items() if k != player}
            current_payoff = payoff_matrix.compute_payoff(player, strategy,
                                                        other_strategies, inputs)

            # Check regret against alternative strategies
            max_alternative_payoff = current_payoff
            for _ in range(5):  # Sample alternative strategies
                alt_strategy = np.random.uniform(-1, 1, len(strategy))
                alt_payoff = payoff_matrix.compute_payoff(player, alt_strategy,
                                                        other_strategies, inputs)
                max_alternative_payoff = max(max_alternative_payoff, alt_payoff)

            regret = max(0, max_alternative_payoff - current_payoff)
            total_regret += regret

        # Convert to stability score (lower regret = higher stability)
        avg_regret = total_regret / num_players
        stability_score = 1.0 / (1.0 + avg_regret)

        return stability_score

    def _compute_vulnerability_score(self, payoffs: Dict[str, float]) -> float:
        """Compute vulnerability score based on payoff distribution."""
        if not payoffs:
            return 1.0

        payoff_values = list(payoffs.values())
        payoff_std = np.std(payoff_values)
        payoff_mean = np.mean(payoff_values)

        # High variance in payoffs indicates vulnerability
        vulnerability = min(payoff_std / (abs(payoff_mean) + 1e-8), 1.0)
        return vulnerability

    def _identify_critical_weights(self, strategies: Dict[str, np.ndarray]) -> List[str]:
        """Identify most critical weight parameters."""
        # Analyze strategy magnitudes to identify critical parameters
        all_strategies = np.array(list(strategies.values()))
        strategy_importance = np.std(all_strategies, axis=0)

        # Get indices of most important strategies
        num_critical = max(1, int(len(strategy_importance) * self.config.critical_weight_ratio))
        critical_indices = np.argsort(strategy_importance)[-num_critical:]

        # Convert to parameter names (simplified)
        critical_weights = [f"param_{idx}" for idx in critical_indices]
        return critical_weights


class GameTheoreticWeightAnalyzer:
    """
    Main analyzer for game-theoretic weight analysis.

    Implements Nash equilibrium-based analysis of neural network weights
    to discover critical configurations and vulnerabilities.
    """

    def __init__(self, config: Optional[GameConfiguration] = None):
        self.config = config or GameConfiguration()
        self.payoff_matrix = None
        self.equilibrium_solver = NashEquilibriumSolver(self.config)
        self.analysis_history = []
        self.logger = logging.getLogger(__name__)

    def analyze_weight_game(self, model: nn.Module, inputs: torch.Tensor) -> Optional[NashEquilibrium]:
        """Perform comprehensive game-theoretic weight analysis."""
        try:
            start_time = time.time()

            # Initialize payoff matrix
            self.payoff_matrix = PayoffMatrix(model, self.config)

            # Identify players (weight parameters or layer groups)
            players = self._identify_players(model)

            if len(players) > self.config.max_players:
                self.logger.warning(f"Too many players ({len(players)}), sampling {self.config.max_players}")
                players = players[:self.config.max_players]

            self.logger.info(f"Analyzing game with {len(players)} players")

            # Find Nash equilibrium
            equilibrium = self.equilibrium_solver.find_nash_equilibrium(
                players, self.payoff_matrix, inputs
            )

            if equilibrium:
                # Store analysis
                self.analysis_history.append({
                    'timestamp': time.time(),
                    'equilibrium': equilibrium,
                    'num_players': len(players),
                    'computation_time': time.time() - start_time
                })

                self.logger.info(f"Game analysis completed in {time.time() - start_time:.2f}s")
                self.logger.info(f"Vulnerability score: {equilibrium.vulnerability_score:.3f}")
                self.logger.info(f"Stability score: {equilibrium.stability_score:.3f}")

            return equilibrium

        except Exception as e:
            self.logger.error(f"Game-theoretic analysis failed: {e}")
            return None

    def _identify_players(self, model: nn.Module) -> List[str]:
        """Identify players in the weight game."""
        players = []

        # Group parameters by layer for manageable game size
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                players.append(f"layer_{name}")

        # If still too many, group by module type
        if len(players) > self.config.max_players:
            player_groups = {}
            for name, module in model.named_modules():
                module_type = type(module).__name__
                if module_type not in player_groups:
                    player_groups[module_type] = []
                player_groups[module_type].append(name)

            players = list(player_groups.keys())

        return players[:self.config.max_players]

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of game-theoretic analysis history."""
        if not self.analysis_history:
            return {"message": "No analysis history available"}

        recent_analyses = self.analysis_history[-10:]  # Last 10 analyses

        vulnerability_scores = [a['equilibrium'].vulnerability_score for a in recent_analyses]
        stability_scores = [a['equilibrium'].stability_score for a in recent_analyses]
        computation_times = [a['computation_time'] for a in recent_analyses]

        return {
            "total_analyses": len(self.analysis_history),
            "recent_analyses": len(recent_analyses),
            "average_vulnerability": np.mean(vulnerability_scores),
            "max_vulnerability": np.max(vulnerability_scores),
            "average_stability": np.mean(stability_scores),
            "min_stability": np.min(stability_scores),
            "average_computation_time": np.mean(computation_times),
            "config": {
                "max_players": self.config.max_players,
                "strategy_space_size": self.config.strategy_space_size,
                "convergence_threshold": self.config.convergence_threshold,
                "vulnerability_threshold": self.config.vulnerability_threshold
            }
        }

    def reset_analysis_history(self) -> None:
        """Reset analysis history."""
        self.analysis_history.clear()
        self.logger.info("Analysis history reset")