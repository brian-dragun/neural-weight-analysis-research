"""
NeuroGame Architecture: Strategic Modeling of Neural Network Interactions

This module implements a novel game-theoretic framework for analyzing neural networks
where individual neurons or groups of neurons are modeled as strategic players in
a multi-agent system. This approach reveals critical weight configurations through
the lens of strategic interactions and equilibrium analysis.

Key Innovations:
- Neurons as strategic players with utility functions
- Network interactions modeled as multi-player games
- Nash equilibrium analysis for critical weight discovery
- Evolutionary stable strategies for long-term vulnerability assessment
- Cooperative vs adversarial neuron dynamics
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict

try:
    from scipy.optimize import minimize, linprog
    from scipy.linalg import eigvals
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Some game-theoretic methods will be disabled.")


class PlayerType(Enum):
    """Types of players in the neural game."""
    NEURON = "neuron"              # Individual neuron
    LAYER = "layer"                # Entire layer as a player
    ATTENTION_HEAD = "attention_head"  # Attention head in transformer
    MLP_COMPONENT = "mlp_component"    # MLP subcomponent
    RESIDUAL_BLOCK = "residual_block"  # Residual connection block


@dataclass
class GameState:
    """State of the neural game at a given time."""
    players: List[str]                    # Player identifiers
    strategies: Dict[str, np.ndarray]     # Current strategies for each player
    payoffs: Dict[str, float]            # Current payoffs for each player
    network_performance: float           # Overall network performance
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityFunction:
    """Utility function for a player in the neural game."""
    player_id: str
    base_utility: float                  # Base utility without interactions
    interaction_matrix: np.ndarray       # Interaction effects with other players
    self_interaction: float             # Self-interaction coefficient
    performance_weight: float           # Weight on overall network performance
    regularization: float               # Regularization term


@dataclass
class GameEquilibrium:
    """Equilibrium state of the neural game."""
    equilibrium_type: str               # "nash", "correlated", "evolutionary"
    strategies: Dict[str, np.ndarray]   # Equilibrium strategies
    payoffs: Dict[str, float]          # Equilibrium payoffs
    stability_score: float             # Stability measure
    convergence_steps: int              # Steps to reach equilibrium
    is_stable: bool                     # Whether equilibrium is stable


class NeuroGameAnalyzer:
    """
    Strategic analysis of neural networks using game theory.

    Models neural network components as strategic players in a multi-agent
    system to discover critical weight configurations and interaction patterns
    that traditional methods might miss.

    Key Features:
    - Multi-level player modeling (neurons, layers, components)
    - Nash equilibrium computation for critical weights
    - Evolutionary stability analysis
    - Cooperative vs adversarial dynamics
    - Vulnerability assessment through strategic interactions
    """

    def __init__(
        self,
        model: nn.Module,
        player_type: PlayerType = PlayerType.LAYER,
        strategy_space_size: int = 10,
        interaction_radius: int = 2
    ):
        """
        Initialize the NeuroGame analyzer.

        Args:
            model: PyTorch model to analyze
            player_type: Type of players to model
            strategy_space_size: Dimension of strategy space for each player
            interaction_radius: Radius of interactions between players
        """
        self.model = model
        self.player_type = player_type
        self.strategy_space_size = strategy_space_size
        self.interaction_radius = interaction_radius

        # Game state management
        self.players = {}
        self.utility_functions = {}
        self.game_history = []
        self.current_equilibrium = None

        # Configuration
        self.config = {
            "learning_rate": 0.01,
            "equilibrium_tolerance": 1e-6,
            "max_iterations": 1000,
            "payoff_normalization": True,
            "interaction_decay": 0.9,
        }

        # Initialize players based on model structure
        self._initialize_players()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"NeuroGame analyzer initialized with {len(self.players)} players")

    def _initialize_players(self) -> None:
        """Initialize players based on the model structure and player type."""
        self.players = {}

        if self.player_type == PlayerType.LAYER:
            # Each layer is a player
            for i, (name, module) in enumerate(self.model.named_modules()):
                if len(list(module.children())) == 0:  # Leaf modules only
                    self.players[f"layer_{i}_{name}"] = {
                        "module": module,
                        "layer_index": i,
                        "name": name,
                        "parameter_count": sum(p.numel() for p in module.parameters()),
                        "strategy": self._initialize_strategy(),
                    }

        elif self.player_type == PlayerType.NEURON:
            # Individual neurons as players (simplified for large models)
            neuron_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Sample subset of neurons for computational efficiency
                    output_size = module.out_features if hasattr(module, 'out_features') else module.out_channels
                    sample_size = min(output_size, 100)  # Limit for scalability

                    for neuron_idx in range(sample_size):
                        player_id = f"neuron_{neuron_count}_{name}_{neuron_idx}"
                        self.players[player_id] = {
                            "module": module,
                            "neuron_index": neuron_idx,
                            "layer_name": name,
                            "strategy": self._initialize_strategy(),
                        }
                        neuron_count += 1

        elif self.player_type == PlayerType.ATTENTION_HEAD:
            # Attention heads as players (for transformer models)
            for name, module in self.model.named_modules():
                if hasattr(module, 'num_heads'):  # Multi-head attention
                    for head_idx in range(module.num_heads):
                        player_id = f"attention_head_{head_idx}_{name}"
                        self.players[player_id] = {
                            "module": module,
                            "head_index": head_idx,
                            "layer_name": name,
                            "strategy": self._initialize_strategy(),
                        }

        # Initialize utility functions for all players
        self._initialize_utility_functions()

    def _initialize_strategy(self) -> np.ndarray:
        """Initialize random strategy for a player."""
        strategy = np.random.random(self.strategy_space_size)
        return strategy / np.sum(strategy)  # Normalize to probability distribution

    def _initialize_utility_functions(self) -> None:
        """Initialize utility functions for all players."""
        player_ids = list(self.players.keys())
        n_players = len(player_ids)

        for player_id in player_ids:
            # Create interaction matrix
            interaction_matrix = np.random.normal(0, 0.1, (n_players, self.strategy_space_size))

            # Stronger interactions with nearby players
            player_idx = player_ids.index(player_id)
            for other_idx, other_id in enumerate(player_ids):
                if other_idx != player_idx:
                    distance = abs(other_idx - player_idx)
                    if distance > self.interaction_radius:
                        interaction_matrix[other_idx] *= self.config["interaction_decay"] ** (distance - self.interaction_radius)

            self.utility_functions[player_id] = UtilityFunction(
                player_id=player_id,
                base_utility=np.random.random(),
                interaction_matrix=interaction_matrix,
                self_interaction=np.random.normal(0, 0.1),
                performance_weight=np.random.uniform(0.5, 1.0),
                regularization=0.01
            )

    def compute_player_utility(
        self,
        player_id: str,
        player_strategy: np.ndarray,
        all_strategies: Dict[str, np.ndarray],
        network_performance: float
    ) -> float:
        """
        Compute utility for a specific player given current strategies.

        Args:
            player_id: ID of the player
            player_strategy: Strategy of the player
            all_strategies: Strategies of all players
            network_performance: Current network performance

        Returns:
            Utility value for the player
        """
        if player_id not in self.utility_functions:
            return 0.0

        utility_func = self.utility_functions[player_id]
        player_ids = list(self.players.keys())

        # Base utility
        utility = utility_func.base_utility

        # Self-interaction term
        utility += utility_func.self_interaction * np.dot(player_strategy, player_strategy)

        # Interaction with other players
        for other_id, other_strategy in all_strategies.items():
            if other_id != player_id and other_id in player_ids:
                other_idx = player_ids.index(other_id)
                interaction_vector = utility_func.interaction_matrix[other_idx]
                utility += np.dot(player_strategy, interaction_vector) * np.mean(other_strategy)

        # Network performance contribution
        utility += utility_func.performance_weight * network_performance

        # Regularization (prefer simpler strategies)
        utility -= utility_func.regularization * np.var(player_strategy)

        return utility

    def simulate_game_step(
        self,
        current_strategies: Dict[str, np.ndarray],
        network_performance: float
    ) -> Dict[str, np.ndarray]:
        """
        Simulate one step of the neural game using best response dynamics.

        Args:
            current_strategies: Current strategies of all players
            network_performance: Current network performance

        Returns:
            Updated strategies after one game step
        """
        new_strategies = {}

        for player_id in self.players.keys():
            # Compute best response for this player
            best_strategy = self._compute_best_response(
                player_id, current_strategies, network_performance
            )
            new_strategies[player_id] = best_strategy

        return new_strategies

    def _compute_best_response(
        self,
        player_id: str,
        other_strategies: Dict[str, np.ndarray],
        network_performance: float
    ) -> np.ndarray:
        """Compute best response strategy for a player."""
        # Objective function to maximize utility
        def objective(strategy):
            all_strategies = other_strategies.copy()
            all_strategies[player_id] = strategy
            utility = self.compute_player_utility(player_id, strategy, all_strategies, network_performance)
            return -utility  # Minimize negative utility

        # Constraints: strategy must be a probability distribution
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},  # Sum to 1
        ]
        bounds = [(0, 1) for _ in range(self.strategy_space_size)]

        # Initial guess
        x0 = other_strategies.get(player_id, self._initialize_strategy())

        if SCIPY_AVAILABLE:
            try:
                result = minimize(
                    objective,
                    x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-9, "disp": False}
                )
                if result.success:
                    return result.x
            except Exception as e:
                self.logger.warning(f"Optimization failed for player {player_id}: {e}")

        # Fallback: gradient ascent
        return self._gradient_ascent_best_response(player_id, other_strategies, network_performance)

    def _gradient_ascent_best_response(
        self,
        player_id: str,
        other_strategies: Dict[str, np.ndarray],
        network_performance: float
    ) -> np.ndarray:
        """Compute best response using gradient ascent."""
        current_strategy = other_strategies.get(player_id, self._initialize_strategy())
        learning_rate = self.config["learning_rate"]

        for _ in range(100):  # Max iterations for gradient ascent
            # Compute gradient (numerical approximation)
            gradient = np.zeros_like(current_strategy)
            epsilon = 1e-6

            for i in range(len(current_strategy)):
                # Forward difference
                strategy_plus = current_strategy.copy()
                strategy_plus[i] += epsilon

                # Renormalize
                strategy_plus = strategy_plus / np.sum(strategy_plus)

                all_strategies_plus = other_strategies.copy()
                all_strategies_plus[player_id] = strategy_plus

                utility_plus = self.compute_player_utility(
                    player_id, strategy_plus, all_strategies_plus, network_performance
                )

                all_strategies_current = other_strategies.copy()
                all_strategies_current[player_id] = current_strategy

                utility_current = self.compute_player_utility(
                    player_id, current_strategy, all_strategies_current, network_performance
                )

                gradient[i] = (utility_plus - utility_current) / epsilon

            # Update strategy
            current_strategy += learning_rate * gradient

            # Project onto simplex (probability distribution constraint)
            current_strategy = np.maximum(current_strategy, 0)
            if np.sum(current_strategy) > 0:
                current_strategy = current_strategy / np.sum(current_strategy)
            else:
                current_strategy = self._initialize_strategy()

        return current_strategy

    def find_nash_equilibrium(
        self,
        network_performance: float,
        initial_strategies: Optional[Dict[str, np.ndarray]] = None
    ) -> GameEquilibrium:
        """
        Find Nash equilibrium of the neural game.

        Args:
            network_performance: Current network performance
            initial_strategies: Initial strategies (random if None)

        Returns:
            GameEquilibrium object with equilibrium information
        """
        # Initialize strategies
        if initial_strategies is None:
            strategies = {player_id: self._initialize_strategy() for player_id in self.players.keys()}
        else:
            strategies = initial_strategies.copy()

        # Iterative best response dynamics
        for iteration in range(self.config["max_iterations"]):
            old_strategies = {k: v.copy() for k, v in strategies.items()}

            # Update strategies using best response
            new_strategies = self.simulate_game_step(strategies, network_performance)

            # Check convergence
            max_change = max(
                np.linalg.norm(new_strategies[player_id] - old_strategies[player_id])
                for player_id in self.players.keys()
            )

            strategies = new_strategies

            if max_change < self.config["equilibrium_tolerance"]:
                # Converged to equilibrium
                payoffs = {
                    player_id: self.compute_player_utility(player_id, strategy, strategies, network_performance)
                    for player_id, strategy in strategies.items()
                }

                stability_score = self._compute_stability_score(strategies, network_performance)

                equilibrium = GameEquilibrium(
                    equilibrium_type="nash",
                    strategies=strategies,
                    payoffs=payoffs,
                    stability_score=stability_score,
                    convergence_steps=iteration + 1,
                    is_stable=stability_score > 0.5
                )

                self.current_equilibrium = equilibrium
                return equilibrium

        # Did not converge
        payoffs = {
            player_id: self.compute_player_utility(player_id, strategy, strategies, network_performance)
            for player_id, strategy in strategies.items()
        }

        return GameEquilibrium(
            equilibrium_type="approximate_nash",
            strategies=strategies,
            payoffs=payoffs,
            stability_score=0.0,
            convergence_steps=self.config["max_iterations"],
            is_stable=False
        )

    def _compute_stability_score(
        self,
        strategies: Dict[str, np.ndarray],
        network_performance: float
    ) -> float:
        """Compute stability score for a strategy profile."""
        total_deviation_incentive = 0.0

        for player_id in self.players.keys():
            # Current utility
            current_utility = self.compute_player_utility(player_id, strategies[player_id], strategies, network_performance)

            # Best response utility
            best_response = self._compute_best_response(player_id, strategies, network_performance)
            other_strategies = strategies.copy()
            other_strategies[player_id] = best_response
            best_utility = self.compute_player_utility(player_id, best_response, other_strategies, network_performance)

            # Deviation incentive
            deviation_incentive = max(0, best_utility - current_utility)
            total_deviation_incentive += deviation_incentive

        # Stability score (lower deviation incentive = higher stability)
        if len(self.players) > 0:
            avg_deviation = total_deviation_incentive / len(self.players)
            stability_score = 1.0 / (1.0 + avg_deviation)
        else:
            stability_score = 1.0

        return stability_score

    def analyze_critical_interactions(
        self,
        equilibrium: GameEquilibrium
    ) -> Dict[str, Any]:
        """
        Analyze critical interactions between players at equilibrium.

        Args:
            equilibrium: Game equilibrium to analyze

        Returns:
            Dictionary with critical interaction analysis
        """
        player_ids = list(self.players.keys())
        n_players = len(player_ids)

        # Compute interaction strength matrix
        interaction_matrix = np.zeros((n_players, n_players))

        for i, player_id in enumerate(player_ids):
            utility_func = self.utility_functions[player_id]
            for j, other_id in enumerate(player_ids):
                if i != j:
                    # Interaction strength = gradient of utility w.r.t. other player's strategy
                    interaction_strength = np.mean(np.abs(utility_func.interaction_matrix[j]))
                    interaction_matrix[i, j] = interaction_strength

        # Find most critical interactions
        critical_interactions = []
        for i in range(n_players):
            for j in range(i + 1, n_players):
                mutual_interaction = interaction_matrix[i, j] + interaction_matrix[j, i]
                critical_interactions.append({
                    "player_1": player_ids[i],
                    "player_2": player_ids[j],
                    "interaction_strength": mutual_interaction,
                    "player_1_to_2": interaction_matrix[i, j],
                    "player_2_to_1": interaction_matrix[j, i]
                })

        # Sort by interaction strength
        critical_interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

        # Compute centrality measures
        player_centrality = {}
        for i, player_id in enumerate(player_ids):
            out_strength = np.sum(interaction_matrix[i, :])
            in_strength = np.sum(interaction_matrix[:, i])
            total_strength = out_strength + in_strength
            player_centrality[player_id] = {
                "out_strength": out_strength,
                "in_strength": in_strength,
                "total_strength": total_strength
            }

        return {
            "interaction_matrix": interaction_matrix,
            "critical_interactions": critical_interactions[:10],  # Top 10
            "player_centrality": player_centrality,
            "most_influential_player": max(player_centrality.keys(),
                                         key=lambda x: player_centrality[x]["total_strength"]),
            "equilibrium_summary": {
                "stability_score": equilibrium.stability_score,
                "convergence_steps": equilibrium.convergence_steps,
                "is_stable": equilibrium.is_stable
            }
        }

    def compute_vulnerability_through_games(
        self,
        perturbation_strength: float = 0.1
    ) -> Dict[str, Any]:
        """
        Compute vulnerability assessment through game-theoretic analysis.

        Args:
            perturbation_strength: Strength of perturbations to test

        Returns:
            Vulnerability analysis results
        """
        if not self.current_equilibrium:
            # Find baseline equilibrium
            baseline_performance = 1.0  # Normalized performance
            self.current_equilibrium = self.find_nash_equilibrium(baseline_performance)

        baseline_equilibrium = self.current_equilibrium
        vulnerability_results = {}

        # Test perturbations to each player
        for player_id in self.players.keys():
            perturbation_results = []

            # Apply random perturbations to this player's utility function
            original_utility = self.utility_functions[player_id]

            for trial in range(5):  # Multiple trials for robustness
                # Perturb interaction matrix
                perturbation = np.random.normal(0, perturbation_strength,
                                               original_utility.interaction_matrix.shape)
                perturbed_matrix = original_utility.interaction_matrix + perturbation

                # Temporarily update utility function
                self.utility_functions[player_id] = UtilityFunction(
                    player_id=original_utility.player_id,
                    base_utility=original_utility.base_utility,
                    interaction_matrix=perturbed_matrix,
                    self_interaction=original_utility.self_interaction,
                    performance_weight=original_utility.performance_weight,
                    regularization=original_utility.regularization
                )

                # Find new equilibrium
                perturbed_equilibrium = self.find_nash_equilibrium(1.0, baseline_equilibrium.strategies)

                # Measure change in equilibrium
                strategy_change = np.linalg.norm(
                    perturbed_equilibrium.strategies[player_id] - baseline_equilibrium.strategies[player_id]
                )

                payoff_change = abs(
                    perturbed_equilibrium.payoffs[player_id] - baseline_equilibrium.payoffs[player_id]
                )

                stability_change = abs(
                    perturbed_equilibrium.stability_score - baseline_equilibrium.stability_score
                )

                perturbation_results.append({
                    "strategy_change": strategy_change,
                    "payoff_change": payoff_change,
                    "stability_change": stability_change
                })

            # Restore original utility function
            self.utility_functions[player_id] = original_utility

            # Compute vulnerability metrics for this player
            avg_strategy_change = np.mean([r["strategy_change"] for r in perturbation_results])
            avg_payoff_change = np.mean([r["payoff_change"] for r in perturbation_results])
            avg_stability_change = np.mean([r["stability_change"] for r in perturbation_results])

            vulnerability_score = (avg_strategy_change + avg_payoff_change + avg_stability_change) / 3

            vulnerability_results[player_id] = {
                "vulnerability_score": vulnerability_score,
                "avg_strategy_change": avg_strategy_change,
                "avg_payoff_change": avg_payoff_change,
                "avg_stability_change": avg_stability_change,
                "is_critical": vulnerability_score > 0.5
            }

        # Overall vulnerability assessment
        all_scores = [r["vulnerability_score"] for r in vulnerability_results.values()]
        overall_vulnerability = {
            "mean_vulnerability": np.mean(all_scores),
            "max_vulnerability": np.max(all_scores),
            "vulnerability_std": np.std(all_scores),
            "critical_players": [pid for pid, r in vulnerability_results.items() if r["is_critical"]],
            "vulnerability_distribution": np.histogram(all_scores, bins=10)[0].tolist()
        }

        return {
            "player_vulnerabilities": vulnerability_results,
            "overall_vulnerability": overall_vulnerability,
            "baseline_equilibrium": baseline_equilibrium,
            "analysis_metadata": {
                "perturbation_strength": perturbation_strength,
                "num_players": len(self.players),
                "player_type": self.player_type.value
            }
        }

    def export_game_analysis(self, output_path: str) -> None:
        """Export comprehensive game analysis to file."""
        analysis_data = {
            "game_configuration": {
                "player_type": self.player_type.value,
                "num_players": len(self.players),
                "strategy_space_size": self.strategy_space_size,
                "interaction_radius": self.interaction_radius
            },
            "current_equilibrium": self.current_equilibrium.__dict__ if self.current_equilibrium else None,
            "players": {pid: {k: v for k, v in player.items() if k != "module"}
                      for pid, player in self.players.items()},
        }

        if self.current_equilibrium:
            analysis_data["critical_interactions"] = self.analyze_critical_interactions(self.current_equilibrium)
            analysis_data["vulnerability_assessment"] = self.compute_vulnerability_through_games()

        import json
        with open(output_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            json.dump(convert_numpy(analysis_data), f, indent=2)

        self.logger.info(f"Game analysis exported to {output_path}")