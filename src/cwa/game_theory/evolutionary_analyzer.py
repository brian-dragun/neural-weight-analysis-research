"""
Evolutionary Stability Analysis for Neural Weight Dynamics

Implements evolutionary game theory to analyze the long-term stability
and evolution of neural network weight configurations under various pressures.

Key Features:
- Evolutionarily Stable Strategy (ESS) computation
- Replicator dynamics modeling
- Population-level weight evolution analysis
- Stability basin analysis
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


class EvolutionaryDynamics(Enum):
    """Types of evolutionary dynamics."""
    REPLICATOR = "replicator"
    SELECTION_MUTATION = "selection_mutation"
    IMITATION = "imitation"
    BEST_RESPONSE = "best_response"
    LOGIT = "logit"


class StabilityType(Enum):
    """Types of evolutionary stability."""
    ESS = "evolutionarily_stable_strategy"
    ASYMPTOTIC_STABLE = "asymptotic_stable"
    LYAPUNOV_STABLE = "lyapunov_stable"
    NEUTRALLY_STABLE = "neutrally_stable"
    UNSTABLE = "unstable"


@dataclass
class ESS:
    """Evolutionarily Stable Strategy result."""
    strategy: np.ndarray  # ESS strategy profile
    stability_type: StabilityType
    basin_size: float  # Size of stability basin
    convergence_rate: float  # Rate of convergence to ESS
    invasion_threshold: float  # Threshold for successful invasion

    # Evolutionary properties
    fitness_landscape: Dict[str, float] = field(default_factory=dict)
    mutation_stability: bool = False
    drift_tolerance: float = 0.0

    # Computational details
    computation_time: float = 0.0
    convergence_iterations: int = 0
    numerical_precision: float = 1e-6


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary stability analysis."""
    # Dynamics parameters
    dynamics_type: EvolutionaryDynamics = EvolutionaryDynamics.REPLICATOR
    time_horizon: float = 100.0  # Integration time
    time_steps: int = 1000

    # ESS computation
    ess_threshold: float = 1e-6  # Convergence threshold
    invasion_size: float = 0.01  # Size of invading mutant population
    max_iterations: int = 1000

    # Stability analysis
    stability_epsilon: float = 1e-4  # Perturbation size for stability test
    basin_resolution: int = 50  # Resolution for basin size computation
    lyapunov_tolerance: float = 1e-3

    # Mutation parameters
    mutation_rate: float = 0.001
    mutation_strength: float = 0.1
    selection_strength: float = 1.0

    # Performance optimization
    numerical_method: str = 'RK45'  # ODE solver method
    rtol: float = 1e-6  # Relative tolerance
    atol: float = 1e-9  # Absolute tolerance


class FitnessFunction:
    """Fitness function for evolutionary dynamics."""

    def __init__(self, model: nn.Module, config: EvolutionaryConfig):
        self.model = model
        self.config = config
        self.baseline_fitness = None
        self._fitness_cache = {}
        self.logger = logging.getLogger(__name__)

    def compute_fitness(self, strategy: np.ndarray, population: np.ndarray,
                       inputs: torch.Tensor) -> float:
        """Compute fitness of a strategy against population."""
        strategy_key = hash(tuple(strategy))
        population_key = hash(tuple(population.flatten()))
        cache_key = f"{strategy_key}_{population_key}"

        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        try:
            # Apply strategy to model
            modified_model = self._apply_strategy(strategy)

            # Compute performance against population
            with torch.no_grad():
                outputs = modified_model(inputs)
                performance = self._compute_performance(outputs, inputs, population)

            fitness = self._performance_to_fitness(performance)
            self._fitness_cache[cache_key] = fitness

            return fitness

        except Exception as e:
            self.logger.warning(f"Fitness computation failed: {e}")
            return 0.0

    def _apply_strategy(self, strategy: np.ndarray) -> nn.Module:
        """Apply strategy to model parameters."""
        import copy
        model_copy = copy.deepcopy(self.model)

        # Apply strategy to weights (simplified implementation)
        param_idx = 0
        for name, param in model_copy.named_parameters():
            if param.requires_grad and param_idx < len(strategy):
                # Apply strategy as weight scaling
                param.data *= (1.0 + strategy[param_idx] * 0.1)
                param_idx += 1

        return model_copy

    def _compute_performance(self, outputs: torch.Tensor, inputs: torch.Tensor,
                           population: np.ndarray) -> float:
        """Compute performance metric considering population dynamics."""
        # Use output stability and magnitude as performance indicators
        magnitude = torch.norm(outputs).item()
        stability = 1.0 / (1.0 + torch.var(outputs).item())

        # Consider population composition in performance
        population_factor = np.mean(population)  # Simplified

        performance = magnitude * stability * (1.0 + population_factor)
        return performance

    def _performance_to_fitness(self, performance: float) -> float:
        """Convert performance to fitness value."""
        if self.baseline_fitness is None:
            self.baseline_fitness = performance

        # Relative fitness
        fitness = performance / (self.baseline_fitness + 1e-8)
        return max(0.0, fitness)

    def compute_payoff_matrix(self, strategies: List[np.ndarray],
                            inputs: torch.Tensor) -> np.ndarray:
        """Compute payoff matrix for strategy interactions."""
        n_strategies = len(strategies)
        payoff_matrix = np.zeros((n_strategies, n_strategies))

        for i, strategy_i in enumerate(strategies):
            for j, strategy_j in enumerate(strategies):
                # Fitness of strategy i against strategy j
                population = np.array([strategies[j]])  # Single opponent
                fitness = self.compute_fitness(strategy_i, population, inputs)
                payoff_matrix[i, j] = fitness

        return payoff_matrix


class ReplicatorDynamics:
    """Implements replicator dynamics for evolutionary analysis."""

    def __init__(self, payoff_matrix: np.ndarray, config: EvolutionaryConfig):
        self.payoff_matrix = payoff_matrix
        self.config = config
        self.logger = logging.getLogger(__name__)

    def replicator_equation(self, t: float, x: np.ndarray) -> np.ndarray:
        """Replicator differential equation."""
        if np.sum(x) == 0:
            return np.zeros_like(x)

        # Normalize population
        x = x / np.sum(x)

        # Average fitness
        avg_fitness = np.dot(x, np.dot(self.payoff_matrix, x))

        # Individual fitness
        individual_fitness = np.dot(self.payoff_matrix, x)

        # Replicator dynamics
        dx_dt = x * (individual_fitness - avg_fitness)

        return dx_dt

    def integrate_dynamics(self, initial_population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate replicator dynamics over time."""
        t_span = (0, self.config.time_horizon)
        t_eval = np.linspace(0, self.config.time_horizon, self.config.time_steps)

        # Normalize initial population
        initial_population = initial_population / np.sum(initial_population)

        # Solve ODE
        solution = solve_ivp(
            self.replicator_equation,
            t_span,
            initial_population,
            t_eval=t_eval,
            method=self.config.numerical_method,
            rtol=self.config.rtol,
            atol=self.config.atol
        )

        return solution.t, solution.y

    def find_fixed_points(self) -> List[np.ndarray]:
        """Find fixed points of replicator dynamics."""
        n_strategies = self.payoff_matrix.shape[0]
        fixed_points = []

        # Check corner equilibria (pure strategies)
        for i in range(n_strategies):
            corner = np.zeros(n_strategies)
            corner[i] = 1.0

            # Check if it's a fixed point
            dx_dt = self.replicator_equation(0, corner)
            if np.allclose(dx_dt, 0, atol=self.config.ess_threshold):
                fixed_points.append(corner)

        # Search for interior fixed points (simplified)
        for _ in range(100):  # Random search
            candidate = np.random.random(n_strategies)
            candidate = candidate / np.sum(candidate)

            # Optimize to find fixed point
            def fixed_point_error(x):
                x = x / np.sum(x)
                dx_dt = self.replicator_equation(0, x)
                return np.sum(dx_dt**2)

            result = minimize(fixed_point_error, candidate, method='L-BFGS-B',
                            bounds=[(0, 1) for _ in range(n_strategies)])

            if result.success and result.fun < self.config.ess_threshold:
                fixed_point = result.x / np.sum(result.x)

                # Check if this is a new fixed point
                is_new = True
                for existing_fp in fixed_points:
                    if np.allclose(fixed_point, existing_fp, atol=self.config.ess_threshold):
                        is_new = False
                        break

                if is_new:
                    fixed_points.append(fixed_point)

        return fixed_points


class ESSAnalyzer:
    """Analyzes Evolutionarily Stable Strategies."""

    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def find_ess(self, payoff_matrix: np.ndarray) -> List[ESS]:
        """Find all ESS in the game."""
        replicator = ReplicatorDynamics(payoff_matrix, self.config)
        fixed_points = replicator.find_fixed_points()

        ess_strategies = []

        for fixed_point in fixed_points:
            if self._is_ess(fixed_point, payoff_matrix):
                # Analyze stability properties
                stability_type = self._classify_stability(fixed_point, payoff_matrix)
                basin_size = self._estimate_basin_size(fixed_point, payoff_matrix)
                convergence_rate = self._estimate_convergence_rate(fixed_point, payoff_matrix)
                invasion_threshold = self._compute_invasion_threshold(fixed_point, payoff_matrix)

                ess = ESS(
                    strategy=fixed_point,
                    stability_type=stability_type,
                    basin_size=basin_size,
                    convergence_rate=convergence_rate,
                    invasion_threshold=invasion_threshold,
                    mutation_stability=self._test_mutation_stability(fixed_point, payoff_matrix)
                )

                ess_strategies.append(ess)

        return ess_strategies

    def _is_ess(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> bool:
        """Test if strategy is evolutionarily stable."""
        n_strategies = len(strategy)

        # ESS condition: for all mutant strategies y != x,
        # either u(x,x) > u(y,x) or u(x,x) = u(y,x) and u(x,y) > u(y,y)

        for i in range(n_strategies):
            if strategy[i] < self.config.ess_threshold:  # Strategy not in support
                # Check invasion condition
                mutant = np.zeros(n_strategies)
                mutant[i] = 1.0

                # Payoffs
                u_x_x = np.dot(strategy, np.dot(payoff_matrix, strategy))
                u_y_x = np.dot(mutant, np.dot(payoff_matrix, strategy))

                if u_y_x > u_x_x + self.config.ess_threshold:
                    return False  # Can be invaded

                if abs(u_y_x - u_x_x) < self.config.ess_threshold:
                    # Check second condition
                    u_x_y = np.dot(strategy, np.dot(payoff_matrix, mutant))
                    u_y_y = np.dot(mutant, np.dot(payoff_matrix, mutant))

                    if u_y_y >= u_x_y - self.config.ess_threshold:
                        return False

        return True

    def _classify_stability(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> StabilityType:
        """Classify the type of stability."""
        if self._is_ess(strategy, payoff_matrix):
            return StabilityType.ESS
        elif self._is_asymptotically_stable(strategy, payoff_matrix):
            return StabilityType.ASYMPTOTIC_STABLE
        elif self._is_lyapunov_stable(strategy, payoff_matrix):
            return StabilityType.LYAPUNOV_STABLE
        elif self._is_neutrally_stable(strategy, payoff_matrix):
            return StabilityType.NEUTRALLY_STABLE
        else:
            return StabilityType.UNSTABLE

    def _is_asymptotically_stable(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> bool:
        """Test asymptotic stability using linearization."""
        # Compute Jacobian of replicator dynamics at fixed point
        n = len(strategy)

        # This is a simplified test
        eigenvalues = np.linalg.eigvals(payoff_matrix - np.outer(strategy, np.dot(payoff_matrix, strategy)))

        # Check if all eigenvalues have negative real parts
        return all(np.real(eig) < -self.config.lyapunov_tolerance for eig in eigenvalues)

    def _is_lyapunov_stable(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> bool:
        """Test Lyapunov stability."""
        # Simplified test: check if small perturbations remain bounded
        replicator = ReplicatorDynamics(payoff_matrix, self.config)

        # Test several perturbations
        for _ in range(10):
            perturbation = np.random.normal(0, self.config.stability_epsilon, len(strategy))
            perturbed_strategy = strategy + perturbation
            perturbed_strategy = np.maximum(0, perturbed_strategy)

            if np.sum(perturbed_strategy) > 0:
                perturbed_strategy = perturbed_strategy / np.sum(perturbed_strategy)

                # Integrate for short time
                t, y = replicator.integrate_dynamics(perturbed_strategy)
                final_state = y[:, -1]

                # Check if remains close to original strategy
                distance = np.linalg.norm(final_state - strategy)
                if distance > 10 * self.config.stability_epsilon:
                    return False

        return True

    def _is_neutrally_stable(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> bool:
        """Test neutral stability."""
        # Simplified: stable but not asymptotically stable
        return (self._is_lyapunov_stable(strategy, payoff_matrix) and
                not self._is_asymptotically_stable(strategy, payoff_matrix))

    def _estimate_basin_size(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> float:
        """Estimate size of basin of attraction."""
        replicator = ReplicatorDynamics(payoff_matrix, self.config)

        converged_count = 0
        total_tests = self.config.basin_resolution

        for _ in range(total_tests):
            # Random initial condition
            initial = np.random.random(len(strategy))
            initial = initial / np.sum(initial)

            # Integrate dynamics
            t, y = replicator.integrate_dynamics(initial)
            final_state = y[:, -1]

            # Check if converged to the strategy
            distance = np.linalg.norm(final_state - strategy)
            if distance < self.config.ess_threshold:
                converged_count += 1

        return converged_count / total_tests

    def _estimate_convergence_rate(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> float:
        """Estimate convergence rate to the strategy."""
        replicator = ReplicatorDynamics(payoff_matrix, self.config)

        # Start near the strategy
        perturbation = np.random.normal(0, self.config.stability_epsilon, len(strategy))
        initial = strategy + perturbation
        initial = np.maximum(0, initial)
        initial = initial / np.sum(initial)

        # Integrate and measure convergence
        t, y = replicator.integrate_dynamics(initial)

        # Compute distances over time
        distances = []
        for i in range(len(t)):
            distance = np.linalg.norm(y[:, i] - strategy)
            distances.append(distance)

        # Estimate exponential decay rate
        if len(distances) > 10:
            log_distances = np.log(np.array(distances[1:]) + 1e-10)
            times = t[1:]

            # Linear regression to estimate decay rate
            if len(times) > 1:
                slope = np.polyfit(times, log_distances, 1)[0]
                return -slope  # Positive convergence rate

        return 0.0

    def _compute_invasion_threshold(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> float:
        """Compute minimum invasion size needed to destabilize strategy."""
        n_strategies = len(strategy)
        min_invasion = 1.0  # Maximum possible

        for i in range(n_strategies):
            if strategy[i] < self.config.ess_threshold:
                # Test invasion by strategy i
                for invasion_size in np.linspace(0.001, 0.1, 50):
                    mixed_population = strategy.copy()
                    mixed_population[i] = invasion_size
                    mixed_population = mixed_population / np.sum(mixed_population)

                    # Check if invasion is successful
                    replicator = ReplicatorDynamics(payoff_matrix, self.config)
                    t, y = replicator.integrate_dynamics(mixed_population)

                    # If invader grows, invasion is successful
                    initial_invader_freq = mixed_population[i]
                    final_invader_freq = y[i, -1]

                    if final_invader_freq > initial_invader_freq * 1.1:
                        min_invasion = min(min_invasion, invasion_size)
                        break

        return min_invasion

    def _test_mutation_stability(self, strategy: np.ndarray, payoff_matrix: np.ndarray) -> bool:
        """Test stability under mutation pressure."""
        # Simplified: check if strategy remains stable under small random mutations
        for _ in range(100):
            mutated_payoff = payoff_matrix + np.random.normal(0, self.config.mutation_strength, payoff_matrix.shape)

            if not self._is_ess(strategy, mutated_payoff):
                return False

        return True


class EvolutionaryStabilityAnalyzer:
    """
    Main analyzer for evolutionary stability of neural weight dynamics.

    Combines replicator dynamics, ESS analysis, and stability assessment
    to understand long-term evolution of weight configurations.
    """

    def __init__(self, config: Optional[EvolutionaryConfig] = None):
        self.config = config or EvolutionaryConfig()
        self.fitness_function = None
        self.ess_analyzer = ESSAnalyzer(self.config)
        self.analysis_history = []
        self.logger = logging.getLogger(__name__)

    def analyze_evolutionary_stability(self, model: nn.Module,
                                     inputs: torch.Tensor) -> List[ESS]:
        """Perform comprehensive evolutionary stability analysis."""
        try:
            start_time = time.time()

            # Initialize fitness function
            self.fitness_function = FitnessFunction(model, self.config)

            # Define strategy space (simplified)
            strategies = self._define_strategy_space(model)

            self.logger.info(f"Analyzing evolutionary stability with {len(strategies)} strategies")

            # Compute payoff matrix
            payoff_matrix = self.fitness_function.compute_payoff_matrix(strategies, inputs)

            # Find ESS
            ess_strategies = self.ess_analyzer.find_ess(payoff_matrix)

            # Analyze evolutionary dynamics
            for ess in ess_strategies:
                ess.computation_time = time.time() - start_time
                ess.fitness_landscape = self._analyze_fitness_landscape(ess.strategy, payoff_matrix)

            # Store analysis
            self.analysis_history.append({
                'timestamp': time.time(),
                'ess_strategies': ess_strategies,
                'num_strategies': len(strategies),
                'computation_time': time.time() - start_time
            })

            self.logger.info(f"Found {len(ess_strategies)} ESS in {time.time() - start_time:.2f}s")

            return ess_strategies

        except Exception as e:
            self.logger.error(f"Evolutionary stability analysis failed: {e}")
            return []

    def _define_strategy_space(self, model: nn.Module) -> List[np.ndarray]:
        """Define discrete strategy space for analysis."""
        # Simplified: create random strategies
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        strategy_dim = min(10, num_params)  # Limit dimension for tractability

        strategies = []

        # Add pure strategies
        for i in range(strategy_dim):
            pure_strategy = np.zeros(strategy_dim)
            pure_strategy[i] = 1.0
            strategies.append(pure_strategy)

        # Add mixed strategies
        for _ in range(20):  # Additional mixed strategies
            mixed_strategy = np.random.random(strategy_dim)
            mixed_strategy = mixed_strategy / np.sum(mixed_strategy)
            strategies.append(mixed_strategy)

        return strategies

    def _analyze_fitness_landscape(self, strategy: np.ndarray,
                                 payoff_matrix: np.ndarray) -> Dict[str, float]:
        """Analyze fitness landscape around ESS."""
        fitness_values = np.dot(payoff_matrix, strategy)

        return {
            "mean_fitness": np.mean(fitness_values),
            "max_fitness": np.max(fitness_values),
            "min_fitness": np.min(fitness_values),
            "fitness_variance": np.var(fitness_values),
            "fitness_range": np.max(fitness_values) - np.min(fitness_values)
        }

    def simulate_evolution(self, model: nn.Module, inputs: torch.Tensor,
                         initial_population: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate evolutionary dynamics over time."""
        if self.fitness_function is None:
            self.fitness_function = FitnessFunction(model, self.config)

        strategies = self._define_strategy_space(model)
        payoff_matrix = self.fitness_function.compute_payoff_matrix(strategies, inputs)

        if initial_population is None:
            # Random initial population
            initial_population = np.random.random(len(strategies))
            initial_population = initial_population / np.sum(initial_population)

        # Run replicator dynamics
        replicator = ReplicatorDynamics(payoff_matrix, self.config)
        t, y = replicator.integrate_dynamics(initial_population)

        return t, y

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of evolutionary stability analysis."""
        if not self.analysis_history:
            return {"message": "No analysis history available"}

        recent_analyses = self.analysis_history[-10:]

        total_ess = sum(len(a['ess_strategies']) for a in recent_analyses)
        computation_times = [a['computation_time'] for a in recent_analyses]

        # Stability type distribution
        stability_types = []
        for analysis in recent_analyses:
            for ess in analysis['ess_strategies']:
                stability_types.append(ess.stability_type.value)

        from collections import Counter
        stability_distribution = Counter(stability_types)

        return {
            "total_analyses": len(self.analysis_history),
            "recent_analyses": len(recent_analyses),
            "total_ess_found": total_ess,
            "average_ess_per_analysis": total_ess / len(recent_analyses) if recent_analyses else 0,
            "stability_type_distribution": dict(stability_distribution),
            "average_computation_time": np.mean(computation_times),
            "config": {
                "dynamics_type": self.config.dynamics_type.value,
                "time_horizon": self.config.time_horizon,
                "ess_threshold": self.config.ess_threshold,
                "mutation_rate": self.config.mutation_rate
            }
        }

    def reset_analysis_history(self) -> None:
        """Reset analysis history."""
        self.analysis_history.clear()
        self.logger.info("Evolutionary analysis history reset")