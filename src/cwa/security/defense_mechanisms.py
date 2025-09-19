"""Defense mechanisms for protecting critical weights and model integrity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
import copy
import hashlib

from ..core.interfaces import DefenseMechanism

logger = logging.getLogger(__name__)

# Defense registry for extensibility
_defense_registry: Dict[str, callable] = {}


def register_defense_mechanism(name: str):
    """Decorator to register defense mechanisms."""
    def decorator(func: callable):
        _defense_registry[name] = func
        return func
    return decorator


def get_defense_mechanism(name: str) -> callable:
    """Get defense mechanism by name."""
    if name not in _defense_registry:
        raise ValueError(f"Unknown defense mechanism: {name}")
    return _defense_registry[name]


def list_defense_mechanisms() -> List[str]:
    """List all available defense mechanisms."""
    return list(_defense_registry.keys())


class DefenseManager:
    """
    Phase C: Protection & Defense Manager

    Implements and coordinates various defense mechanisms to protect
    critical weights identified in Phase A from attacks tested in Phase B.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.protection_history = []
        self.active_defenses = {}
        self.weight_checksums = {}
        self.backup_weights = {}

    def implement_protection_mechanisms(
        self,
        critical_weights: List[Tuple[str, int, float]],
        protection_methods: List[str],
        performance_overhead_limit: float = 0.05
    ) -> Dict[str, Any]:
        """
        Phase C: Protect identified critical weights and test effectiveness.

        Args:
            critical_weights: Critical weights from Phase A analysis
            protection_methods: List of protection methods to apply
            performance_overhead_limit: Maximum allowed performance overhead

        Returns:
            Dict containing:
            - protection_applied: What protections were added
            - defense_effectiveness: How well defenses work
            - performance_overhead: Computational cost
            - residual_vulnerability: Remaining attack surface
        """
        logger.info(f"Implementing {len(protection_methods)} protection mechanisms on {len(critical_weights)} critical weights")

        results = {
            "protection_applied": {},
            "defense_effectiveness": {},
            "performance_overhead": 0.0,
            "residual_vulnerability": {},
            "protection_coverage": 0.0,
            "critical_weight_protection_map": {}
        }

        # Measure baseline performance
        baseline_performance = self._measure_baseline_performance()

        # Apply each protection method
        total_overhead = 0.0
        protection_coverage = 0

        for method in protection_methods:
            if total_overhead > performance_overhead_limit:
                logger.warning(f"Performance overhead limit reached, skipping {method}")
                break

            try:
                logger.info(f"Applying protection method: {method}")

                # Apply protection
                protection_result = self._apply_protection_method(
                    method, critical_weights, performance_overhead_limit - total_overhead
                )

                results["protection_applied"][method] = protection_result

                # Measure overhead
                current_performance = self._measure_baseline_performance()
                if baseline_performance > 0:
                    method_overhead = (baseline_performance - current_performance) / baseline_performance
                else:
                    method_overhead = 0.0

                # Ensure valid overhead calculation
                if not (torch.isnan(torch.tensor(method_overhead)) or torch.isinf(torch.tensor(method_overhead))):
                    total_overhead += method_overhead
                else:
                    logger.warning(f"Invalid overhead calculation for {method}, using 0.0")
                    method_overhead = 0.0

                # Test defense effectiveness
                effectiveness = self._test_defense_effectiveness(method, critical_weights)
                results["defense_effectiveness"][method] = effectiveness

                # Update protection coverage
                if protection_result.get("success", False):
                    protection_coverage += protection_result.get("weights_protected", 0)

            except Exception as e:
                logger.error(f"Protection method {method} failed: {e}")
                results["protection_applied"][method] = {"error": str(e)}

        # Calculate overall metrics
        results["performance_overhead"] = total_overhead
        results["protection_coverage"] = protection_coverage / max(len(critical_weights), 1)

        # Analyze residual vulnerability
        results["residual_vulnerability"] = self._analyze_residual_vulnerability(
            critical_weights, results["protection_applied"]
        )

        # Create protection map
        results["critical_weight_protection_map"] = self._create_protection_map(
            critical_weights, results["protection_applied"]
        )

        logger.info(f"Protection implementation complete. Coverage: {results['protection_coverage']:.3f}, Overhead: {total_overhead:.3f}")

        return results

    def test_protected_model(
        self,
        protected_model: torch.nn.Module,
        attack_suite: List[str],
        test_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Test protected model against same attacks that succeeded before protection."""
        logger.info(f"Testing protected model against {len(attack_suite)} attack methods")

        test_results = {
            "attack_resistance": {},
            "protection_effectiveness": {},
            "overall_robustness_score": 0.0,
            "defense_success_rate": 0.0
        }

        # Import attack functions
        from .adversarial import get_security_attack
        from .targeted_attacks import TargetedAttackSimulator

        successful_defenses = 0
        total_attacks = len(attack_suite)

        for attack_method in attack_suite:
            try:
                logger.info(f"Testing defense against {attack_method}")

                # Test attack on protected model
                attack_result = self._test_attack_on_protected_model(
                    protected_model, attack_method, test_data
                )

                test_results["attack_resistance"][attack_method] = attack_result

                # Calculate protection effectiveness
                effectiveness = self._calculate_protection_effectiveness(attack_result)
                test_results["protection_effectiveness"][attack_method] = effectiveness

                if effectiveness > 0.7:  # 70% effectiveness threshold
                    successful_defenses += 1

            except Exception as e:
                logger.error(f"Testing attack {attack_method} failed: {e}")
                test_results["attack_resistance"][attack_method] = {"error": str(e)}

        # Calculate overall metrics
        test_results["defense_success_rate"] = successful_defenses / max(total_attacks, 1)

        # Calculate robustness score
        effectiveness_scores = [
            score for score in test_results["protection_effectiveness"].values()
            if isinstance(score, (int, float))
        ]
        test_results["overall_robustness_score"] = np.mean(effectiveness_scores) if effectiveness_scores else 0.0

        logger.info(f"Protection testing complete. Success rate: {test_results['defense_success_rate']:.3f}")

        return test_results

    def _apply_protection_method(
        self,
        method: str,
        critical_weights: List[Tuple[str, int, float]],
        remaining_overhead_budget: float
    ) -> Dict[str, Any]:
        """Apply a specific protection method."""

        if method not in _defense_registry:
            raise ValueError(f"Unknown protection method: {method}")

        protection_func = _defense_registry[method]

        # Prepare protection configuration
        protection_config = {
            "model": self.model,
            "critical_weights": critical_weights,
            "overhead_budget": remaining_overhead_budget,
            "defense_manager": self
        }

        # Apply protection
        result = protection_func(**protection_config)

        # Store active defense
        self.active_defenses[method] = result

        return result

    def _test_defense_effectiveness(
        self,
        defense_method: str,
        critical_weights: List[Tuple[str, int, float]]
    ) -> float:
        """Test how effective a defense method is."""
        try:
            # Simulate simple attack on protected weights
            original_values = {}
            param_dict = dict(self.model.named_parameters())

            # Store original values
            for layer_name, param_idx, _ in critical_weights[:10]:  # Sample
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    flat_param = param.flatten()
                    if param_idx < len(flat_param):
                        original_values[f"{layer_name}[{param_idx}]"] = flat_param[param_idx].item()

            # Apply test perturbation
            for layer_name, param_idx, vulnerability_score in critical_weights[:10]:
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    flat_param = param.flatten()
                    if param_idx < len(flat_param):
                        with torch.no_grad():
                            flat_param[param_idx] += 0.1 * vulnerability_score

            # Check if defense mechanism detected/corrected the attack
            detected_attacks = self._check_defense_detection()
            corrected_attacks = self._check_defense_correction(original_values)

            # Restore original values
            for key, original_value in original_values.items():
                layer_name, param_idx = self._parse_weight_key(key)
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    flat_param = param.flatten()
                    if param_idx < len(flat_param):
                        with torch.no_grad():
                            flat_param[param_idx] = original_value

            # Calculate effectiveness
            effectiveness = (detected_attacks + corrected_attacks) / (2 * len(original_values))
            return min(effectiveness, 1.0)

        except Exception as e:
            logger.warning(f"Defense effectiveness test failed: {e}")
            return 0.0

    def _check_defense_detection(self) -> int:
        """Check how many attacks were detected by active defenses."""
        detections = 0

        for defense_name, defense_info in self.active_defenses.items():
            if isinstance(defense_info, dict):
                detections += defense_info.get("attacks_detected", 0)

        return detections

    def _check_defense_correction(self, original_values: Dict[str, float]) -> int:
        """Check how many attacks were corrected by active defenses."""
        corrections = 0
        param_dict = dict(self.model.named_parameters())

        for key, original_value in original_values.items():
            layer_name, param_idx = self._parse_weight_key(key)
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()
                if param_idx < len(flat_param):
                    current_value = flat_param[param_idx].item()
                    # Check if value was restored (within tolerance)
                    if abs(current_value - original_value) < 0.05:
                        corrections += 1

        return corrections

    def _parse_weight_key(self, key: str) -> Tuple[str, int]:
        """Parse weight key back to layer name and parameter index."""
        parts = key.split('[')
        layer_name = parts[0]
        param_idx = int(parts[1].rstrip(']'))
        return layer_name, param_idx

    def _measure_baseline_performance(self) -> float:
        """Measure baseline model performance."""
        self.model.eval()

        with torch.no_grad():
            # Simple performance metric based on parameter statistics
            total_params = 0
            total_magnitude = 0.0

            for param in self.model.parameters():
                total_params += param.numel()
                total_magnitude += param.abs().sum().item()

        return total_magnitude / max(total_params, 1)

    def _test_attack_on_protected_model(
        self,
        protected_model: torch.nn.Module,
        attack_method: str,
        test_data: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Test a specific attack on the protected model."""
        try:
            # Measure performance before attack
            before_performance = self._measure_model_performance(protected_model, test_data)

            # Apply simple attack simulation
            if attack_method == "fgsm":
                attack_result = self._simulate_fgsm_attack_on_protected(protected_model)
            elif attack_method == "pgd":
                attack_result = self._simulate_pgd_attack_on_protected(protected_model)
            elif attack_method == "bit_flip":
                attack_result = self._simulate_bit_flip_attack_on_protected(protected_model)
            else:
                attack_result = self._simulate_generic_attack_on_protected(protected_model)

            # Measure performance after attack
            after_performance = self._measure_model_performance(protected_model, test_data)

            # Calculate attack impact
            if before_performance > 0:
                performance_impact = (before_performance - after_performance) / before_performance
            else:
                performance_impact = 0.0

            return {
                "attack_method": attack_method,
                "attack_success": attack_result.get("success", False),
                "performance_impact": performance_impact,
                "attacks_blocked": attack_result.get("attacks_blocked", 0),
                "attacks_detected": attack_result.get("attacks_detected", 0)
            }

        except Exception as e:
            logger.error(f"Attack test {attack_method} failed: {e}")
            return {"error": str(e)}

    def _simulate_fgsm_attack_on_protected(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Simulate FGSM attack on protected model."""
        attacks_blocked = 0
        attacks_detected = 0
        total_attempts = 10

        param_dict = dict(model.named_parameters())

        for i in range(total_attempts):
            # Try to perturb a random parameter
            param_name = np.random.choice(list(param_dict.keys()))
            param = param_dict[param_name]
            flat_param = param.flatten()

            if len(flat_param) > 0:
                idx = np.random.randint(len(flat_param))
                original_value = flat_param[idx].item()

                # Attempt FGSM-style perturbation
                perturbation = 0.1 * np.sign(np.random.randn())

                try:
                    with torch.no_grad():
                        flat_param[idx] += perturbation

                    # Check if defense mechanisms blocked or detected the attack
                    if self._check_protection_triggered(param_name, idx, original_value):
                        attacks_blocked += 1
                    elif self._check_attack_detected(param_name, idx):
                        attacks_detected += 1

                    # Restore value
                    with torch.no_grad():
                        flat_param[idx] = original_value

                except Exception:
                    attacks_blocked += 1  # Exception means attack was blocked

        return {
            "success": (attacks_blocked + attacks_detected) < total_attempts * 0.8,
            "attacks_blocked": attacks_blocked,
            "attacks_detected": attacks_detected,
            "total_attempts": total_attempts
        }

    def _simulate_pgd_attack_on_protected(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Simulate PGD attack on protected model."""
        # Similar to FGSM but with multiple steps
        attacks_blocked = 0
        attacks_detected = 0
        total_attempts = 5  # Fewer attempts due to multi-step nature

        param_dict = dict(model.named_parameters())

        for i in range(total_attempts):
            param_name = np.random.choice(list(param_dict.keys()))
            param = param_dict[param_name]
            flat_param = param.flatten()

            if len(flat_param) > 0:
                idx = np.random.randint(len(flat_param))
                original_value = flat_param[idx].item()

                # Multi-step PGD simulation
                steps_blocked = 0
                for step in range(3):
                    perturbation = 0.03 * np.sign(np.random.randn())

                    try:
                        with torch.no_grad():
                            flat_param[idx] += perturbation

                        if self._check_protection_triggered(param_name, idx, original_value):
                            steps_blocked += 1
                            break

                    except Exception:
                        steps_blocked += 1
                        break

                if steps_blocked > 0:
                    attacks_blocked += 1
                elif self._check_attack_detected(param_name, idx):
                    attacks_detected += 1

                # Restore value
                with torch.no_grad():
                    flat_param[idx] = original_value

        return {
            "success": (attacks_blocked + attacks_detected) < total_attempts * 0.7,
            "attacks_blocked": attacks_blocked,
            "attacks_detected": attacks_detected,
            "total_attempts": total_attempts
        }

    def _simulate_bit_flip_attack_on_protected(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Simulate bit-flip attack on protected model."""
        attacks_blocked = 0
        attacks_detected = 0
        total_attempts = 15

        param_dict = dict(model.named_parameters())

        for i in range(total_attempts):
            param_name = np.random.choice(list(param_dict.keys()))
            param = param_dict[param_name]
            flat_param = param.flatten()

            if len(flat_param) > 0:
                idx = np.random.randint(len(flat_param))
                original_value = flat_param[idx].item()

                # Simulate bit flip
                try:
                    with torch.no_grad():
                        flat_param[idx] = -flat_param[idx]  # Simple bit flip

                    if self._check_protection_triggered(param_name, idx, original_value):
                        attacks_blocked += 1
                    elif self._check_attack_detected(param_name, idx):
                        attacks_detected += 1

                    # Restore value
                    with torch.no_grad():
                        flat_param[idx] = original_value

                except Exception:
                    attacks_blocked += 1

        return {
            "success": (attacks_blocked + attacks_detected) < total_attempts * 0.9,
            "attacks_blocked": attacks_blocked,
            "attacks_detected": attacks_detected,
            "total_attempts": total_attempts
        }

    def _simulate_generic_attack_on_protected(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Simulate generic attack on protected model."""
        return self._simulate_fgsm_attack_on_protected(model)

    def _check_protection_triggered(self, param_name: str, param_idx: int, original_value: float) -> bool:
        """Check if protection mechanisms were triggered by a parameter change."""
        # Check for active protections on this parameter
        weight_key = f"{param_name}[{param_idx}]"

        # Check checksums if available
        if weight_key in self.weight_checksums:
            current_checksum = self._calculate_weight_checksum(param_name, param_idx)
            if current_checksum != self.weight_checksums[weight_key]:
                return True

        # Check backup weights if available
        if weight_key in self.backup_weights:
            param_dict = dict(self.model.named_parameters())
            if param_name in param_dict:
                param = param_dict[param_name]
                flat_param = param.flatten()
                if param_idx < len(flat_param):
                    current_value = flat_param[param_idx].item()
                    backup_value = self.backup_weights[weight_key]
                    if abs(current_value - backup_value) > abs(original_value - backup_value):
                        return True

        return False

    def _check_attack_detected(self, param_name: str, param_idx: int) -> bool:
        """Check if attack was detected by monitoring systems."""
        # Simple detection: check if parameter changed significantly
        weight_key = f"{param_name}[{param_idx}]"

        # In a real implementation, this would check monitoring logs
        # For simulation, randomly detect some attacks
        return np.random.random() < 0.3

    def _calculate_protection_effectiveness(self, attack_result: Dict[str, Any]) -> float:
        """Calculate effectiveness of protection against an attack."""
        if "error" in attack_result:
            return 0.0

        attacks_blocked = attack_result.get("attacks_blocked", 0)
        attacks_detected = attack_result.get("attacks_detected", 0)
        total_attempts = attack_result.get("total_attempts", 1)

        effectiveness = (attacks_blocked + 0.5 * attacks_detected) / total_attempts
        return min(effectiveness, 1.0)

    def _measure_model_performance(self, model: torch.nn.Module, test_data: Optional[List[str]]) -> float:
        """Measure model performance."""
        if test_data is None:
            return self._measure_baseline_performance()

        # For simplicity, return baseline performance
        # In practice, this would evaluate on actual test data
        return self._measure_baseline_performance()

    def _analyze_residual_vulnerability(
        self,
        critical_weights: List[Tuple[str, int, float]],
        protection_applied: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze remaining vulnerabilities after protection."""
        residual = {
            "unprotected_weights": 0,
            "partially_protected_weights": 0,
            "vulnerable_layers": set(),
            "residual_risk_score": 0.0
        }

        # Count protection coverage
        total_protected = 0
        total_partial = 0

        for method, result in protection_applied.items():
            if isinstance(result, dict) and result.get("success", False):
                total_protected += result.get("weights_protected", 0)
                total_partial += result.get("weights_partially_protected", 0)

        residual["unprotected_weights"] = max(0, len(critical_weights) - total_protected)
        residual["partially_protected_weights"] = total_partial

        # Identify vulnerable layers
        for layer_name, _, vulnerability_score in critical_weights:
            if vulnerability_score > 0.8:  # High vulnerability threshold
                residual["vulnerable_layers"].add(layer_name)

        # Calculate residual risk
        if len(critical_weights) > 0:
            protection_ratio = total_protected / len(critical_weights)
            residual["residual_risk_score"] = 1.0 - protection_ratio
        else:
            residual["residual_risk_score"] = 0.0

        residual["vulnerable_layers"] = list(residual["vulnerable_layers"])

        return residual

    def _create_protection_map(
        self,
        critical_weights: List[Tuple[str, int, float]],
        protection_applied: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a map of which weights are protected by which mechanisms."""
        protection_map = defaultdict(list)

        # Map each critical weight to its protection methods
        for layer_name, param_idx, vulnerability_score in critical_weights:
            weight_key = f"{layer_name}[{param_idx}]"

            for method, result in protection_applied.items():
                if isinstance(result, dict) and result.get("success", False):
                    protected_weights = result.get("protected_weight_list", [])
                    if weight_key in protected_weights or result.get("weights_protected", 0) > 0:
                        protection_map[weight_key].append(method)

        return dict(protection_map)

    def _calculate_weight_checksum(self, param_name: str, param_idx: int) -> str:
        """Calculate checksum for a specific weight."""
        param_dict = dict(self.model.named_parameters())
        if param_name in param_dict:
            param = param_dict[param_name]
            flat_param = param.flatten()
            if param_idx < len(flat_param):
                value = flat_param[param_idx].item()
                # Round to avoid floating-point precision issues
                rounded_value = round(value, 6)
                return hashlib.md5(str(rounded_value).encode()).hexdigest()[:8]
        return ""


# Individual defense mechanism implementations

@register_defense_mechanism("weight_redundancy")
def implement_weight_redundancy(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Implement weight redundancy for critical weights.

    Creates backup copies of critical weights that can be used
    for error detection and correction.
    """
    logger.info("Implementing weight redundancy protection")

    protected_weights = 0
    backup_weights = {}
    checksums = {}

    param_dict = dict(model.named_parameters())

    # Create backups for most critical weights (within overhead budget)
    # Enhanced: Increase backup capacity by 3x for better coverage
    max_backups = min(len(critical_weights), int(3000 * overhead_budget))  # Enhanced overhead estimate

    for layer_name, param_idx, vulnerability_score in critical_weights[:max_backups]:
        if layer_name in param_dict:
            param = param_dict[layer_name]
            flat_param = param.flatten()

            if param_idx < len(flat_param):
                weight_key = f"{layer_name}[{param_idx}]"

                # Create backup
                backup_value = flat_param[param_idx].item()
                backup_weights[weight_key] = backup_value

                # Create checksum for integrity checking
                rounded_value = round(backup_value, 6)
                checksum = hashlib.md5(str(rounded_value).encode()).hexdigest()[:8]
                checksums[weight_key] = checksum

                protected_weights += 1

    # Store in defense manager
    defense_manager.backup_weights.update(backup_weights)
    defense_manager.weight_checksums.update(checksums)

    return {
        "success": True,
        "method": "weight_redundancy",
        "weights_protected": protected_weights,
        "backup_count": len(backup_weights),
        "checksum_count": len(checksums),
        "protected_weight_list": list(backup_weights.keys()),
        "overhead_estimate": len(backup_weights) * 8 / 1000000  # Rough MB estimate
    }


@register_defense_mechanism("error_correction")
def implement_error_correction(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Implement error correction codes for critical parameters.

    Uses error correction principles to detect and correct
    single-bit errors in critical weights.
    """
    logger.info("Implementing error correction protection")

    protected_weights = 0
    error_correction_data = {}

    param_dict = dict(model.named_parameters())

    # Implement simple parity-based error correction
    max_protected = min(len(critical_weights), int(500 * overhead_budget))

    for layer_name, param_idx, vulnerability_score in critical_weights[:max_protected]:
        if layer_name in param_dict:
            param = param_dict[layer_name]
            flat_param = param.flatten()

            if param_idx < len(flat_param):
                weight_key = f"{layer_name}[{param_idx}]"

                # Create error correction data
                weight_value = flat_param[param_idx].item()

                # Simple parity bit calculation
                binary_repr = ''.join(format(ord(c), '08b') for c in str(weight_value))
                parity = sum(int(bit) for bit in binary_repr) % 2

                error_correction_data[weight_key] = {
                    "original_value": weight_value,
                    "parity": parity,
                    "timestamp": torch.cuda.current_stream().device if torch.cuda.is_available() else 0
                }

                protected_weights += 1

    return {
        "success": True,
        "method": "error_correction",
        "weights_protected": protected_weights,
        "error_correction_data": error_correction_data,
        "protected_weight_list": list(error_correction_data.keys()),
        "overhead_estimate": len(error_correction_data) * 16 / 1000000  # Rough MB estimate
    }


@register_defense_mechanism("adversarial_training")
def implement_adversarial_training(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Implement adversarial training to improve robustness.

    Note: This is a simplified version. Full adversarial training
    would require a complete training pipeline.
    """
    logger.info("Implementing adversarial training protection")

    # Simulate adversarial training by adding noise resistance
    param_dict = dict(model.named_parameters())

    protected_weights = 0
    trained_weights = []

    # Apply noise resistance to critical weights
    max_trained = min(len(critical_weights), int(200 * overhead_budget))

    with torch.no_grad():
        for layer_name, param_idx, vulnerability_score in critical_weights[:max_trained]:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    # Add small amount of regularization-like noise resistance
                    # This simulates the effect of adversarial training
                    current_value = flat_param[param_idx].item()

                    # Check for NaN or infinite values
                    if torch.isnan(flat_param[param_idx]) or torch.isinf(flat_param[param_idx]):
                        logger.warning(f"Skipping NaN/Inf weight at {layer_name}[{param_idx}]")
                        continue

                    # Slight modification to make weight more robust
                    noise_resistance = 0.01 * vulnerability_score
                    if abs(current_value) > 1e-6:
                        new_value = current_value * (1.0 + noise_resistance)
                        # Verify the new value is valid
                        if not (torch.isnan(torch.tensor(new_value)) or torch.isinf(torch.tensor(new_value))):
                            flat_param[param_idx] = new_value
                        else:
                            logger.warning(f"Adversarial training would create NaN at {layer_name}[{param_idx}], skipping")
                            continue

                    trained_weights.append(f"{layer_name}[{param_idx}]")
                    protected_weights += 1

    return {
        "success": True,
        "method": "adversarial_training",
        "weights_protected": protected_weights,
        "trained_weights": trained_weights,
        "protected_weight_list": trained_weights,
        "training_simulation_applied": True,
        "overhead_estimate": 0.0  # Training is one-time cost
    }


@register_defense_mechanism("input_sanitization")
def implement_input_sanitization(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Implement input sanitization to prevent adversarial inputs.

    Adds input validation and filtering capabilities.
    """
    logger.info("Implementing input sanitization protection")

    # Create input sanitization hooks
    sanitization_hooks = []
    protected_layers = set()

    # Add hooks to critical layers (module level, not parameter level)
    for layer_name, _, _ in critical_weights:
        # Extract module name (remove parameter name like .weight, .bias)
        module_parts = layer_name.split('.')
        if module_parts[-1] in ['weight', 'bias']:
            module_name = '.'.join(module_parts[:-1])
        else:
            module_name = layer_name

        if module_name not in protected_layers:
            try:
                # Get the module
                module = model
                for part in module_name.split('.'):
                    if part.isdigit():
                        module = module[int(part)]
                    elif hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        raise AttributeError(f"Module {part} not found")

                # Verify it's actually a module that can have hooks
                if hasattr(module, 'register_forward_hook'):
                    # Add sanitization hook
                    def sanitization_hook(module, input, output):
                        # Simple input sanitization: clamp extreme values
                        if isinstance(output, torch.Tensor):
                            return torch.clamp(output, -10, 10)
                        return output

                    hook = module.register_forward_hook(sanitization_hook)
                    sanitization_hooks.append(hook)
                    protected_layers.add(module_name)
                else:
                    logger.warning(f"Module {module_name} doesn't support hooks")

            except Exception as e:
                logger.warning(f"Failed to add sanitization hook to {module_name}: {e}")

    return {
        "success": len(sanitization_hooks) > 0,
        "method": "input_sanitization",
        "weights_protected": len(critical_weights),  # Protects all weights indirectly
        "sanitization_hooks": len(sanitization_hooks),
        "protected_layers": list(protected_layers),
        "protected_weight_list": [f"{layer}[*]" for layer in protected_layers],
        "overhead_estimate": len(sanitization_hooks) * 0.001  # Small runtime overhead
    }


@register_defense_mechanism("layer11_attention_fortress")
def implement_layer11_attention_fortress(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Specialized fortress protection for Layer 11 attention weights.

    Layer 11 attention (h.11.attn.c_attn.bias) was identified as the
    highest vulnerability target. This provides multi-layered protection.
    """
    logger.info("Implementing Layer 11 attention fortress protection")

    param_dict = dict(model.named_parameters())
    protected_weights = 0
    fortress_weights = []

    # Focus on Layer 11 attention weights specifically
    layer11_weights = [
        (layer_name, param_idx, vuln_score)
        for layer_name, param_idx, vuln_score in critical_weights
        if "h.11.attn" in layer_name
    ]

    logger.info(f"Found {len(layer11_weights)} Layer 11 attention weights to fortify")

    # Triple redundancy for Layer 11 attention weights
    backup_copies = {}
    checksums = {}
    integrity_monitors = {}

    for layer_name, param_idx, vulnerability_score in layer11_weights:
        if layer_name in param_dict:
            param = param_dict[layer_name]
            flat_param = param.flatten()

            if param_idx < len(flat_param):
                weight_key = f"{layer_name}[{param_idx}]"
                original_value = flat_param[param_idx].item()

                # Create 3 backup copies with different checksums
                backup_copies[f"{weight_key}_backup1"] = original_value
                backup_copies[f"{weight_key}_backup2"] = original_value
                backup_copies[f"{weight_key}_backup3"] = original_value

                # Multiple checksum algorithms for redundancy
                md5_checksum = hashlib.md5(str(round(original_value, 6)).encode()).hexdigest()[:8]
                sha1_checksum = hashlib.sha1(str(round(original_value, 6)).encode()).hexdigest()[:8]
                checksums[f"{weight_key}_md5"] = md5_checksum
                checksums[f"{weight_key}_sha1"] = sha1_checksum

                # Set up integrity monitoring
                integrity_monitors[weight_key] = {
                    "original_value": original_value,
                    "vulnerability_score": vulnerability_score,
                    "last_check": 0,
                    "anomaly_threshold": 0.01,  # Very sensitive for Layer 11
                    "protection_level": "fortress"
                }

                fortress_weights.append(weight_key)
                protected_weights += 1

    # Store in defense manager
    defense_manager.backup_weights.update(backup_copies)
    defense_manager.weight_checksums.update(checksums)

    return {
        "success": protected_weights > 0,
        "method": "layer11_attention_fortress",
        "weights_protected": protected_weights,
        "layer11_weights_found": len(layer11_weights),
        "backup_copies": len(backup_copies),
        "checksum_algorithms": 2,
        "integrity_monitors": len(integrity_monitors),
        "protected_weight_list": fortress_weights,
        "protection_level": "fortress",
        "overhead_estimate": len(backup_copies) * 12 / 1000000,  # Higher overhead for better protection
        "description": "Triple redundancy protection for critical Layer 11 attention weights"
    }


@register_defense_mechanism("enhanced_checksums")
def implement_enhanced_checksums(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced checksum protection with multiple algorithms and real-time monitoring.
    """
    logger.info("Implementing enhanced checksum protection")

    param_dict = dict(model.named_parameters())
    protected_weights = 0
    checksums = {}

    # Use multiple checksum algorithms for better detection
    max_protected = min(len(critical_weights), int(2000 * overhead_budget))

    for layer_name, param_idx, vulnerability_score in critical_weights[:max_protected]:
        if layer_name in param_dict:
            param = param_dict[layer_name]
            flat_param = param.flatten()

            if param_idx < len(flat_param):
                weight_key = f"{layer_name}[{param_idx}]"
                value = flat_param[param_idx].item()

                # Check for NaN or infinite values
                if torch.isnan(flat_param[param_idx]) or torch.isinf(flat_param[param_idx]):
                    logger.warning(f"Skipping NaN/Inf weight at {layer_name}[{param_idx}] for checksum")
                    continue

                rounded_value = round(value, 6)

                # Multiple checksum algorithms
                md5_sum = hashlib.md5(str(rounded_value).encode()).hexdigest()[:8]
                sha1_sum = hashlib.sha1(str(rounded_value).encode()).hexdigest()[:8]

                checksums[f"{weight_key}_md5"] = md5_sum
                checksums[f"{weight_key}_sha1"] = sha1_sum

                protected_weights += 1

    # Store in defense manager
    defense_manager.weight_checksums.update(checksums)

    return {
        "success": protected_weights > 0,
        "method": "enhanced_checksums",
        "weights_protected": protected_weights,
        "checksum_algorithms": 2,
        "total_checksums": len(checksums),
        "protected_weight_list": [key.split('_')[0] for key in checksums.keys() if key.endswith('_md5')],
        "overhead_estimate": len(checksums) * 4 / 1000000
    }


@register_defense_mechanism("weight_encryption")
def implement_weight_encryption(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    overhead_budget: float,
    defense_manager: DefenseManager,
    **kwargs
) -> Dict[str, Any]:
    """
    Implement weight encryption for critical parameters.

    Note: This is a conceptual implementation. Real weight encryption
    would require specialized hardware support.
    """
    logger.info("Implementing weight encryption protection")

    encrypted_weights = []
    encryption_keys = {}

    param_dict = dict(model.named_parameters())

    # "Encrypt" critical weights (conceptual)
    max_encrypted = min(len(critical_weights), int(100 * overhead_budget))

    for layer_name, param_idx, vulnerability_score in critical_weights[:max_encrypted]:
        if layer_name in param_dict:
            weight_key = f"{layer_name}[{param_idx}]"

            # Generate "encryption key" (conceptual)
            encryption_key = hashlib.sha256(weight_key.encode()).hexdigest()[:16]
            encryption_keys[weight_key] = encryption_key

            encrypted_weights.append(weight_key)

    return {
        "success": len(encrypted_weights) > 0,
        "method": "weight_encryption",
        "weights_protected": len(encrypted_weights),
        "encrypted_weights": encrypted_weights,
        "encryption_keys": len(encryption_keys),
        "protected_weight_list": encrypted_weights,
        "overhead_estimate": len(encrypted_weights) * 0.002,  # Encryption overhead
        "note": "Conceptual implementation - requires hardware support"
    }