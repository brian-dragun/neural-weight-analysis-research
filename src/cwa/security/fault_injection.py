"""Fault injection system for hardware fault simulation and analysis."""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
import random
import time
from enum import Enum

from ..core.interfaces import FaultInjectionResult
from ..core.config import FaultInjectionConfig

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of hardware faults that can be simulated."""
    BIT_FLIP = "bit_flip"
    STUCK_AT_ZERO = "stuck_at_zero"
    STUCK_AT_ONE = "stuck_at_one"
    RANDOM_NOISE = "random_noise"
    TRANSIENT_FAULT = "transient_fault"
    PERMANENT_FAULT = "permanent_fault"
    RADIATION_EFFECT = "radiation_effect"
    TEMPERATURE_VARIATION = "temperature_variation"


class FaultInjector:
    """
    Hardware fault injection simulator for studying model robustness.

    Simulates various types of hardware faults that can occur in real
    deployment scenarios, including cosmic radiation, temperature effects,
    and manufacturing defects.
    """

    def __init__(self, config: Optional[FaultInjectionConfig] = None):
        self.config = config or FaultInjectionConfig()
        self.active_faults = []
        self.fault_history = []
        self.permanent_faults = {}
        self.transient_fault_timers = {}

    def inject_faults_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        fault_config: Optional[Dict[str, Any]] = None
    ) -> FaultInjectionResult:
        """
        Inject faults specifically on critical weights identified in Phase A.

        Args:
            model: Target model
            critical_weights: Critical weights from vulnerability analysis
            fault_config: Fault injection configuration

        Returns:
            FaultInjectionResult with detailed fault analysis
        """
        if fault_config is None:
            fault_config = {}

        fault_types = fault_config.get("fault_types", self.config.fault_types)
        injection_rate = fault_config.get("injection_rate", self.config.injection_rate)

        logger.info(f"Injecting faults on {len(critical_weights)} critical weights")
        logger.info(f"Fault types: {fault_types}, injection rate: {injection_rate}")

        start_time = time.time()
        injected_faults = []
        critical_failures = []
        baseline_performance = self._measure_baseline_performance(model)

        # Inject faults by type
        for fault_type in fault_types:
            logger.info(f"Injecting {fault_type} faults")

            fault_results = self._inject_fault_type(
                model, critical_weights, fault_type, injection_rate
            )

            injected_faults.extend(fault_results)

            # Check for critical failures after each fault type
            current_performance = self._measure_performance_after_faults(model)
            performance_degradation = (baseline_performance - current_performance) / baseline_performance

            if performance_degradation > 0.8:  # Critical failure threshold
                critical_failures.append({
                    "fault_type": fault_type,
                    "performance_degradation": performance_degradation,
                    "injected_faults": len(fault_results)
                })

        # Calculate recovery time estimate
        recovery_time = self._estimate_recovery_time(injected_faults)

        # Final performance measurement
        final_performance = self._measure_performance_after_faults(model)
        total_degradation = (baseline_performance - final_performance) / baseline_performance

        result = FaultInjectionResult(
            injected_faults=injected_faults,
            performance_degradation=total_degradation,
            recovery_time=recovery_time,
            critical_failures=[f["fault_type"] for f in critical_failures]
        )

        # Store in history
        self.fault_history.append({
            "timestamp": time.time(),
            "duration": time.time() - start_time,
            "result": result
        })

        logger.info(f"Fault injection complete. Degradation: {total_degradation:.3f}")

        return result

    def inject_bit_flip_faults(
        self,
        model: torch.nn.Module,
        target_weights: List[str],
        flip_probability: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Simulate bit-flip errors in model weights.

        Bit flips can occur due to cosmic radiation, electromagnetic interference,
        or memory corruption in hardware.
        """
        logger.info(f"Injecting bit-flip faults with probability {flip_probability}")

        injected_faults = []
        param_dict = dict(model.named_parameters())

        for layer_name in target_weights:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                fault_results = self._apply_bit_flip_to_parameter(param, layer_name, flip_probability)
                injected_faults.extend(fault_results)

        return injected_faults

    def inject_stuck_at_faults(
        self,
        model: torch.nn.Module,
        fault_type: str,
        target_weights: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simulate stuck-at-zero/one faults.

        These faults represent hardware defects where transistors or memory
        cells become permanently stuck at a particular value.
        """
        logger.info(f"Injecting {fault_type} faults")

        if fault_type not in ["stuck_at_zero", "stuck_at_one"]:
            raise ValueError(f"Invalid stuck-at fault type: {fault_type}")

        stuck_value = 0.0 if fault_type == "stuck_at_zero" else 1.0
        injected_faults = []

        param_dict = dict(model.named_parameters())
        target_layers = target_weights or list(param_dict.keys())

        for layer_name in target_layers:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                fault_results = self._apply_stuck_at_fault(param, layer_name, stuck_value)
                injected_faults.extend(fault_results)

        return injected_faults

    def inject_transient_faults(
        self,
        model: torch.nn.Module,
        duration: float,
        fault_intensity: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Inject temporary faults that recover over time.

        Transient faults represent temporary hardware issues that may
        resolve themselves, such as voltage fluctuations or thermal effects.
        """
        logger.info(f"Injecting transient faults for {duration} seconds")

        injected_faults = []
        start_time = time.time()

        # Store original parameter values for recovery
        original_params = {}
        param_dict = dict(model.named_parameters())

        for name, param in param_dict.items():
            original_params[name] = param.clone()

        # Apply transient faults
        for name, param in param_dict.items():
            if random.random() < self.config.injection_rate:
                fault_mask = torch.rand_like(param) < fault_intensity
                noise = torch.randn_like(param) * 0.1

                with torch.no_grad():
                    param[fault_mask] += noise[fault_mask]

                injected_faults.append({
                    "fault_type": "transient_fault",
                    "layer_name": name,
                    "affected_elements": fault_mask.sum().item(),
                    "start_time": start_time,
                    "duration": duration,
                    "intensity": fault_intensity
                })

        # Schedule recovery
        self.transient_fault_timers[start_time] = {
            "original_params": original_params,
            "recovery_time": start_time + duration,
            "model": model
        }

        return injected_faults

    def simulate_radiation_effects(
        self,
        model: torch.nn.Module,
        intensity: float,
        exposure_time: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Simulate cosmic radiation effects on model parameters.

        High-energy particles can cause bit flips and parameter corruption
        in space applications or high-altitude deployments.
        """
        logger.info(f"Simulating radiation effects with intensity {intensity}")

        injected_faults = []

        # Radiation affects all parameters with different probabilities
        param_dict = dict(model.named_parameters())

        for name, param in param_dict.items():
            # Probability of radiation effect increases with parameter count
            param_count = param.numel()
            radiation_probability = min(intensity * param_count / 1000000, 0.1)

            if random.random() < radiation_probability:
                fault_results = self._apply_radiation_damage(param, name, intensity, exposure_time)
                injected_faults.extend(fault_results)

        return injected_faults

    def simulate_temperature_effects(
        self,
        model: torch.nn.Module,
        temperature_delta: float,
        thermal_coefficient: float = 0.001
    ) -> List[Dict[str, Any]]:
        """
        Simulate temperature variation effects on model weights.

        Temperature changes can cause parameter drift in analog hardware
        and affect the precision of computations.
        """
        logger.info(f"Simulating temperature effects: Δ={temperature_delta}°C")

        injected_faults = []
        param_dict = dict(model.named_parameters())

        for name, param in param_dict.items():
            # Temperature affects different layer types differently
            layer_sensitivity = self._get_temperature_sensitivity(name)

            # Apply temperature-induced parameter drift
            with torch.no_grad():
                drift = thermal_coefficient * temperature_delta * layer_sensitivity
                noise = torch.randn_like(param) * abs(drift)
                param += noise

            injected_faults.append({
                "fault_type": "temperature_variation",
                "layer_name": name,
                "temperature_delta": temperature_delta,
                "thermal_coefficient": thermal_coefficient,
                "layer_sensitivity": layer_sensitivity,
                "affected_parameters": param.numel()
            })

        return injected_faults

    def recover_from_faults(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Attempt to recover model from injected faults.

        Implements various recovery strategies depending on fault types.
        """
        logger.info("Attempting fault recovery")

        recovery_results = {
            "recovered_faults": 0,
            "permanent_faults": 0,
            "recovery_methods": [],
            "recovery_success_rate": 0.0
        }

        # Process transient fault recovery
        current_time = time.time()
        recovered_transient = 0

        for start_time, fault_info in list(self.transient_fault_timers.items()):
            if current_time >= fault_info["recovery_time"]:
                # Recover transient fault
                self._recover_transient_fault(fault_info)
                del self.transient_fault_timers[start_time]
                recovered_transient += 1

        recovery_results["recovered_faults"] += recovered_transient
        if recovered_transient > 0:
            recovery_results["recovery_methods"].append("transient_recovery")

        # Handle permanent faults (require retraining or backup weights)
        permanent_count = len(self.permanent_faults)
        recovery_results["permanent_faults"] = permanent_count

        if permanent_count > 0:
            recovery_results["recovery_methods"].append("permanent_fault_mitigation")

        # Calculate overall recovery success rate
        total_faults = len(self.active_faults)
        if total_faults > 0:
            recovery_results["recovery_success_rate"] = (
                recovery_results["recovered_faults"] / total_faults
            )

        logger.info(f"Recovery complete. Success rate: {recovery_results['recovery_success_rate']:.3f}")

        return recovery_results

    def _inject_fault_type(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        fault_type: str,
        injection_rate: float
    ) -> List[Dict[str, Any]]:
        """Inject a specific type of fault on critical weights."""

        if fault_type == FaultType.BIT_FLIP.value:
            return self._inject_bit_flips_on_critical_weights(model, critical_weights, injection_rate)

        elif fault_type == FaultType.STUCK_AT_ZERO.value:
            return self._inject_stuck_at_faults_on_critical_weights(model, critical_weights, 0.0)

        elif fault_type == FaultType.STUCK_AT_ONE.value:
            return self._inject_stuck_at_faults_on_critical_weights(model, critical_weights, 1.0)

        elif fault_type == FaultType.RANDOM_NOISE.value:
            return self._inject_noise_faults_on_critical_weights(model, critical_weights, injection_rate)

        elif fault_type == FaultType.RADIATION_EFFECT.value:
            return self._inject_radiation_faults_on_critical_weights(model, critical_weights, injection_rate)

        else:
            logger.warning(f"Unknown fault type: {fault_type}")
            return []

    def _inject_bit_flips_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        injection_rate: float
    ) -> List[Dict[str, Any]]:
        """Inject bit-flip faults specifically on critical weights."""
        injected_faults = []
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights:
            if random.random() < injection_rate * vulnerability_score:
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    fault_result = self._apply_bit_flip_to_specific_weight(
                        param, layer_name, param_idx, vulnerability_score
                    )
                    if fault_result:
                        injected_faults.append(fault_result)

        return injected_faults

    def _inject_stuck_at_faults_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        stuck_value: float
    ) -> List[Dict[str, Any]]:
        """Inject stuck-at faults on critical weights."""
        injected_faults = []
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights[:20]:  # Limit for impact
            if random.random() < self.config.injection_rate * vulnerability_score:
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    fault_result = self._apply_stuck_at_to_specific_weight(
                        param, layer_name, param_idx, stuck_value, vulnerability_score
                    )
                    if fault_result:
                        injected_faults.append(fault_result)

        return injected_faults

    def _inject_noise_faults_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        injection_rate: float
    ) -> List[Dict[str, Any]]:
        """Inject random noise faults on critical weights."""
        injected_faults = []
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights:
            if random.random() < injection_rate * vulnerability_score:
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    fault_result = self._apply_noise_to_specific_weight(
                        param, layer_name, param_idx, vulnerability_score
                    )
                    if fault_result:
                        injected_faults.append(fault_result)

        return injected_faults

    def _inject_radiation_faults_on_critical_weights(
        self,
        model: torch.nn.Module,
        critical_weights: List[Tuple[str, int, float]],
        injection_rate: float
    ) -> List[Dict[str, Any]]:
        """Inject radiation-induced faults on critical weights."""
        injected_faults = []
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights:
            # Higher vulnerability weights more likely to be affected by radiation
            radiation_probability = injection_rate * vulnerability_score * 0.5

            if random.random() < radiation_probability:
                if layer_name in param_dict:
                    param = param_dict[layer_name]
                    fault_result = self._apply_radiation_to_specific_weight(
                        param, layer_name, param_idx, vulnerability_score
                    )
                    if fault_result:
                        injected_faults.append(fault_result)

        return injected_faults

    def _apply_bit_flip_to_parameter(
        self, param: torch.Tensor, layer_name: str, flip_probability: float
    ) -> List[Dict[str, Any]]:
        """Apply bit-flip faults to a parameter tensor."""
        injected_faults = []

        # Convert to binary representation and flip random bits
        flat_param = param.flatten()
        num_flips = int(len(flat_param) * flip_probability)

        if num_flips > 0:
            flip_indices = random.sample(range(len(flat_param)), num_flips)

            with torch.no_grad():
                for idx in flip_indices:
                    # Simple bit flip simulation: multiply by -1 or add/subtract small value
                    original_value = flat_param[idx].item()

                    # Simulate bit flip in floating point representation
                    if random.random() < 0.5:
                        flat_param[idx] = -flat_param[idx]  # Sign bit flip
                    else:
                        # Mantissa bit flip (simplified)
                        bit_value = 2 ** random.randint(-10, -1)
                        flat_param[idx] += bit_value if random.random() < 0.5 else -bit_value

                    injected_faults.append({
                        "fault_type": "bit_flip",
                        "layer_name": layer_name,
                        "parameter_index": idx,
                        "original_value": original_value,
                        "new_value": flat_param[idx].item(),
                        "timestamp": time.time()
                    })

        return injected_faults

    def _apply_bit_flip_to_specific_weight(
        self,
        param: torch.Tensor,
        layer_name: str,
        param_idx: int,
        vulnerability_score: float
    ) -> Optional[Dict[str, Any]]:
        """Apply bit flip to a specific weight parameter."""
        flat_param = param.flatten()

        if param_idx < len(flat_param):
            original_value = flat_param[param_idx].item()

            with torch.no_grad():
                # Vulnerability-aware bit flip intensity
                if vulnerability_score > 0.8:
                    # High vulnerability - more severe bit flip
                    flat_param[param_idx] = -flat_param[param_idx]  # Sign flip
                else:
                    # Lower vulnerability - mantissa bit flip
                    bit_value = 2 ** random.randint(-15, -5)
                    flat_param[param_idx] += bit_value if random.random() < 0.5 else -bit_value

            self.active_faults.append({
                "type": "bit_flip",
                "layer": layer_name,
                "index": param_idx,
                "original": original_value
            })

            return {
                "fault_type": "bit_flip",
                "layer_name": layer_name,
                "parameter_index": param_idx,
                "original_value": original_value,
                "new_value": flat_param[param_idx].item(),
                "vulnerability_score": vulnerability_score,
                "timestamp": time.time()
            }

        return None

    def _apply_stuck_at_fault(
        self, param: torch.Tensor, layer_name: str, stuck_value: float
    ) -> List[Dict[str, Any]]:
        """Apply stuck-at fault to random parameters."""
        injected_faults = []
        flat_param = param.flatten()

        # Select random subset of parameters to stick
        num_stuck = int(len(flat_param) * self.config.injection_rate)
        stuck_indices = random.sample(range(len(flat_param)), num_stuck)

        with torch.no_grad():
            for idx in stuck_indices:
                original_value = flat_param[idx].item()
                flat_param[idx] = stuck_value

                # Mark as permanent fault
                fault_key = f"{layer_name}[{idx}]"
                self.permanent_faults[fault_key] = {
                    "original_value": original_value,
                    "stuck_value": stuck_value,
                    "timestamp": time.time()
                }

                injected_faults.append({
                    "fault_type": f"stuck_at_{stuck_value}",
                    "layer_name": layer_name,
                    "parameter_index": idx,
                    "original_value": original_value,
                    "stuck_value": stuck_value,
                    "is_permanent": True,
                    "timestamp": time.time()
                })

        return injected_faults

    def _apply_stuck_at_to_specific_weight(
        self,
        param: torch.Tensor,
        layer_name: str,
        param_idx: int,
        stuck_value: float,
        vulnerability_score: float
    ) -> Optional[Dict[str, Any]]:
        """Apply stuck-at fault to specific weight."""
        flat_param = param.flatten()

        if param_idx < len(flat_param):
            original_value = flat_param[param_idx].item()

            with torch.no_grad():
                flat_param[param_idx] = stuck_value

            # Mark as permanent fault
            fault_key = f"{layer_name}[{param_idx}]"
            self.permanent_faults[fault_key] = {
                "original_value": original_value,
                "stuck_value": stuck_value,
                "vulnerability_score": vulnerability_score,
                "timestamp": time.time()
            }

            return {
                "fault_type": f"stuck_at_{stuck_value}",
                "layer_name": layer_name,
                "parameter_index": param_idx,
                "original_value": original_value,
                "stuck_value": stuck_value,
                "vulnerability_score": vulnerability_score,
                "is_permanent": True,
                "timestamp": time.time()
            }

        return None

    def _apply_noise_to_specific_weight(
        self,
        param: torch.Tensor,
        layer_name: str,
        param_idx: int,
        vulnerability_score: float
    ) -> Optional[Dict[str, Any]]:
        """Apply random noise fault to specific weight."""
        flat_param = param.flatten()

        if param_idx < len(flat_param):
            original_value = flat_param[param_idx].item()

            # Noise intensity based on vulnerability score
            noise_intensity = 0.1 * vulnerability_score
            noise = np.random.normal(0, noise_intensity)

            with torch.no_grad():
                flat_param[param_idx] += noise

            return {
                "fault_type": "random_noise",
                "layer_name": layer_name,
                "parameter_index": param_idx,
                "original_value": original_value,
                "new_value": flat_param[param_idx].item(),
                "noise_intensity": noise_intensity,
                "vulnerability_score": vulnerability_score,
                "timestamp": time.time()
            }

        return None

    def _apply_radiation_damage(
        self,
        param: torch.Tensor,
        layer_name: str,
        intensity: float,
        exposure_time: float
    ) -> List[Dict[str, Any]]:
        """Apply radiation damage to parameter."""
        injected_faults = []
        flat_param = param.flatten()

        # Radiation causes multiple random bit flips
        num_hits = int(intensity * exposure_time * len(flat_param) / 1000)
        hit_indices = random.sample(range(len(flat_param)), min(num_hits, len(flat_param)))

        with torch.no_grad():
            for idx in hit_indices:
                original_value = flat_param[idx].item()

                # Multiple types of radiation damage
                damage_type = random.choice(["bit_flip", "value_corruption", "zero_out"])

                if damage_type == "bit_flip":
                    flat_param[idx] = -flat_param[idx]
                elif damage_type == "value_corruption":
                    flat_param[idx] *= random.uniform(0.1, 10.0)
                else:  # zero_out
                    flat_param[idx] = 0.0

                injected_faults.append({
                    "fault_type": "radiation_effect",
                    "layer_name": layer_name,
                    "parameter_index": idx,
                    "original_value": original_value,
                    "new_value": flat_param[idx].item(),
                    "damage_type": damage_type,
                    "intensity": intensity,
                    "exposure_time": exposure_time,
                    "timestamp": time.time()
                })

        return injected_faults

    def _apply_radiation_to_specific_weight(
        self,
        param: torch.Tensor,
        layer_name: str,
        param_idx: int,
        vulnerability_score: float
    ) -> Optional[Dict[str, Any]]:
        """Apply radiation effect to specific weight."""
        flat_param = param.flatten()

        if param_idx < len(flat_param):
            original_value = flat_param[param_idx].item()

            # Radiation effect intensity based on vulnerability
            effect_intensity = vulnerability_score * random.uniform(0.5, 2.0)

            damage_type = random.choice(["bit_flip", "value_corruption", "partial_zero"])

            with torch.no_grad():
                if damage_type == "bit_flip":
                    flat_param[param_idx] = -flat_param[param_idx]
                elif damage_type == "value_corruption":
                    flat_param[param_idx] *= effect_intensity
                else:  # partial_zero
                    flat_param[param_idx] *= (1.0 - vulnerability_score)

            return {
                "fault_type": "radiation_effect",
                "layer_name": layer_name,
                "parameter_index": param_idx,
                "original_value": original_value,
                "new_value": flat_param[param_idx].item(),
                "damage_type": damage_type,
                "effect_intensity": effect_intensity,
                "vulnerability_score": vulnerability_score,
                "timestamp": time.time()
            }

        return None

    def _get_temperature_sensitivity(self, layer_name: str) -> float:
        """Get temperature sensitivity factor for different layer types."""
        name_lower = layer_name.lower()

        if any(x in name_lower for x in ['embed']):
            return 1.5  # Embeddings more sensitive to temperature
        elif any(x in name_lower for x in ['attn', 'attention']):
            return 1.3  # Attention layers sensitive
        elif any(x in name_lower for x in ['ffn', 'mlp']):
            return 1.1  # Feed-forward moderately sensitive
        else:
            return 1.0  # Default sensitivity

    def _measure_baseline_performance(self, model: torch.nn.Module) -> float:
        """Measure baseline performance before fault injection."""
        model.eval()

        # Simple performance metric: parameter stability
        with torch.no_grad():
            total_params = 0
            total_magnitude = 0.0

            for param in model.parameters():
                total_params += param.numel()
                total_magnitude += param.abs().sum().item()

        return total_magnitude / max(total_params, 1)

    def _measure_performance_after_faults(self, model: torch.nn.Module) -> float:
        """Measure performance after fault injection."""
        return self._measure_baseline_performance(model)

    def _estimate_recovery_time(self, injected_faults: List[Dict[str, Any]]) -> Optional[float]:
        """Estimate time required for fault recovery."""
        if not injected_faults:
            return 0.0

        # Count different fault types
        fault_type_counts = defaultdict(int)
        for fault in injected_faults:
            fault_type_counts[fault.get("fault_type", "unknown")] += 1

        # Estimate recovery time based on fault types and counts
        recovery_time = 0.0

        # Transient faults recover automatically
        recovery_time += fault_type_counts.get("transient_fault", 0) * 0.1

        # Bit flips might be detectable and correctable
        recovery_time += fault_type_counts.get("bit_flip", 0) * 0.5

        # Stuck-at faults require hardware replacement/bypass
        recovery_time += fault_type_counts.get("stuck_at_zero", 0) * 5.0
        recovery_time += fault_type_counts.get("stuck_at_one", 0) * 5.0

        # Radiation effects may require retraining
        recovery_time += fault_type_counts.get("radiation_effect", 0) * 2.0

        # Noise faults might be filtered
        recovery_time += fault_type_counts.get("random_noise", 0) * 1.0

        return recovery_time

    def _recover_transient_fault(self, fault_info: Dict[str, Any]) -> None:
        """Recover from a transient fault by restoring original parameters."""
        model = fault_info["model"]
        original_params = fault_info["original_params"]

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])

        logger.info("Transient fault recovered")

    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fault injection statistics."""
        stats = {
            "active_faults": len(self.active_faults),
            "permanent_faults": len(self.permanent_faults),
            "transient_faults": len(self.transient_fault_timers),
            "fault_history_count": len(self.fault_history),
            "fault_type_distribution": defaultdict(int),
            "layer_fault_distribution": defaultdict(int)
        }

        # Analyze fault distribution
        for fault in self.active_faults:
            stats["fault_type_distribution"][fault.get("type", "unknown")] += 1
            stats["layer_fault_distribution"][fault.get("layer", "unknown")] += 1

        return dict(stats)

    def clear_all_faults(self) -> None:
        """Clear all fault tracking data."""
        self.active_faults.clear()
        self.permanent_faults.clear()
        self.transient_fault_timers.clear()
        logger.info("All fault data cleared")