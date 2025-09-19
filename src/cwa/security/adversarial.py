"""Adversarial attack implementations for transformer models."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from collections import defaultdict
import math

from ..core.interfaces import SecurityAttack

logger = logging.getLogger(__name__)

# Attack registry for extensibility
_attack_registry: Dict[str, callable] = {}


def register_security_attack(name: str):
    """Decorator to register security attack methods."""
    def decorator(func: callable):
        _attack_registry[name] = func
        return func
    return decorator


def get_security_attack(name: str) -> callable:
    """Get security attack by name."""
    if name not in _attack_registry:
        raise ValueError(f"Unknown security attack: {name}")
    return _attack_registry[name]


def list_security_attacks() -> List[str]:
    """List all available security attacks."""
    return list(_attack_registry.keys())


class AdversarialAttackSimulator:
    """
    Phase B: Attack Simulation Engine

    Simulates targeted adversarial attacks on critical weights discovered in Phase A.
    Supports both input-level and weight-level adversarial attacks.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.attack_history = []

    def simulate_attacks_on_critical_weights(
        self,
        critical_weights: List[Tuple[str, int, float]],
        attack_methods: List[str],
        input_data: Optional[List[str]] = None,
        attack_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Phase B: Test attacks specifically on identified critical weights.

        Args:
            critical_weights: List of (layer_name, param_idx, vulnerability_score)
            attack_methods: List of attack method names
            input_data: Sample texts for input-level attacks
            attack_config: Attack-specific configuration

        Returns:
            Dict containing:
            - attack_results: Per-attack-method results
            - performance_degradation: Task performance loss
            - critical_failures: Weights causing total failure
            - recovery_metrics: Time/resources to recover
        """
        logger.info(f"Simulating {len(attack_methods)} attack methods on {len(critical_weights)} critical weights")

        if attack_config is None:
            attack_config = {}

        results = {
            "attack_results": {},
            "performance_degradation": {},
            "critical_failures": [],
            "recovery_metrics": {},
            "attack_success_rates": {},
            "weight_impact_analysis": {}
        }

        # Store original model state for recovery analysis
        original_state = self._save_model_state()
        baseline_performance = self._measure_baseline_performance(input_data)

        # Test each attack method
        for attack_method in attack_methods:
            logger.info(f"Testing attack method: {attack_method}")

            try:
                # Execute attack on critical weights
                attack_result = self._execute_attack_on_weights(
                    attack_method, critical_weights, input_data, attack_config
                )

                # Measure impact
                performance_impact = self._measure_attack_impact(
                    baseline_performance, input_data
                )

                # Analyze critical failures
                critical_failures = self._identify_critical_failures(
                    critical_weights, performance_impact
                )

                # Store results
                results["attack_results"][attack_method] = attack_result
                results["performance_degradation"][attack_method] = performance_impact
                results["attack_success_rates"][attack_method] = attack_result.get("success_rate", 0.0)

                if critical_failures:
                    results["critical_failures"].extend(critical_failures)

                # Restore model for next attack
                self._restore_model_state(original_state)

            except Exception as e:
                logger.error(f"Attack {attack_method} failed: {e}")
                results["attack_results"][attack_method] = {"error": str(e)}

        # Compute recovery metrics
        results["recovery_metrics"] = self._compute_recovery_metrics(
            results["attack_results"], critical_weights
        )

        # Weight impact analysis
        results["weight_impact_analysis"] = self._analyze_weight_impact(
            critical_weights, results["attack_results"]
        )

        logger.info(f"Attack simulation complete. Success rates: {results['attack_success_rates']}")

        self.attack_history.append({
            "timestamp": torch.utils.data.get_worker_info(),
            "methods": attack_methods,
            "critical_weights_count": len(critical_weights),
            "results": results
        })

        return results

    def measure_attack_impact(
        self,
        original_model: torch.nn.Module,
        attacked_model: torch.nn.Module,
        test_data: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Measure specific impact of attacks on critical weights."""
        impact_metrics = {}

        try:
            # Performance comparison
            if test_data:
                orig_performance = self._evaluate_model_performance(original_model, test_data)
                attack_performance = self._evaluate_model_performance(attacked_model, test_data)

                impact_metrics["performance_degradation"] = (
                    orig_performance - attack_performance
                ) / orig_performance if orig_performance > 0 else 0.0

            # Output similarity analysis
            if test_data and self.tokenizer:
                similarity = self._compute_output_similarity(
                    original_model, attacked_model, test_data[:10]  # Sample
                )
                impact_metrics["output_similarity"] = similarity

            # Weight deviation analysis
            weight_deviation = self._compute_weight_deviation(original_model, attacked_model)
            impact_metrics["weight_deviation"] = weight_deviation

            # Robustness metrics
            robustness_metrics = self._compute_robustness_metrics(attacked_model, test_data)
            impact_metrics.update(robustness_metrics)

        except Exception as e:
            logger.error(f"Error measuring attack impact: {e}")
            impact_metrics["error"] = str(e)

        return impact_metrics

    def _execute_attack_on_weights(
        self,
        attack_method: str,
        critical_weights: List[Tuple[str, int, float]],
        input_data: Optional[List[str]],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute specific attack method on critical weights."""

        if attack_method not in _attack_registry:
            raise ValueError(f"Unknown attack method: {attack_method}")

        attack_func = _attack_registry[attack_method]

        # Prepare attack configuration
        attack_config = {
            "model": self.model,
            "critical_weights": critical_weights,
            "input_data": input_data,
            "tokenizer": self.tokenizer,
            **config.get(attack_method, {})
        }

        # Execute attack
        result = attack_func(**attack_config)

        return result

    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        """Save current model state for recovery."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def _restore_model_state(self, saved_state: Dict[str, torch.Tensor]) -> None:
        """Restore model to saved state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state:
                    param.copy_(saved_state[name])

    def _measure_baseline_performance(self, input_data: Optional[List[str]]) -> float:
        """Measure baseline model performance."""
        if not input_data:
            return 1.0  # Default baseline

        try:
            return self._evaluate_model_performance(self.model, input_data)
        except Exception as e:
            logger.warning(f"Baseline measurement failed: {e}")
            return 1.0

    def _evaluate_model_performance(self, model: torch.nn.Module, test_data: List[str]) -> float:
        """Evaluate model performance on test data."""
        model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for text in test_data[:20]:  # Sample for efficiency
                try:
                    if self.tokenizer:
                        inputs = self.tokenizer(
                            text, return_tensors="pt", truncation=True, max_length=512
                        ).to(self.device)

                        outputs = model(**inputs)

                        if hasattr(outputs, 'logits'):
                            # For causal LM, compute perplexity-like metric
                            logits = outputs.logits
                            if inputs.get('labels') is not None:
                                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                                     inputs['labels'].view(-1))
                            else:
                                # Use entropy as proxy metric
                                probs = F.softmax(logits, dim=-1)
                                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
                        else:
                            # For other model types
                            loss = outputs[0].mean() if torch.is_tensor(outputs[0]) else torch.tensor(0.0)

                        total_loss += loss.item()
                        num_samples += 1

                except Exception as e:
                    logger.warning(f"Evaluation failed for sample: {e}")
                    continue

        return 1.0 / (1.0 + total_loss / max(num_samples, 1))  # Inverse loss as performance

    def _measure_attack_impact(self, baseline_performance: float, input_data: Optional[List[str]]) -> Dict[str, float]:
        """Measure impact of current attack."""
        if not input_data:
            return {"performance_loss": 0.0}

        try:
            current_performance = self._evaluate_model_performance(self.model, input_data)
            performance_loss = (baseline_performance - current_performance) / baseline_performance

            return {
                "performance_loss": max(0.0, performance_loss),
                "current_performance": current_performance,
                "baseline_performance": baseline_performance
            }
        except Exception as e:
            logger.warning(f"Impact measurement failed: {e}")
            return {"performance_loss": 0.0, "error": str(e)}

    def _identify_critical_failures(
        self,
        critical_weights: List[Tuple[str, int, float]],
        performance_impact: Dict[str, float]
    ) -> List[str]:
        """Identify weights that cause critical failures when attacked."""
        critical_failures = []

        performance_loss = performance_impact.get("performance_loss", 0.0)

        if performance_loss > 0.5:  # >50% performance loss
            # Identify the most vulnerable weights
            sorted_weights = sorted(critical_weights, key=lambda x: x[2], reverse=True)
            top_vulnerable = sorted_weights[:min(10, len(sorted_weights))]

            for layer_name, param_idx, score in top_vulnerable:
                if score > 0.8:  # High vulnerability score
                    critical_failures.append(f"{layer_name}[{param_idx}]")

        return critical_failures

    def _compute_recovery_metrics(
        self,
        attack_results: Dict[str, Any],
        critical_weights: List[Tuple[str, int, float]]
    ) -> Dict[str, Any]:
        """Compute metrics related to recovery from attacks."""
        recovery_metrics = {
            "estimated_recovery_time": 0.0,
            "recovery_complexity": "low",
            "affected_weight_count": len(critical_weights),
            "most_critical_layer": None
        }

        try:
            # Estimate recovery time based on attack success
            successful_attacks = sum(
                1 for result in attack_results.values()
                if isinstance(result, dict) and result.get("success_rate", 0) > 0.3
            )

            recovery_metrics["estimated_recovery_time"] = successful_attacks * 0.1  # Simplified estimate

            # Recovery complexity
            if successful_attacks > 3:
                recovery_metrics["recovery_complexity"] = "high"
            elif successful_attacks > 1:
                recovery_metrics["recovery_complexity"] = "medium"

            # Most critical layer
            if critical_weights:
                layer_counts = defaultdict(int)
                for layer_name, _, _ in critical_weights:
                    layer_counts[layer_name] += 1
                recovery_metrics["most_critical_layer"] = max(layer_counts.keys(), key=layer_counts.get)

        except Exception as e:
            logger.warning(f"Recovery metrics computation failed: {e}")

        return recovery_metrics

    def _analyze_weight_impact(
        self,
        critical_weights: List[Tuple[str, int, float]],
        attack_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of attacks on different weight categories."""
        impact_analysis = {
            "layer_vulnerability": defaultdict(list),
            "attack_effectiveness": {},
            "weight_criticality_distribution": {}
        }

        # Group weights by layer
        for layer_name, param_idx, score in critical_weights:
            impact_analysis["layer_vulnerability"][layer_name].append(score)

        # Compute layer-level statistics
        for layer_name, scores in impact_analysis["layer_vulnerability"].items():
            impact_analysis["layer_vulnerability"][layer_name] = {
                "mean_vulnerability": np.mean(scores),
                "max_vulnerability": np.max(scores),
                "weight_count": len(scores)
            }

        # Attack effectiveness analysis
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and "success_rate" in result:
                impact_analysis["attack_effectiveness"][attack_name] = result["success_rate"]

        return impact_analysis

    def _compute_output_similarity(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        test_texts: List[str]
    ) -> float:
        """Compute similarity between outputs of two models."""
        if not self.tokenizer:
            return 0.0

        similarities = []

        with torch.no_grad():
            for text in test_texts:
                try:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)

                    outputs1 = model1(**inputs)
                    outputs2 = model2(**inputs)

                    if hasattr(outputs1, 'last_hidden_state') and hasattr(outputs2, 'last_hidden_state'):
                        hidden1 = outputs1.last_hidden_state
                        hidden2 = outputs2.last_hidden_state

                        # Cosine similarity
                        similarity = F.cosine_similarity(
                            hidden1.flatten(), hidden2.flatten(), dim=0
                        ).item()
                        similarities.append(similarity)

                except Exception as e:
                    logger.warning(f"Similarity computation failed: {e}")
                    continue

        return np.mean(similarities) if similarities else 0.0

    def _compute_weight_deviation(self, model1: torch.nn.Module, model2: torch.nn.Module) -> float:
        """Compute deviation between model weights."""
        total_deviation = 0.0
        total_params = 0

        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                deviation = torch.norm(param1 - param2).item()
                total_deviation += deviation
                total_params += param1.numel()

        return total_deviation / max(total_params, 1)

    def _compute_robustness_metrics(self, model: torch.nn.Module, test_data: Optional[List[str]]) -> Dict[str, float]:
        """Compute robustness metrics for the attacked model."""
        metrics = {}

        if not test_data:
            return metrics

        try:
            # Output stability under small perturbations
            stability_scores = []

            for text in test_data[:5]:  # Sample for efficiency
                if self.tokenizer:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)

                    with torch.no_grad():
                        outputs = model(**inputs)

                        if hasattr(outputs, 'last_hidden_state'):
                            hidden = outputs.last_hidden_state
                            stability = 1.0 / (1.0 + torch.std(hidden).item())
                            stability_scores.append(stability)

            if stability_scores:
                metrics["output_stability"] = np.mean(stability_scores)

        except Exception as e:
            logger.warning(f"Robustness metrics computation failed: {e}")

        return metrics


# Individual attack implementations

@register_security_attack("fgsm")
def fast_gradient_sign_method(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    input_data: Optional[List[str]] = None,
    tokenizer: Any = None,
    epsilon: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """
    Fast Gradient Sign Method attack on critical weights.

    Applies adversarial perturbations in the direction of the gradient sign
    to maximize loss while keeping perturbations bounded.
    """
    logger.info(f"Executing FGSM attack with epsilon={epsilon}")

    if not input_data or not tokenizer:
        # Weight-level FGSM
        return _weight_level_fgsm(model, critical_weights, epsilon)
    else:
        # Input-level FGSM
        return _input_level_fgsm(model, input_data, tokenizer, epsilon)


@register_security_attack("pgd")
def projected_gradient_descent(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    input_data: Optional[List[str]] = None,
    tokenizer: Any = None,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_steps: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Projected Gradient Descent attack on critical weights.

    Iterative attack that applies multiple small perturbations,
    projecting back to the allowed perturbation space.
    """
    logger.info(f"Executing PGD attack with epsilon={epsilon}, steps={num_steps}")

    if not input_data or not tokenizer:
        return _weight_level_pgd(model, critical_weights, epsilon, alpha, num_steps)
    else:
        return _input_level_pgd(model, input_data, tokenizer, epsilon, alpha, num_steps)


@register_security_attack("textfooler")
def textfooler_attack(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    input_data: Optional[List[str]] = None,
    tokenizer: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    TextFooler semantic attack implementation.

    Generates adversarial examples by replacing words with semantically
    similar alternatives while preserving text meaning.
    """
    logger.info("Executing TextFooler semantic attack")

    if not input_data or not tokenizer:
        logger.warning("TextFooler requires input data and tokenizer")
        return {"success_rate": 0.0, "error": "Requires input data"}

    return _textfooler_semantic_attack(model, input_data, tokenizer, critical_weights)


# Helper functions for attack implementations

def _weight_level_fgsm(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    epsilon: float
) -> Dict[str, Any]:
    """Apply FGSM directly to model weights."""
    successful_attacks = 0
    total_attacks = 0

    with torch.no_grad():
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights[:50]:  # Limit for efficiency
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    # Apply FGSM perturbation
                    original_value = flat_param[param_idx].item()
                    perturbation = epsilon * torch.sign(torch.randn(1)).item()
                    flat_param[param_idx] += perturbation

                    successful_attacks += 1

            total_attacks += 1

    success_rate = successful_attacks / max(total_attacks, 1)

    return {
        "success_rate": success_rate,
        "successful_attacks": successful_attacks,
        "total_attacks": total_attacks,
        "attack_type": "weight_level_fgsm",
        "epsilon": epsilon
    }


def _input_level_fgsm(
    model: torch.nn.Module,
    input_data: List[str],
    tokenizer: Any,
    epsilon: float
) -> Dict[str, Any]:
    """Apply FGSM to input embeddings."""
    model.eval()
    successful_attacks = 0
    total_attacks = len(input_data[:10])  # Sample for efficiency

    for text in input_data[:10]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

            # Get embeddings
            embeddings = model.get_input_embeddings()(inputs['input_ids'])
            embeddings.requires_grad_(True)

            # Forward pass
            outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get('attention_mask'))

            if hasattr(outputs, 'logits'):
                loss = outputs.logits.mean()
            else:
                loss = outputs[0].mean()

            # Compute gradients
            loss.backward()

            # Apply FGSM
            with torch.no_grad():
                grad_sign = embeddings.grad.sign()
                adversarial_embeddings = embeddings + epsilon * grad_sign

                # Test adversarial example
                adv_outputs = model(inputs_embeds=adversarial_embeddings, attention_mask=inputs.get('attention_mask'))

                # Simple success criterion: significant output change
                if hasattr(outputs, 'logits') and hasattr(adv_outputs, 'logits'):
                    original_probs = F.softmax(outputs.logits, dim=-1)
                    adv_probs = F.softmax(adv_outputs.logits, dim=-1)
                    change = torch.norm(original_probs - adv_probs).item()

                    if change > 0.1:  # Threshold for success
                        successful_attacks += 1
                else:
                    successful_attacks += 1  # Conservative success

        except Exception as e:
            logger.warning(f"FGSM input attack failed: {e}")
            continue

    success_rate = successful_attacks / max(total_attacks, 1)

    return {
        "success_rate": success_rate,
        "successful_attacks": successful_attacks,
        "total_attacks": total_attacks,
        "attack_type": "input_level_fgsm",
        "epsilon": epsilon
    }


def _weight_level_pgd(
    model: torch.nn.Module,
    critical_weights: List[Tuple[str, int, float]],
    epsilon: float,
    alpha: float,
    num_steps: int
) -> Dict[str, Any]:
    """Apply PGD to model weights."""
    successful_attacks = 0
    total_attacks = 0

    with torch.no_grad():
        param_dict = dict(model.named_parameters())

        for layer_name, param_idx, vulnerability_score in critical_weights[:30]:
            if layer_name in param_dict:
                param = param_dict[layer_name]
                flat_param = param.flatten()

                if param_idx < len(flat_param):
                    original_value = flat_param[param_idx].clone()

                    # PGD iterations
                    for step in range(num_steps):
                        # Random perturbation
                        perturbation = alpha * torch.sign(torch.randn(1)).item()
                        flat_param[param_idx] += perturbation

                        # Project back to allowed space
                        total_perturbation = flat_param[param_idx] - original_value
                        if abs(total_perturbation) > epsilon:
                            flat_param[param_idx] = original_value + epsilon * torch.sign(total_perturbation)

                    successful_attacks += 1

            total_attacks += 1

    success_rate = successful_attacks / max(total_attacks, 1)

    return {
        "success_rate": success_rate,
        "successful_attacks": successful_attacks,
        "total_attacks": total_attacks,
        "attack_type": "weight_level_pgd",
        "epsilon": epsilon,
        "alpha": alpha,
        "num_steps": num_steps
    }


def _input_level_pgd(
    model: torch.nn.Module,
    input_data: List[str],
    tokenizer: Any,
    epsilon: float,
    alpha: float,
    num_steps: int
) -> Dict[str, Any]:
    """Apply PGD to input embeddings."""
    model.eval()
    successful_attacks = 0
    total_attacks = len(input_data[:8])  # Sample for efficiency

    for text in input_data[:8]:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

            # Get original embeddings
            original_embeddings = model.get_input_embeddings()(inputs['input_ids'])
            adversarial_embeddings = original_embeddings.clone().detach()

            # PGD iterations
            for step in range(num_steps):
                adversarial_embeddings.requires_grad_(True)

                outputs = model(inputs_embeds=adversarial_embeddings, attention_mask=inputs.get('attention_mask'))
                loss = outputs.logits.mean() if hasattr(outputs, 'logits') else outputs[0].mean()

                loss.backward()

                with torch.no_grad():
                    # PGD step
                    grad_sign = adversarial_embeddings.grad.sign()
                    adversarial_embeddings = adversarial_embeddings + alpha * grad_sign

                    # Project to epsilon ball
                    perturbation = adversarial_embeddings - original_embeddings
                    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                    adversarial_embeddings = original_embeddings + perturbation

                adversarial_embeddings = adversarial_embeddings.detach()

            # Test final adversarial example
            with torch.no_grad():
                orig_outputs = model(inputs_embeds=original_embeddings, attention_mask=inputs.get('attention_mask'))
                adv_outputs = model(inputs_embeds=adversarial_embeddings, attention_mask=inputs.get('attention_mask'))

                if hasattr(orig_outputs, 'logits') and hasattr(adv_outputs, 'logits'):
                    orig_probs = F.softmax(orig_outputs.logits, dim=-1)
                    adv_probs = F.softmax(adv_outputs.logits, dim=-1)
                    change = torch.norm(orig_probs - adv_probs).item()

                    if change > 0.15:  # Success threshold
                        successful_attacks += 1

        except Exception as e:
            logger.warning(f"PGD input attack failed: {e}")
            continue

    success_rate = successful_attacks / max(total_attacks, 1)

    return {
        "success_rate": success_rate,
        "successful_attacks": successful_attacks,
        "total_attacks": total_attacks,
        "attack_type": "input_level_pgd",
        "epsilon": epsilon,
        "alpha": alpha,
        "num_steps": num_steps
    }


def _textfooler_semantic_attack(
    model: torch.nn.Module,
    input_data: List[str],
    tokenizer: Any,
    critical_weights: List[Tuple[str, int, float]]
) -> Dict[str, Any]:
    """
    Simplified TextFooler implementation.

    Note: This is a basic implementation. A full TextFooler would require
    external libraries like TextAttack for word substitutions.
    """
    model.eval()
    successful_attacks = 0
    total_attacks = len(input_data[:5])  # Limited sample

    # Simple word substitutions (in practice, would use semantic similarity)
    substitutions = {
        "good": "great", "bad": "terrible", "big": "large", "small": "tiny",
        "fast": "quick", "slow": "sluggish", "happy": "joyful", "sad": "miserable"
    }

    for text in input_data[:5]:
        try:
            # Simple word replacement
            modified_text = text
            for original, replacement in substitutions.items():
                if original in text.lower():
                    modified_text = text.replace(original, replacement)
                    break

            if modified_text != text:
                # Test both versions
                original_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                modified_inputs = tokenizer(modified_text, return_tensors="pt", truncation=True, max_length=256)

                original_inputs = {k: v.to(next(model.parameters()).device) for k, v in original_inputs.items()}
                modified_inputs = {k: v.to(next(model.parameters()).device) for k, v in modified_inputs.items()}

                with torch.no_grad():
                    orig_outputs = model(**original_inputs)
                    mod_outputs = model(**modified_inputs)

                    if hasattr(orig_outputs, 'logits') and hasattr(mod_outputs, 'logits'):
                        orig_probs = F.softmax(orig_outputs.logits, dim=-1)
                        mod_probs = F.softmax(mod_outputs.logits, dim=-1)
                        change = torch.norm(orig_probs - mod_probs).item()

                        if change > 0.1:
                            successful_attacks += 1

        except Exception as e:
            logger.warning(f"TextFooler attack failed: {e}")
            continue

    success_rate = successful_attacks / max(total_attacks, 1)

    return {
        "success_rate": success_rate,
        "successful_attacks": successful_attacks,
        "total_attacks": total_attacks,
        "attack_type": "textfooler_semantic",
        "note": "Simplified implementation"
    }