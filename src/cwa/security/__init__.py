"""Security components for Critical Weight Analysis."""

from .adversarial import (
    AdversarialAttackSimulator,
    register_security_attack,
    get_security_attack,
    list_security_attacks,
    fast_gradient_sign_method,
    projected_gradient_descent,
    textfooler_attack
)
from .targeted_attacks import TargetedAttackSimulator
from .fault_injection import FaultInjector, FaultType
from .defense_mechanisms import (
    DefenseManager,
    register_defense_mechanism,
    get_defense_mechanism,
    list_defense_mechanisms
)
from .weight_protection import CriticalWeightProtector

__all__ = [
    "AdversarialAttackSimulator",
    "TargetedAttackSimulator",
    "FaultInjector",
    "FaultType",
    "DefenseManager",
    "CriticalWeightProtector",
    "register_security_attack",
    "get_security_attack",
    "list_security_attacks",
    "register_defense_mechanism",
    "get_defense_mechanism",
    "list_defense_mechanisms",
    "fast_gradient_sign_method",
    "projected_gradient_descent",
    "textfooler_attack",
]