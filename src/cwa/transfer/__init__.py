"""
Cross-Architecture Transfer Analysis Module

This module implements advanced transfer learning analysis to understand how
critical weight vulnerabilities and patterns transfer across different neural
network architectures and domains.

Key Components:
- TransferAnalyzer: Main transfer pattern analysis
- ArchitectureMapper: Cross-architecture weight mapping
- VulnerabilityTransferDetector: Vulnerability pattern transfer
- DomainTransferAnalyzer: Cross-domain transfer analysis
"""

from .transfer_analyzer import TransferAnalyzer, TransferPattern, TransferType
from .architecture_mapper import ArchitectureMapper, MappingStrategy, WeightMapping
from .vulnerability_transfer import VulnerabilityTransferDetector, VulnerabilityTransfer
from .domain_transfer import DomainTransferAnalyzer, DomainAlignment, TransferMetric

__all__ = [
    "TransferAnalyzer",
    "TransferPattern",
    "TransferType",
    "ArchitectureMapper",
    "MappingStrategy",
    "WeightMapping",
    "VulnerabilityTransferDetector",
    "VulnerabilityTransfer",
    "DomainTransferAnalyzer",
    "DomainAlignment",
    "TransferMetric",
]