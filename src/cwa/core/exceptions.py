"""Custom exceptions for Critical Weight Analysis."""


class CWAError(Exception):
    """Base exception for CWA operations."""
    pass


class ModelLoadError(CWAError):
    """Raised when model loading fails."""
    pass


class AnalysisError(CWAError):
    """Raised when analysis operations fail."""
    pass


class ValidationError(CWAError):
    """Raised when validation fails."""
    pass


class ConfigurationError(CWAError):
    """Raised when configuration is invalid."""
    pass


class InsufficientDataError(CWAError):
    """Raised when insufficient data for analysis."""
    pass


class SuperWeightNotFoundError(AnalysisError):
    """Raised when super weight coordinates are not found."""
    pass


class EnsembleDiscoveryError(AnalysisError):
    """Raised when ensemble discovery fails."""
    pass