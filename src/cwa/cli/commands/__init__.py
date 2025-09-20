"""CLI command modules."""

from .basic_commands import basic_app
from .phase_commands import phase_app
from .research_commands import research_app
from .monitoring_commands import monitoring_app
from .analysis_commands import analysis_app

__all__ = [
    "basic_app",
    "phase_app",
    "research_app",
    "monitoring_app",
    "analysis_app"
]