"""Streamlined Command Line Interface for Critical Weight Analysis tool."""

import typer
from rich.console import Console

# Import modular command apps
from .commands import (
    basic_app,
    phase_app,
    research_app,
    monitoring_app,
    analysis_app
)

# Create main app with clear organization
app = typer.Typer(
    help="Critical Weight Analysis & Cybersecurity Tool",
    pretty_exceptions_show_locals=False
)
console = Console()

# Add command groups to main app
app.add_typer(basic_app, name="basic", help="Basic CWA operations")
app.add_typer(phase_app, name="security", help="A/B/C Security analysis phases")
app.add_typer(research_app, name="research", help="Super weight research & analysis")
app.add_typer(monitoring_app, name="monitor", help="Real-time monitoring & detection")
app.add_typer(analysis_app, name="advanced", help="Game theory & transfer analysis")

# Keep some core commands at top level for backwards compatibility
@app.command()
def run(
    config_path: str,
    output_dir: str = None
):
    """Run a basic CWA experiment (alias for basic run)."""
    from .commands.basic_commands import run as basic_run
    basic_run(config_path, output_dir)


@app.command("phase-a")
def phase_a(
    model_name: str,
    output_dir: str = "phase_a_results",
    metric: str = "security_gradient",
    top_k: int = 500,
    vulnerability_threshold: float = 0.8,
    device: str = "cuda"
):
    """Phase A: Critical Weight Discovery (alias for security phase-a)."""
    from .commands.phase_commands import run_phase_a
    run_phase_a(model_name, output_dir, metric, top_k, vulnerability_threshold, device)


@app.command("phase-b")
def phase_b(
    model_name: str,
    critical_weights_file: str,
    output_dir: str = "phase_b_results",
    attack_methods: str = "fgsm,pgd,bit_flip",
    device: str = "cuda"
):
    """Phase B: Attack Simulation (alias for security phase-b)."""
    from .commands.phase_commands import run_phase_b
    run_phase_b(model_name, critical_weights_file, output_dir, attack_methods, device)


@app.command("phase-c")
def phase_c(
    model_name: str,
    critical_weights_file: str,
    attack_results_file: str,
    output_dir: str = "phase_c_results",
    protection_methods: str = "weight_redundancy,checksums,adversarial_training",
    device: str = "cuda"
):
    """Phase C: Protection & Defense (alias for security phase-c)."""
    from .commands.phase_commands import run_phase_c
    run_phase_c(model_name, critical_weights_file, attack_results_file, output_dir, protection_methods, device)


@app.command("extract-critical-weights")
def extract_critical_weights(
    model_name: str,
    mode: str = "super_weight_discovery",
    sensitivity_threshold: float = 0.7,
    top_k_percent: float = 0.001,
    layer_focus: str = "early",
    output_format: str = "csv,json,plots",
    output_dir: str = "research_output",
    research_mode: bool = True,
    device: str = "cuda"
):
    """Extract critical weights (alias for research extract-critical-weights)."""
    from .commands.research_commands import extract_critical_weights as research_extract
    research_extract(
        model_name, mode, sensitivity_threshold, top_k_percent,
        layer_focus, output_format, output_dir, research_mode, device
    )


@app.command()
def info():
    """Show system information (alias for basic info)."""
    from .commands.basic_commands import info as basic_info
    basic_info()


@app.command()
def list_models():
    """List available models (alias for basic list-models)."""
    from .commands.basic_commands import list_models as basic_list_models
    basic_list_models()


if __name__ == "__main__":
    app()