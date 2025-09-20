"""Real-time monitoring and security commands."""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional
import pandas as pd

from ...core.models import LambdaLabsLLMManager
from ...core.data import create_sample_data

monitoring_app = typer.Typer(help="Real-time monitoring and security commands")
console = Console()


@monitoring_app.command("monitor-realtime")
def monitor_realtime(
    model_name: str,
    detection_algorithms: str = "statistical,gradient,activation,weight_drift",
    circuit_breaker_config: str = "auto",
    latency_target: float = 1.0,
    anomaly_thresholds: str = "adaptive",
    output_dir: Optional[str] = None,
    device: str = "auto"
):
    """Real-time security monitoring with circuit breakers (Phase 2)."""
    try:
        from ...monitoring.realtime_monitor import RealtimeSecurityMonitor, MonitoringConfig
        from ...monitoring.circuit_breaker import CircuitBreakerConfig
        from ...monitoring.anomaly_detector import DetectionConfig

        console.print(f"[bold green]🕰️ Starting real-time monitoring for {model_name}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"realtime_monitoring_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model()
        tokenizer = model_manager.tokenizer

        # Configure monitoring
        algorithms = [alg.strip() for alg in detection_algorithms.split(",")]

        # Setup monitoring configuration
        monitoring_config = MonitoringConfig()
        monitoring_config.enable_circuit_breaker = True
        monitoring_config.target_latency_ms = latency_target

        # Initialize monitor
        monitor = RealtimeSecurityMonitor(config=monitoring_config)

        # Create sample inputs for monitoring
        sample_inputs = create_sample_data(tokenizer, num_samples=100)

        console.print(f"[blue]Detection algorithms: {', '.join(algorithms)}[/blue]")
        console.print(f"[blue]Target latency: {latency_target}ms[/blue]")
        console.print(f"[blue]Circuit breaker: {circuit_breaker_config}[/blue]")

        # Run monitoring session
        monitoring_results = []

        for i, input_batch in enumerate(sample_inputs):
            if i >= 50:  # Limit for demo
                break

            # Monitor inference
            response = monitor.monitor_inference(model, input_batch)
            monitoring_results.append(response)

            if response.alerts:
                console.print(f"[red]⚠️  Alert detected at batch {i}: {len(response.alerts)} anomalies[/red]")

        console.print(f"[bold green]✅ Monitoring session complete![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Real-time monitoring failed: {e}[/red]")
        raise typer.Exit(1)