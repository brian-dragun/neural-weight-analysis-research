"""Command line interface for Critical Weight Analysis tool."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import yaml
import logging
from typing import Optional
import torch

from ..core.config import ExperimentConfig
from ..core.models import LambdaLabsLLMManager
from ..core.data import create_sample_data, create_data_loader
from ..sensitivity.registry import get_sensitivity_metric, list_sensitivity_metrics
from ..utils.logging import setup_logging

app = typer.Typer(help="Critical Weight Analysis & Cybersecurity Tool - Foundation")
console = Console()


@app.command()
def run(
    config_path: str,
    output_dir: Optional[str] = None
):
    """Run a basic CWA experiment."""
    try:
        # Load configuration
        config = ExperimentConfig.from_yaml(Path(config_path))
        if output_dir:
            config.output_dir = output_dir

        console.print(f"[bold green]Starting experiment: {config.name}[/bold green]")

        # Setup logging
        setup_logging(Path(config.output_dir))

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load model with Lambda Labs optimization
        console.print("[yellow]Loading Hugging Face LLM on Lambda Labs GPU...[/yellow]")
        model_manager = LambdaLabsLLMManager(config.model.dict())
        model = model_manager.load_model()

        # Display Lambda Labs specific memory usage
        memory_info = model_manager.estimate_memory_usage()
        console.print(f"[blue]Lambda Labs GPU Memory Usage: {memory_info.get('model_size_gb', 0):.2f} GB[/blue]")

        # Create sample data
        console.print("[yellow]Creating sample data...[/yellow]")
        sample_texts = create_sample_data(config.data_samples)
        data_loader = create_data_loader(sample_texts, model_manager.tokenizer)

        # Run sensitivity analysis
        console.print("[yellow]Running sensitivity analysis...[/yellow]")
        sensitivity_func = get_sensitivity_metric(config.sensitivity.metric)
        results = sensitivity_func(
            model=model,
            data_loader=data_loader,
            top_k=config.sensitivity.top_k
        )

        # Save results
        results_path = Path(config.output_dir) / "results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                "experiment": config.name,
                "model_info": model_manager.get_model_info(),
                "sensitivity_results": {
                    "metric": results.metric_name,
                    "top_weights": results.top_k_weights[:10],  # Save top 10
                    "metadata": results.metadata
                }
            }, f)

        console.print(f"[bold green]✅ Experiment completed! Results saved to {results_path}[/bold green]")

        # Display summary
        _display_results_summary(results, model_manager.get_model_info())

    except Exception as e:
        console.print(f"[bold red]❌ Experiment failed: {e}[/bold red]")
        raise


@app.command()
def create_config(
    name: str,
    model: str = "microsoft/DialoGPT-small",
    model_size: str = "small",
    metric: str = "basic_gradient",
    device: str = "cuda",  # Lambda Labs default
    quantization: Optional[str] = None,
    output_path: str = "config.yaml"
):
    """Create a new experiment configuration optimized for Lambda Labs GPU VMs."""
    config = ExperimentConfig(
        name=name,
        model={
            "name": model,
            "model_size": model_size,
            "device": device,
            "torch_dtype": "float16",  # Lambda Labs default
            "quantization": quantization,
            "device_map": "auto",
            "cache_dir": "/tmp/hf_cache"
        },
        sensitivity={"metric": metric}
    )

    with open(output_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)

    console.print(f"[green]✅ Lambda Labs configuration saved to {output_path}[/green]")
    console.print(f"[blue]Model: {model} (Size: {model_size}, Device: {device}, FP16 optimized)[/blue]")


@app.command()
def list_models():
    """List recommended Hugging Face LLM models by size category."""

    models_by_size = {
        "small": [
            ("microsoft/DialoGPT-small", "124M parameters - Good for testing"),
            ("gpt2", "124M parameters - Classic GPT-2"),
            ("distilgpt2", "82M parameters - Distilled GPT-2"),
            ("microsoft/phi-2", "2.7B parameters - Microsoft Phi-2")
        ],
        "medium": [
            ("gpt2-medium", "355M parameters - GPT-2 Medium"),
            ("microsoft/phi-3-mini-4k-instruct", "3.8B parameters - Phi-3 Mini"),
            ("mistralai/Mistral-7B-v0.1", "7B parameters - Mistral 7B"),
            ("meta-llama/Llama-2-7b-hf", "7B parameters - LLaMA 2 7B")
        ],
        "large": [
            ("meta-llama/Llama-2-13b-hf", "13B parameters - LLaMA 2 13B"),
            ("mistralai/Mixtral-8x7B-v0.1", "46.7B parameters - Mixtral 8x7B"),
            ("meta-llama/Llama-2-70b-hf", "70B parameters - LLaMA 2 70B"),
            ("microsoft/phi-3-medium-4k-instruct", "14B parameters - Phi-3 Medium")
        ]
    }

    for size, models in models_by_size.items():
        table = Table(title=f"{size.title()} Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Description", style="green")

        for model_name, description in models:
            table.add_row(model_name, description)

        console.print(table)
        console.print()  # Add spacing


@app.command()
def model_info(model_name: str):
    """Get detailed information about a specific Hugging Face model for Lambda Labs."""
    try:
        console.print(f"[yellow]Loading model info for Lambda Labs: {model_name}[/yellow]")

        # Create a minimal config to get model info
        config = {"name": model_name, "model_size": "unknown", "device": "cuda"}
        manager = LambdaLabsLLMManager(config)

        # Load just the tokenizer and config (lightweight)
        from transformers import AutoConfig, AutoTokenizer

        model_config = AutoConfig.from_pretrained(model_name, cache_dir="/tmp/hf_cache")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/hf_cache")

        # Estimate parameters from config
        vocab_size = getattr(model_config, 'vocab_size', 'unknown')
        hidden_size = getattr(model_config, 'hidden_size', 'unknown')
        num_layers = getattr(model_config, 'num_hidden_layers', 'unknown')

        table = Table(title=f"Lambda Labs Model Information: {model_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Architecture", getattr(model_config, 'model_type', 'unknown'))
        table.add_row("Vocabulary Size", str(vocab_size))
        table.add_row("Hidden Size", str(hidden_size))
        table.add_row("Number of Layers", str(num_layers))
        table.add_row("Max Position", str(getattr(model_config, 'max_position_embeddings', 'unknown')))
        table.add_row("Lambda GPU Recommended", "✓ CUDA FP16" if torch.cuda.is_available() else "⚠️  CPU Fallback")

        console.print(table)

    except Exception as e:
        console.print(f"[red]❌ Failed to get Lambda Labs model info: {e}[/red]")


@app.command()
def list_metrics():
    """List available sensitivity metrics."""
    metrics = list_sensitivity_metrics()

    table = Table(title="Available Sensitivity Metrics")
    table.add_column("Metric Name", style="cyan")
    table.add_column("Description", style="green")

    for metric in metrics:
        # Add basic descriptions
        descriptions = {
            "basic_gradient": "Basic gradient-based sensitivity analysis"
        }
        table.add_row(metric, descriptions.get(metric, "No description available"))

    console.print(table)


@app.command()
def info():
    """Show Lambda Labs system information."""
    import torch

    table = Table(title="Lambda Labs System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status/Version", style="green")

    table.add_row("Python", "3.12+")
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", "✓ Ready" if torch.cuda.is_available() else "❌ Not Available")

    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda)
        table.add_row("GPU Count", str(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            table.add_row(f"Lambda GPU {i}", f"{gpu_name} ({gpu_memory:.1f}GB)")

    table.add_row("Available Metrics", str(len(list_sensitivity_metrics())))
    table.add_row("Cache Directory", "/tmp/hf_cache")
    table.add_row("Optimized For", "Lambda Labs GPU VMs")

    console.print(table)


def _display_results_summary(results, model_info):
    """Display results summary."""
    table = Table(title="Experiment Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", model_info.get("model_name", "Unknown"))
    table.add_row("Total Parameters", f"{model_info.get('total_parameters', 0):,}")
    table.add_row("Sensitivity Metric", results.metric_name)
    table.add_row("Top Weights Found", str(len(results.top_k_weights)))

    if results.top_k_weights:
        table.add_row("Highest Score", f"{results.top_k_weights[0][2]:.6f}")
        table.add_row("Most Critical Layer", results.top_k_weights[0][0])

    console.print(table)


if __name__ == "__main__":
    app()