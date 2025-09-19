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
from ..sensitivity.security_analyzer import SecurityWeightAnalyzer
from ..security.adversarial import AdversarialAttackSimulator
from ..security.targeted_attacks import TargetedAttackSimulator
from ..security.fault_injection import FaultInjector
from ..security.defense_mechanisms import DefenseManager
from ..security.weight_protection import CriticalWeightProtector
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


# Phase A: Critical Weight Discovery Commands

@app.command("phase-a")
def run_phase_a(
    model_name: str,
    output_dir: str = "phase_a_results",
    metric: str = "security_gradient",
    top_k: int = 500,
    vulnerability_threshold: float = 0.8,
    device: str = "cuda"
):
    """
    Phase A: Critical Weight Discovery & Vulnerability Analysis

    Identifies critical weights most vulnerable to attacks using security-aware
    sensitivity analysis.
    """
    console.print(f"[bold blue]🔍 Phase A: Critical Weight Discovery[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Metric: {metric}")
    console.print(f"Vulnerability Threshold: {vulnerability_threshold}")

    try:
        # Setup
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(Path(output_dir))

        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        model_config = {
            "name": model_name,
            "device": device,
            "torch_dtype": "float16"
        }
        model_manager = LambdaLabsLLMManager(model_config)
        model = model_manager.load_model()

        # Create data
        console.print("[yellow]Preparing calibration data...[/yellow]")
        sample_texts = create_sample_data(100)
        data_loader = create_data_loader(sample_texts, model_manager.tokenizer)

        # Run security analysis
        console.print("[yellow]Discovering critical weights...[/yellow]")
        analyzer = SecurityWeightAnalyzer(vulnerability_threshold)
        critical_analysis = analyzer.discover_critical_weights(
            model, data_loader, vulnerability_threshold, top_k
        )

        # Save results
        results_path = Path(output_dir) / "critical_weights.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                "phase": "A",
                "model": model_name,
                "critical_weights": critical_analysis.critical_weights[:50],  # Save top 50
                "vulnerability_map": critical_analysis.vulnerability_map,
                "attack_surface": critical_analysis.attack_surface,
                "metadata": critical_analysis.metadata
            }, f)

        console.print(f"[bold green]✅ Phase A Complete![/bold green]")
        console.print(f"Critical weights discovered: {len(critical_analysis.critical_weights)}")
        console.print(f"Results saved to: {results_path}")

        # Display summary
        table = Table(title="Phase A Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Critical Weights Found", str(len(critical_analysis.critical_weights)))
        table.add_row("Vulnerability Threshold", str(vulnerability_threshold))
        table.add_row("Most Vulnerable Layer", max(critical_analysis.vulnerability_map.keys(),
                     key=critical_analysis.vulnerability_map.get))
        table.add_row("Attack Vectors Identified", str(len(critical_analysis.attack_surface.get("attack_vectors", []))))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]❌ Phase A failed: {e}[/bold red]")
        raise


@app.command("phase-b")
def run_phase_b(
    model_name: str,
    critical_weights_file: str,
    output_dir: str = "phase_b_results",
    attack_methods: str = "fgsm,pgd,bit_flip",
    device: str = "cuda"
):
    """
    Phase B: Attack Simulation on Critical Weights

    Tests targeted attacks on critical weights discovered in Phase A.
    """
    console.print(f"[bold red]⚔️  Phase B: Attack Simulation[/bold red]")
    console.print(f"Model: {model_name}")
    console.print(f"Critical weights file: {critical_weights_file}")

    attack_list = attack_methods.split(",")
    console.print(f"Attack methods: {attack_list}")

    try:
        # Setup
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(Path(output_dir))

        # Load critical weights
        with open(critical_weights_file, 'r') as f:
            phase_a_results = yaml.unsafe_load(f)
        critical_weights = phase_a_results["critical_weights"]

        console.print(f"Loaded {len(critical_weights)} critical weights")

        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        model_config = {
            "name": model_name,
            "device": device,
            "torch_dtype": "float16"
        }
        model_manager = LambdaLabsLLMManager(model_config)
        model = model_manager.load_model()

        # Create test data
        console.print("[yellow]Preparing test data...[/yellow]")
        sample_texts = create_sample_data(50)

        # Run targeted attacks
        console.print("[yellow]Simulating attacks on critical weights...[/yellow]")
        attack_simulator = TargetedAttackSimulator(model, model_manager.tokenizer)
        attack_results = attack_simulator.simulate_attacks_on_critical_weights(
            model, critical_weights, attack_list, sample_texts
        )

        # Test fault injection
        if "bit_flip" in attack_list or "fault_injection" in attack_list:
            console.print("[yellow]Running fault injection tests...[/yellow]")
            fault_injector = FaultInjector()
            fault_results = fault_injector.inject_faults_on_critical_weights(
                model, critical_weights
            )
            attack_results["fault_injection"] = {
                "injected_faults": len(fault_results.injected_faults),
                "performance_degradation": fault_results.performance_degradation,
                "critical_failures": fault_results.critical_failures
            }

        # Save results
        results_path = Path(output_dir) / "attack_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                "phase": "B",
                "model": model_name,
                "attack_methods": attack_list,
                "critical_weights_tested": len(critical_weights),
                "attack_results": attack_results
            }, f)

        console.print(f"[bold green]✅ Phase B Complete![/bold green]")
        console.print(f"Results saved to: {results_path}")

        # Display summary
        table = Table(title="Phase B Attack Results")
        table.add_column("Attack Method", style="cyan")
        table.add_column("Success Rate", style="green")

        for method, result in attack_results.get("attack_results", {}).items():
            if isinstance(result, dict):
                success_rate = result.get("success_rate", 0.0)
                table.add_row(method, f"{success_rate:.3f}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]❌ Phase B failed: {e}[/bold red]")
        raise


@app.command("phase-c")
def run_phase_c(
    model_name: str,
    critical_weights_file: str,
    attack_results_file: str,
    output_dir: str = "phase_c_results",
    protection_methods: str = "weight_redundancy,checksums,adversarial_training",
    device: str = "cuda"
):
    """
    Phase C: Protection & Defense Implementation

    Implements defense mechanisms to protect critical weights from attacks.
    """
    console.print(f"[bold green]🛡️  Phase C: Protection & Defense[/bold green]")
    console.print(f"Model: {model_name}")
    console.print(f"Critical weights file: {critical_weights_file}")

    protection_list = protection_methods.split(",")
    console.print(f"Protection methods: {protection_list}")

    try:
        # Setup
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(Path(output_dir))

        # Load previous results
        with open(critical_weights_file, 'r') as f:
            phase_a_results = yaml.unsafe_load(f)
        critical_weights = phase_a_results["critical_weights"]

        with open(attack_results_file, 'r') as f:
            phase_b_results = yaml.safe_load(f)
        successful_attacks = [
            method for method, result in phase_b_results.get("attack_results", {}).items()
            if isinstance(result, dict) and result.get("success_rate", 0) > 0.3
        ]

        console.print(f"Protecting {len(critical_weights)} critical weights")
        console.print(f"Defending against {len(successful_attacks)} successful attacks")

        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        model_config = {
            "name": model_name,
            "device": device,
            "torch_dtype": "float16"
        }
        model_manager = LambdaLabsLLMManager(model_config)
        model = model_manager.load_model()

        # Implement protections
        console.print("[yellow]Implementing protection mechanisms...[/yellow]")
        defense_manager = DefenseManager(model)
        protection_results = defense_manager.implement_protection_mechanisms(
            critical_weights, protection_list
        )

        # Test protected model
        console.print("[yellow]Testing protected model...[/yellow]")

        # If no successful attacks, create synthetic test suite
        if len(successful_attacks) == 0:
            console.print("[yellow]No successful attacks found, using synthetic attack suite for testing...[/yellow]")
            synthetic_attacks = ["fgsm", "pgd", "bit_flip", "random_noise"]
            test_results = defense_manager.test_protected_model(model, synthetic_attacks)
        else:
            test_results = defense_manager.test_protected_model(model, successful_attacks)

        # Calculate overall security score based on protection coverage
        protection_coverage = protection_results.get("protection_coverage", 0.0)
        residual_risk = protection_results.get("residual_vulnerability", {}).get("residual_risk_score", 1.0)

        # Security score: combination of coverage and risk reduction
        security_score = min(1.0, protection_coverage * (1.0 - max(0.0, residual_risk)))
        test_results["overall_security_score"] = security_score

        # Save results
        results_path = Path(output_dir) / "protection_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump({
                "phase": "C",
                "model": model_name,
                "protection_methods": protection_list,
                "protection_results": protection_results,
                "test_results": test_results,
                "protected_weights": len(critical_weights)
            }, f)

        console.print(f"[bold green]✅ Phase C Complete![/bold green]")
        console.print(f"Results saved to: {results_path}")

        # Display summary
        table = Table(title="Phase C Protection Results")
        table.add_column("Protection Method", style="cyan")
        table.add_column("Effectiveness", style="green")

        for method, effectiveness in protection_results.get("defense_effectiveness", {}).items():
            if isinstance(effectiveness, (int, float)):
                table.add_row(method, f"{effectiveness:.3f}")

        overall_score = test_results.get("overall_security_score", 0.0)
        table.add_row("Overall Security Score", f"{overall_score:.3f}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]❌ Phase C failed: {e}[/bold red]")
        raise


@app.command("run-complete-pipeline")
def run_complete_pipeline(
    model_name: str,
    output_dir: str = "complete_pipeline_results",
    vulnerability_threshold: float = 0.8,
    top_k: int = 500,
    attack_methods: str = "fgsm,pgd,bit_flip,fault_injection",
    protection_methods: str = "weight_redundancy,checksums,adversarial_training,input_sanitization",
    device: str = "cuda"
):
    """
    Run Complete A→B→C Security Research Pipeline

    Executes the full workflow:
    1. Phase A: Discover critical weights
    2. Phase B: Attack critical weights
    3. Phase C: Protect and re-test
    """
    console.print(f"[bold magenta]🚀 Complete A→B→C Security Pipeline[/bold magenta]")
    console.print(f"Model: {model_name}")
    console.print(f"Output: {output_dir}")

    try:
        # Setup main output directory
        main_output = Path(output_dir)
        main_output.mkdir(parents=True, exist_ok=True)
        setup_logging(main_output)

        # Phase A: Critical Weight Discovery
        console.print(f"\n[bold blue]Phase A: Critical Weight Discovery[/bold blue]")
        phase_a_dir = main_output / "phase_a"
        phase_a_dir.mkdir(exist_ok=True)

        # Load model once
        console.print("[yellow]Loading model...[/yellow]")
        model_config = {
            "name": model_name,
            "device": device,
            "torch_dtype": "float16"
        }
        model_manager = LambdaLabsLLMManager(model_config)
        model = model_manager.load_model()

        # Create data once
        sample_texts = create_sample_data(100)
        data_loader = create_data_loader(sample_texts, model_manager.tokenizer)

        # Phase A execution
        analyzer = SecurityWeightAnalyzer(vulnerability_threshold)
        critical_analysis = analyzer.discover_critical_weights(
            model, data_loader, vulnerability_threshold, top_k
        )

        phase_a_results = {
            "phase": "A",
            "model": model_name,
            "critical_weights": critical_analysis.critical_weights,
            "vulnerability_map": critical_analysis.vulnerability_map,
            "attack_surface": critical_analysis.attack_surface,
            "metadata": critical_analysis.metadata
        }

        critical_weights_file = phase_a_dir / "critical_weights.yaml"
        with open(critical_weights_file, 'w') as f:
            yaml.dump(phase_a_results, f)

        console.print(f"✅ Phase A: {len(critical_analysis.critical_weights)} critical weights discovered")

        # Phase B: Attack Simulation
        console.print(f"\n[bold red]Phase B: Attack Simulation[/bold red]")
        phase_b_dir = main_output / "phase_b"
        phase_b_dir.mkdir(exist_ok=True)

        attack_list = attack_methods.split(",")
        attack_simulator = TargetedAttackSimulator(model, model_manager.tokenizer)
        attack_results = attack_simulator.simulate_attacks_on_critical_weights(
            model, critical_analysis.critical_weights, attack_list, sample_texts
        )

        # Add fault injection
        if "fault_injection" in attack_list:
            fault_injector = FaultInjector()
            fault_results = fault_injector.inject_faults_on_critical_weights(
                model, critical_analysis.critical_weights
            )
            attack_results["fault_injection"] = {
                "injected_faults": len(fault_results.injected_faults),
                "performance_degradation": fault_results.performance_degradation
            }

        phase_b_results = {
            "phase": "B",
            "model": model_name,
            "attack_methods": attack_list,
            "attack_results": attack_results
        }

        attack_results_file = phase_b_dir / "attack_results.yaml"
        with open(attack_results_file, 'w') as f:
            yaml.dump(phase_b_results, f)

        successful_attacks = [
            method for method, result in attack_results.get("attack_results", {}).items()
            if isinstance(result, dict) and result.get("success_rate", 0) > 0.3
        ]

        console.print(f"✅ Phase B: {len(successful_attacks)} successful attacks identified")

        # Phase C: Protection & Defense
        console.print(f"\n[bold green]Phase C: Protection & Defense[/bold green]")
        phase_c_dir = main_output / "phase_c"
        phase_c_dir.mkdir(exist_ok=True)

        protection_list = protection_methods.split(",")
        defense_manager = DefenseManager(model)
        protection_results = defense_manager.implement_protection_mechanisms(
            critical_analysis.critical_weights, protection_list
        )

        # Test protected model
        protector = CriticalWeightProtector(model)
        test_results = protector.test_protected_model(model, successful_attacks)

        phase_c_results = {
            "phase": "C",
            "model": model_name,
            "protection_methods": protection_list,
            "protection_results": protection_results,
            "test_results": test_results
        }

        protection_results_file = phase_c_dir / "protection_results.yaml"
        with open(protection_results_file, 'w') as f:
            yaml.dump(phase_c_results, f)

        console.print(f"✅ Phase C: {protection_results.get('protection_coverage', 0):.3f} protection coverage")

        # Final Summary Report
        summary_report = {
            "pipeline": "A→B→C Complete",
            "model": model_name,
            "timestamp": str(Path(output_dir).stat().st_mtime),
            "phase_a_summary": {
                "critical_weights_found": len(critical_analysis.critical_weights),
                "vulnerability_threshold": vulnerability_threshold,
                "most_vulnerable_layer": max(critical_analysis.vulnerability_map.keys(),
                                           key=critical_analysis.vulnerability_map.get)
            },
            "phase_b_summary": {
                "attack_methods_tested": len(attack_list),
                "successful_attacks": len(successful_attacks),
                "max_success_rate": max([
                    result.get("success_rate", 0) for result in attack_results.get("attack_results", {}).values()
                    if isinstance(result, dict)
                ], default=0)
            },
            "phase_c_summary": {
                "protection_methods_applied": len(protection_list),
                "protection_coverage": protection_results.get("protection_coverage", 0),
                "overall_security_score": test_results.get("overall_security_score", 0),
                "performance_overhead": protection_results.get("performance_overhead", 0)
            }
        }

        summary_file = main_output / "pipeline_summary.yaml"
        with open(summary_file, 'w') as f:
            yaml.dump(summary_report, f)

        # Display final results
        console.print(f"\n[bold magenta]🎉 Complete Pipeline Results[/bold magenta]")

        table = Table(title="A→B→C Pipeline Summary")
        table.add_column("Phase", style="cyan")
        table.add_column("Key Metric", style="green")
        table.add_column("Value", style="yellow")

        table.add_row("Phase A", "Critical Weights", str(len(critical_analysis.critical_weights)))
        table.add_row("Phase B", "Successful Attacks", str(len(successful_attacks)))
        table.add_row("Phase C", "Protection Coverage", f"{protection_results.get('protection_coverage', 0):.3f}")
        table.add_row("Overall", "Security Score", f"{test_results.get('overall_security_score', 0):.3f}")

        console.print(table)
        console.print(f"\n[bold green]✅ Complete pipeline results saved to: {main_output}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]❌ Complete pipeline failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    app()