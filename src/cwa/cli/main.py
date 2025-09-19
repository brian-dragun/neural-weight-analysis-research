"""Command line interface for Critical Weight Analysis tool."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, Any
import torch
import pandas as pd

from ..core.config import ExperimentConfig
from ..core.models import LambdaLabsLLMManager
from ..core.data import create_sample_data, create_data_loader
from ..sensitivity.registry import get_sensitivity_metric, list_sensitivity_metrics
from ..sensitivity.security_analyzer import SecurityWeightAnalyzer
from ..security.adversarial import AdversarialAttackSimulator
from ..security.targeted_attacks import TargetedAttackSimulator
from ..security.fault_injection import FaultInjector
from ..security.defense_mechanisms import DefenseManager
from ..research.super_weight_analyzer import SuperWeightAnalyzer
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
    """
    🔬 Research-focused critical weight extraction for PhD-level super weight investigation.

    Implements methodologies for discovering "super weights" in transformer models based on:
    - Activation magnitude monitoring (>1e3 threshold)
    - Hessian-based sensitivity scoring with FKeras integration
    - Focus on early layers (2-4) and mlp.down_proj components
    - Extract top 0.001% of weights by sensitivity

    Args:
        model_name: Model to analyze (e.g., "gpt2", "microsoft/DialoGPT-small")
        mode: Analysis mode ("super_weight_discovery", "validation", "comprehensive")
        sensitivity_threshold: Minimum sensitivity score for weight inclusion
        top_k_percent: Percentage of top weights to extract (0.001 = 0.001%)
        layer_focus: Layer range focus ("early", "middle", "late", "all")
        output_format: Export formats ("csv", "json", "plots" or combinations)
        output_dir: Directory for research outputs
        research_mode: Enable PhD research-specific analysis
        device: Device to use ("cuda" or "cpu")
    """
    console = Console()

    try:
        console.print(f"[bold blue]🔬 Research Mode: Critical Weight Extraction[/bold blue]")
        console.print(f"Model: {model_name}")
        console.print(f"Mode: {mode}")
        console.print(f"Sensitivity Threshold: {sensitivity_threshold}")
        console.print(f"Top-K Percentage: {top_k_percent}%")
        console.print(f"Layer Focus: {layer_focus}")

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        setup_logging(output_path)

        # Load model
        console.print("[yellow]Loading model for research analysis...[/yellow]")
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model(model_name)

        # Initialize SuperWeightAnalyzer
        analyzer = SuperWeightAnalyzer(model, model_name)

        # Extract critical weights
        console.print(f"[yellow]Extracting critical weights in {mode} mode...[/yellow]")
        research_data = analyzer.extract_critical_weights(
            mode=mode,
            sensitivity_threshold=sensitivity_threshold,
            top_k_percent=top_k_percent,
            layer_focus=layer_focus,
            output_dir=output_dir
        )

        # Display results
        console.print(f"[bold green]✅ Critical Weight Extraction Complete![/bold green]")
        console.print(f"Results saved to: {output_path}")

        # Display summary table
        table = Table(title="Research Extraction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        stats = research_data["statistics"]
        table.add_row("Total Weights Analyzed", f"{stats['total_weights_analyzed']:,}")
        table.add_row("Critical Weights Found", f"{stats['critical_weights_found']}")
        table.add_row("Discovery Rate", f"{stats['discovery_rate']:.6f}%")
        table.add_row("Average Sensitivity", f"{stats['avg_sensitivity']:.6f}")
        table.add_row("Layer Coverage", f"{stats['layer_coverage']} layers")
        table.add_row("Component Diversity", f"{stats['component_diversity']}")

        console.print(table)

        # Research findings
        analysis = research_data["analysis_results"]
        console.print(f"\n[bold yellow]🔍 Key Research Findings:[/bold yellow]")
        console.print(f"• Early Layer Concentration: {analysis['architectural']['early_layer_concentration']:.1%}")
        console.print(f"• Avg Perplexity Impact: {analysis['behavioral']['avg_perplexity_impact']:.1f}x")
        console.print(f"• Max Perplexity Impact: {analysis['behavioral']['max_perplexity_impact']:.1f}x")
        console.print(f"• Significant Weights: {analysis['behavioral']['significant_weights']}/{analysis['behavioral']['tested_weights']}")

    except Exception as e:
        console.print(f"[bold red]❌ Critical weight extraction failed: {e}[/bold red]")
        raise


@app.command("validate-super-weights")
def validate_super_weights(
    model_name: str,
    coordinates: str,
    perplexity_threshold: float = 100,
    output_dir: str = "validation_output",
    export_results: bool = True,
    device: str = "cuda"
):
    """
    🧪 Validate specific super weight coordinates using 100× perplexity methodology.

    Validates known super weight coordinates like:
    - [(2, 'mlp.down_proj', [3968, 7003])] for Llama-7B
    - [(1, 'mlp.down_proj', [2070, 7310])] for Mistral-7B

    Uses bit-level analysis and perplexity impact measurement to confirm super weight status.

    Args:
        model_name: Model to validate against
        coordinates: Weight coordinates as string, e.g., "[(2, 'mlp.down_proj', [3968, 7003])]"
        perplexity_threshold: Minimum perplexity increase to confirm super weight (default: 100x)
        output_dir: Directory for validation outputs
        export_results: Export detailed validation results
        device: Device to use ("cuda" or "cpu")
    """
    console = Console()

    try:
        console.print(f"[bold blue]🧪 Super Weight Validation[/bold blue]")
        console.print(f"Model: {model_name}")
        console.print(f"Perplexity Threshold: {perplexity_threshold}x")

        # Parse coordinates
        import ast
        try:
            coord_list = ast.literal_eval(coordinates)
            if not isinstance(coord_list, list):
                coord_list = [coord_list]
        except Exception as e:
            raise ValueError(f"Invalid coordinates format: {e}. Expected: [(layer, 'component', [row, col])]")

        console.print(f"Validating {len(coord_list)} coordinate(s)")

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        setup_logging(output_path)

        # Load model
        console.print("[yellow]Loading model for validation...[/yellow]")
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model(model_name)

        # Initialize SuperWeightAnalyzer
        analyzer = SuperWeightAnalyzer(model, model_name)

        # Validate coordinates
        console.print("[yellow]Validating super weight coordinates...[/yellow]")
        validation_report = analyzer.validate_super_weights(
            coordinates=coord_list,
            perplexity_threshold=perplexity_threshold,
            output_dir=output_dir
        )

        # Display results
        console.print(f"[bold green]✅ Super Weight Validation Complete![/bold green]")
        console.print(f"Results saved to: {output_path}")

        # Display validation table
        table = Table(title="Super Weight Validation Results")
        table.add_column("Coordinate", style="cyan")
        table.add_column("Perplexity Impact", style="yellow")
        table.add_column("Super Weight?", style="green")
        table.add_column("Baseline", style="blue")

        for coord_key, result in validation_report["results"].items():
            coord = result["coordinates"]
            coord_str = f"L{coord[0]}.{coord[1]}[{coord[2][0]},{coord[2][1]}]"

            impact = result["perplexity_impact"]["multiplier"]
            is_super = "✅ YES" if result["is_super_weight"] else "❌ NO"
            baseline = f"{result['perplexity_impact']['baseline_perplexity']:.2f}"

            table.add_row(coord_str, f"{impact:.1f}x", is_super, baseline)

        console.print(table)

        # Summary
        summary = validation_report["summary"]
        console.print(f"\n[bold yellow]📊 Validation Summary:[/bold yellow]")
        console.print(f"• Coordinates Tested: {summary['total_tested']}")
        console.print(f"• Confirmed Super Weights: {summary['confirmed_super_weights']}")
        console.print(f"• Confirmation Rate: {summary['confirmation_rate']:.1%}")
        console.print(f"• Average Impact: {summary['avg_perplexity_impact']:.1f}x")
        console.print(f"• Maximum Impact: {summary['max_perplexity_impact']:.1f}x")

    except Exception as e:
        console.print(f"[bold red]❌ Super weight validation failed: {e}[/bold red]")
        raise


@app.command("research-extract")
def research_extract(
    model_name: str,
    focus: str = "attention-mechanisms",
    threshold: float = 0.7,
    export_format: str = "research-csv",
    include_metadata: bool = True,
    analysis_types: str = "statistical,behavioral,architectural",
    output_dir: str = "research_extract_output",
    device: str = "cuda"
):
    """
    🎓 Specialized research extraction for PhD-level analysis.

    Focused extraction modes for specific research areas:
    - attention-mechanisms: Focus on attention layers and components
    - mlp-components: Focus on MLP layers (down_proj, up_proj, gate_proj)
    - early-layers: Focus on layers 0-3 for super weight discovery
    - layer-norms: Focus on normalization components
    - comprehensive: Full model analysis

    Args:
        model_name: Model to analyze
        focus: Research focus area
        threshold: Sensitivity threshold for inclusion
        export_format: Output format ("research-csv", "publication-ready", "replication-data")
        include_metadata: Include detailed metadata for replication
        analysis_types: Types of analysis to perform (comma-separated)
        output_dir: Output directory for research data
        device: Device to use
    """
    console = Console()

    try:
        console.print(f"[bold blue]🎓 PhD Research Extraction Mode[/bold blue]")
        console.print(f"Model: {model_name}")
        console.print(f"Research Focus: {focus}")
        console.print(f"Analysis Types: {analysis_types}")

        # Map focus to extraction parameters
        focus_mapping = {
            "attention-mechanisms": {
                "mode": "comprehensive",
                "layer_focus": "all",
                "component_filter": "attn"
            },
            "mlp-components": {
                "mode": "super_weight_discovery",
                "layer_focus": "early",
                "component_filter": "mlp"
            },
            "early-layers": {
                "mode": "super_weight_discovery",
                "layer_focus": "early",
                "component_filter": "all"
            },
            "layer-norms": {
                "mode": "comprehensive",
                "layer_focus": "all",
                "component_filter": "norm"
            },
            "comprehensive": {
                "mode": "comprehensive",
                "layer_focus": "all",
                "component_filter": "all"
            }
        }

        if focus not in focus_mapping:
            raise ValueError(f"Unknown focus: {focus}. Available: {list(focus_mapping.keys())}")

        params = focus_mapping[focus]

        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        setup_logging(output_path)

        # Load model
        console.print("[yellow]Loading model for research extraction...[/yellow]")
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model(model_name)

        # Initialize analyzer
        analyzer = SuperWeightAnalyzer(model, model_name)

        # Perform extraction
        console.print(f"[yellow]Performing {focus} analysis...[/yellow]")
        research_data = analyzer.extract_critical_weights(
            mode=params["mode"],
            sensitivity_threshold=threshold,
            top_k_percent=0.001,  # PhD research precision
            layer_focus=params["layer_focus"],
            output_dir=output_dir
        )

        # Generate specialized research outputs
        console.print("[yellow]Generating research-specific outputs...[/yellow]")

        if export_format == "publication-ready":
            _generate_publication_tables(research_data, output_path)
        elif export_format == "replication-data":
            _generate_replication_package(research_data, output_path, model_name)

        console.print(f"[bold green]✅ Research Extraction Complete![/bold green]")
        console.print(f"Research data saved to: {output_path}")

        # Display focused results
        table = Table(title=f"Research Results: {focus.title()}")
        table.add_column("Research Metric", style="cyan")
        table.add_column("Value", style="green")

        stats = research_data["statistics"]
        analysis = research_data["analysis_results"]

        table.add_row("Focus Area", focus)
        table.add_row("Critical Weights", f"{stats['critical_weights_found']}")
        table.add_row("Discovery Rate", f"{stats['discovery_rate']:.6f}%")

        if focus == "attention-mechanisms":
            attn_weights = sum(1 for w in research_data["discovered_weights"] if "attn" in w["component"])
            table.add_row("Attention Weights", f"{attn_weights}")
        elif focus == "mlp-components":
            mlp_weights = sum(1 for w in research_data["discovered_weights"] if "mlp" in w["component"])
            table.add_row("MLP Weights", f"{mlp_weights}")

        table.add_row("Early Layer %", f"{analysis['architectural']['early_layer_concentration']:.1%}")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]❌ Research extraction failed: {e}[/bold red]")
        raise


@app.command("spectral-analysis")
def spectral_analysis(
    model_name: str,
    target_layers: Optional[str] = typer.Option(None, help="Comma-separated layer names to analyze (None for all)"),
    analysis_types: str = typer.Option("signatures,transitions,stability,correlations", help="Types of spectral analysis"),
    top_k: int = typer.Option(10, help="Number of top critical configurations to identify"),
    output_dir: str = typer.Option("spectral_analysis_results", help="Output directory"),
    device: str = typer.Option("auto", help="Device to use (cuda/cpu/auto)"),
    include_pac_bounds: bool = typer.Option(True, help="Include PAC-Bayesian bounds"),
    confidence_level: float = typer.Option(0.95, help="Confidence level for PAC bounds")
):
    """
    Advanced spectral vulnerability analysis using eigenvalue-based methods.

    Analyzes neural network weight matrices using spectral learning to identify
    critical configurations, phase transitions, and vulnerability patterns that
    traditional gradient-based methods might miss.

    Features:
    - Spectral signature analysis
    - Phase transition detection
    - Stability assessment under perturbations
    - PAC-Bayesian theoretical guarantees
    - Cross-layer spectral correlations
    """
    console.print("[bold blue]🔬 Spectral Vulnerability Analysis[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Analysis Types: {analysis_types}")
    console.print(f"Device: {device}")

    try:
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"Output directory: {output_path}")

        # Setup logging
        setup_logging(output_path)

        # Load model
        console.print("[yellow]Loading model...[/yellow]")
        manager = LambdaLabsLLMManager({"model_name": model_name, "device": device})
        model = manager.load_model()

        # Import the spectral analyzer
        from ..sensitivity.spectral_analyzer import SpectralVulnerabilityAnalyzer

        # Initialize analyzer
        console.print("[yellow]Initializing spectral analyzer...[/yellow]")
        analyzer = SpectralVulnerabilityAnalyzer(model, device=device)

        # Parse target layers
        target_layer_list = None
        if target_layers:
            target_layer_list = [layer.strip() for layer in target_layers.split(",")]

        # Parse analysis types
        analysis_type_list = [t.strip() for t in analysis_types.split(",")]

        # Run comprehensive spectral analysis
        console.print("[yellow]Running spectral vulnerability analysis...[/yellow]")
        results = analyzer.analyze_spectral_vulnerabilities(
            target_layers=target_layer_list,
            analysis_types=analysis_type_list
        )

        # Detect critical configurations
        console.print("[yellow]Identifying critical eigenvalue configurations...[/yellow]")
        critical_configs = analyzer.detect_critical_eigenvalue_configurations(
            target_layers=target_layer_list,
            top_k=top_k
        )

        # Compute PAC-Bayesian bounds if requested
        pac_bounds = None
        if include_pac_bounds and "spectral_signatures" in results:
            console.print("[yellow]Computing PAC-Bayesian bounds...[/yellow]")
            signatures = list(results["spectral_signatures"].values())
            pac_bounds = analyzer.compute_pac_bayesian_bounds(
                signatures, confidence=confidence_level
            )
            results["pac_bounds"] = pac_bounds

        # Add critical configurations to results
        results["critical_configurations"] = critical_configs

        # Export results
        console.print("[yellow]Exporting analysis results...[/yellow]")
        export_path = analyzer.export_spectral_analysis(
            results, output_path, include_visualizations=True
        )

        console.print(f"[bold green]✅ Spectral Analysis Complete![/bold green]")
        console.print(f"Results saved to: {export_path}")

        # Display summary table
        table = Table(title="Spectral Vulnerability Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Details", style="yellow")

        summary = results.get("vulnerability_summary", {})

        table.add_row(
            "Parameters Analyzed",
            f"{summary.get('total_parameters_analyzed', 0)}",
            "Total neural network parameters"
        )
        table.add_row(
            "High Vulnerability",
            f"{summary.get('high_vulnerability_count', 0)}",
            "Parameters with vulnerability > 0.8"
        )
        table.add_row(
            "Medium Vulnerability",
            f"{summary.get('medium_vulnerability_count', 0)}",
            "Parameters with vulnerability 0.5-0.8"
        )
        table.add_row(
            "Overall Vulnerability",
            f"{summary.get('overall_vulnerability_score', 0.0):.3f}",
            "Average vulnerability score"
        )
        table.add_row(
            "Critical Configurations",
            f"{len(critical_configs)}",
            "Eigenvalue configurations requiring attention"
        )

        if pac_bounds:
            global_bounds = pac_bounds.get("global_bounds", {})
            table.add_row(
                "PAC Upper Bound",
                f"{global_bounds.get('upper_bound', 0.0):.3f}",
                f"{confidence_level*100}% confidence interval"
            )
            table.add_row(
                "Vulnerability Certificates",
                f"{len(pac_bounds.get('vulnerability_certificates', []))}",
                "Theoretical guarantees issued"
            )

        console.print(table)

        # Display critical configurations
        if critical_configs:
            console.print(f"\n[bold red]🚨 Top {min(5, len(critical_configs))} Critical Configurations:[/bold red]")

            config_table = Table()
            config_table.add_column("Rank", style="cyan")
            config_table.add_column("Layer.Parameter", style="yellow")
            config_table.add_column("Vulnerability", style="red")
            config_table.add_column("Spectral Gap", style="blue")
            config_table.add_column("Condition #", style="green")

            for i, config in enumerate(critical_configs[:5]):
                sig = config["spectral_signature"]
                config_table.add_row(
                    f"{i+1}",
                    f"{sig.layer_name}.{sig.parameter_name}",
                    f"{sig.vulnerability_score:.3f}",
                    f"{sig.spectral_gap:.3f}",
                    f"{sig.condition_number:.1f}"
                )

            console.print(config_table)

        # Display recommendations
        console.print(f"\n[bold blue]💡 Recommendations:[/bold blue]")
        if summary.get('high_vulnerability_count', 0) > 0:
            console.print("• Consider spectral regularization for high-vulnerability parameters")
            console.print("• Implement eigenvalue smoothing for phase transition mitigation")
            console.print("• Monitor spectral properties during training")

        if pac_bounds and len(pac_bounds.get('vulnerability_certificates', [])) > 0:
            console.print("• Review issued vulnerability certificates for security implications")
            console.print("• Consider implementing certified defenses for guaranteed robustness")

        console.print("• Integrate spectral analysis into your security monitoring pipeline")

    except Exception as e:
        console.print(f"[bold red]❌ Spectral analysis failed: {e}[/bold red]")
        raise


def _generate_publication_tables(research_data: Dict[str, Any], output_path: Path):
    """Generate publication-ready tables and figures."""
    import pandas as pd

    # Table 1: Discovery Statistics
    stats = research_data["statistics"]
    table1_data = {
        "Metric": ["Total Weights Analyzed", "Critical Weights Found", "Discovery Rate (%)",
                  "Average Sensitivity", "Layer Coverage", "Component Diversity"],
        "Value": [f"{stats['total_weights_analyzed']:,}", stats['critical_weights_found'],
                 f"{stats['discovery_rate']:.6f}", f"{stats['avg_sensitivity']:.6f}",
                 stats['layer_coverage'], stats['component_diversity']]
    }

    pd.DataFrame(table1_data).to_csv(output_path / "table1_discovery_statistics.csv", index=False)

    # Table 2: Layer Distribution
    layer_dist = research_data["analysis_results"]["summary"]["layer_distribution"]
    table2_data = {
        "Layer": list(layer_dist.keys()),
        "Critical_Weights": list(layer_dist.values())
    }

    pd.DataFrame(table2_data).to_csv(output_path / "table2_layer_distribution.csv", index=False)

    # Table 3: Top Critical Weights
    if research_data["discovered_weights"]:
        top_weights = research_data["discovered_weights"][:20]  # Top 20 for publication
        table3_data = []

        for i, weight in enumerate(top_weights):
            table3_data.append({
                "Rank": i + 1,
                "Layer": weight["layer_index"],
                "Component": weight["component"],
                "Row": weight["coordinates"][0],
                "Col": weight["coordinates"][1],
                "Sensitivity": f"{weight['sensitivity_score']:.6f}"
            })

        pd.DataFrame(table3_data).to_csv(output_path / "table3_top_critical_weights.csv", index=False)


def _generate_replication_package(research_data: Dict[str, Any], output_path: Path, model_name: str):
    """Generate complete replication package for research."""
    import json

    # Replication metadata
    replication_data = {
        "model_name": model_name,
        "extraction_timestamp": pd.Timestamp.now().isoformat(),
        "methodology": {
            "activation_threshold": 1e3,
            "sensitivity_calculation": "hessian_approximation",
            "top_k_percentage": 0.001,
            "layer_focus": "early_layers_0_to_3",
            "perplexity_validation": "100x_threshold"
        },
        "full_results": research_data,
        "replication_commands": [
            f"cwa extract-critical-weights {model_name} --mode super_weight_discovery --top-k-percent 0.001 --layer-focus early",
            f"cwa validate-super-weights {model_name} --coordinates 'DISCOVERED_COORDINATES' --perplexity-threshold 100"
        ]
    }

    with open(output_path / "replication_package.json", 'w') as f:
        json.dump(replication_data, f, indent=2, default=str)

    # Generate coordinate files for replication
    if research_data["discovered_weights"]:
        coordinates = []
        for weight in research_data["discovered_weights"]:
            coord = (weight["layer_index"], weight["component"], weight["coordinates"])
            coordinates.append(coord)

        with open(output_path / "discovered_coordinates.txt", 'w') as f:
            f.write(str(coordinates))


# ============================================================================
# PHASE 2 COMMANDS: Advanced Security & Game-Theoretic Analysis
# ============================================================================

@app.command("monitor-realtime")
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
        from ..monitoring.realtime_monitor import RealtimeSecurityMonitor, MonitoringConfig
        from ..monitoring.circuit_breaker import CircuitBreakerConfig
        from ..monitoring.anomaly_detector import DetectionConfig

        console.print(f"[bold green]🕰️ Starting real-time monitoring for {model_name}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"realtime_monitoring_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        model_manager = LambdaLabsLLMManager()
        model, tokenizer = model_manager.load_model(model_name, device=device)

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
        with console.status("[bold green]Running real-time monitoring..."):
            monitoring_results = []

            for i, input_batch in enumerate(sample_inputs):
                if i >= 50:  # Limit for demo
                    break

                # Monitor inference
                response = monitor.monitor_inference(model, input_batch)
                monitoring_results.append(response)

                if response.alerts:
                    console.print(f"[red]⚠️  Alert detected at batch {i}: {len(response.alerts)} anomalies[/red]")

        # Save results
        results = {
            "monitoring_session": {
                "model_name": model_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "algorithms": algorithms,
                "total_batches": len(monitoring_results),
                "total_alerts": sum(len(r.alerts) for r in monitoring_results)
            },
            "performance_metrics": monitor.get_performance_summary(),
            "anomaly_summary": monitor.get_anomaly_summary()
        }

        # Export results
        with open(output_path / "monitoring_session.json", 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)

        # Display summary
        table = Table(title="Real-Time Monitoring Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description")

        table.add_row("Batches Processed", str(len(monitoring_results)), "Total inference batches monitored")
        table.add_row("Total Alerts", str(results["monitoring_session"]["total_alerts"]), "Anomalies detected")
        table.add_row("Average Latency", f"{results['performance_metrics'].get('avg_latency_ms', 0):.2f}ms", "Per-batch processing time")
        table.add_row("Circuit Breaker Trips", str(results['performance_metrics'].get('circuit_breaker_trips', 0)), "Fail-safe activations")

        console.print(table)
        console.print(f"[bold green]✅ Monitoring results saved to: {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Real-time monitoring failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("analyze-game-theory")
def analyze_game_theory(
    model_name: str,
    game_types: str = "nash_equilibrium,cooperative,evolutionary",
    max_players: int = 50,
    strategy_space_size: int = 10,
    convergence_threshold: float = 1e-6,
    coalition_analysis: str = "shapley_values,core_analysis",
    dynamics_type: str = "replicator",
    output_dir: Optional[str] = None,
    device: str = "auto"
):
    """Game-theoretic weight analysis (Phase 2)."""
    try:
        from ..game_theory.neurogame_analyzer import NeuroGameAnalyzer, GameConfig
        from ..game_theory.game_theoretic_analyzer import GameTheoreticWeightAnalyzer, GameConfiguration
        from ..game_theory.cooperative_analyzer import CooperativeGameAnalyzer, CooperativeGameConfig
        from ..game_theory.evolutionary_analyzer import EvolutionaryStabilityAnalyzer, EvolutionaryConfig

        console.print(f"[bold green]🎮 Starting game-theoretic analysis for {model_name}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"game_theory_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        model_manager = LambdaLabsLLMManager()
        model, tokenizer = model_manager.load_model(model_name, device=device)

        # Create sample inputs
        sample_inputs = create_sample_data(tokenizer, num_samples=10)
        inputs = sample_inputs[0]  # Use first batch

        game_types_list = [gt.strip() for gt in game_types.split(",")]
        results = {}

        # Nash Equilibrium Analysis
        if "nash_equilibrium" in game_types_list:
            console.print("[blue]🎯 Running Nash equilibrium analysis...[/blue]")
            config = GameConfiguration(
                max_players=max_players,
                strategy_space_size=strategy_space_size,
                convergence_threshold=convergence_threshold
            )
            analyzer = GameTheoreticWeightAnalyzer(config)

            with console.status("[bold green]Computing Nash equilibrium..."):
                equilibrium = analyzer.analyze_weight_game(model, inputs)

            if equilibrium:
                results["nash_equilibrium"] = {
                    "vulnerability_score": equilibrium.vulnerability_score,
                    "stability_score": equilibrium.stability_score,
                    "convergence_iterations": equilibrium.convergence_iterations,
                    "critical_weights": equilibrium.critical_weights
                }
                console.print(f"[green]✅ Nash equilibrium found (vulnerability: {equilibrium.vulnerability_score:.3f})[/green]")
            else:
                console.print("[yellow]⚠️  No Nash equilibrium found[/yellow]")

        # Cooperative Game Analysis
        if "cooperative" in game_types_list:
            console.print("[blue]🤝 Running cooperative game analysis...[/blue]")
            config = CooperativeGameConfig(
                max_coalition_size=min(max_players, 10),
                compute_shapley_values=True
            )
            analyzer = CooperativeGameAnalyzer(config)

            with console.status("[bold green]Analyzing coalition structures..."):
                coalition_structure = analyzer.analyze_cooperative_structure(model, inputs)

            if coalition_structure:
                results["cooperative"] = {
                    "stability_score": coalition_structure.stability_score,
                    "efficiency_score": coalition_structure.efficiency_score,
                    "num_coalitions": len(coalition_structure.coalitions),
                    "shapley_values": dict(list(coalition_structure.shapley_values.items())[:10])  # Top 10
                }
                console.print(f"[green]✅ Coalition analysis complete (stability: {coalition_structure.stability_score:.3f})[/green]")

        # Evolutionary Stability Analysis
        if "evolutionary" in game_types_list:
            console.print("[blue]🧬 Running evolutionary stability analysis...[/blue]")
            config = EvolutionaryConfig(
                dynamics_type=getattr(__import__('src.cwa.game_theory.evolutionary_analyzer', fromlist=['EvolutionaryDynamics']).EvolutionaryDynamics, dynamics_type.upper()),
                time_horizon=100.0
            )
            analyzer = EvolutionaryStabilityAnalyzer(config)

            with console.status("[bold green]Computing evolutionary stable strategies..."):
                ess_strategies = analyzer.analyze_evolutionary_stability(model, inputs)

            if ess_strategies:
                results["evolutionary"] = {
                    "num_ess": len(ess_strategies),
                    "average_stability": np.mean([ess.stability_score for ess in ess_strategies]),
                    "average_basin_size": np.mean([ess.basin_size for ess in ess_strategies]),
                    "strategies": [
                        {
                            "stability_type": ess.stability_type.value,
                            "basin_size": ess.basin_size,
                            "convergence_rate": ess.convergence_rate
                        } for ess in ess_strategies[:5]  # Top 5
                    ]
                }
                console.print(f"[green]✅ Found {len(ess_strategies)} evolutionarily stable strategies[/green]")

        # Save results
        analysis_summary = {
            "model_name": model_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "game_types": game_types_list,
            "configuration": {
                "max_players": max_players,
                "strategy_space_size": strategy_space_size,
                "convergence_threshold": convergence_threshold
            },
            "results": results
        }

        with open(output_path / "game_theory_analysis.json", 'w') as f:
            import json
            json.dump(analysis_summary, f, indent=2, default=str)

        # Display summary table
        table = Table(title="Game-Theoretic Analysis Summary")
        table.add_column("Analysis Type", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Key Metric")

        for game_type in game_types_list:
            if game_type in results:
                if game_type == "nash_equilibrium":
                    status = "✅ Complete"
                    metric = f"Vulnerability: {results[game_type]['vulnerability_score']:.3f}"
                elif game_type == "cooperative":
                    status = "✅ Complete"
                    metric = f"Stability: {results[game_type]['stability_score']:.3f}"
                elif game_type == "evolutionary":
                    status = "✅ Complete"
                    metric = f"ESS Found: {results[game_type]['num_ess']}"
                else:
                    status = "✅ Complete"
                    metric = "Analysis successful"
            else:
                status = "❌ Failed"
                metric = "No results"

            table.add_row(game_type.replace("_", " ").title(), status, metric)

        console.print(table)
        console.print(f"[bold green]✅ Game theory analysis saved to: {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Game-theoretic analysis failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("analyze-transfer")
def analyze_transfer(
    source_model: str,
    target_model: str,
    mapping_strategies: str = "geometric,semantic,interpolation",
    transfer_types: str = "architecture,vulnerability",
    similarity_threshold: float = 0.7,
    vulnerability_transfer_analysis: bool = True,
    output_dir: Optional[str] = None,
    device: str = "auto"
):
    """Cross-architecture transfer analysis (Phase 2)."""
    try:
        from ..transfer.transfer_analyzer import TransferAnalyzer, TransferConfig
        from ..transfer.architecture_mapper import ArchitectureMapper, MappingConfig, MappingStrategy

        console.print(f"[bold green]🔄 Analyzing transfer: {source_model} → {target_model}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"transfer_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load models
        model_manager = LambdaLabsLLMManager()

        console.print(f"[blue]Loading source model: {source_model}[/blue]")
        source_model_obj, source_tokenizer = model_manager.load_model(source_model, device=device)

        console.print(f"[blue]Loading target model: {target_model}[/blue]")
        target_model_obj, target_tokenizer = model_manager.load_model(target_model, device=device)

        # Configure transfer analysis
        transfer_config = TransferConfig(
            similarity_threshold=similarity_threshold
        )
        mapping_config = MappingConfig(
            compatibility_threshold=similarity_threshold
        )

        # Initialize analyzers
        transfer_analyzer = TransferAnalyzer(transfer_config)
        architecture_mapper = ArchitectureMapper(mapping_config)

        # Parse mapping strategies
        strategies_list = [s.strip() for s in mapping_strategies.split(",")]

        results = {}

        # Transfer Pattern Analysis
        console.print("[blue]🔍 Analyzing transfer patterns...[/blue]")
        with console.status("[bold green]Extracting and matching patterns..."):
            transfer_patterns = transfer_analyzer.analyze_transfer_potential(
                source_model_obj, target_model_obj, source_model, target_model
            )

        if transfer_patterns:
            results["transfer_patterns"] = {
                "num_patterns": len(transfer_patterns),
                "average_transferability": np.mean([p.transferability_score for p in transfer_patterns]),
                "high_quality_patterns": len([p for p in transfer_patterns if p.transferability_score > 0.8]),
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "transfer_type": p.transfer_type.value,
                        "transferability_score": p.transferability_score,
                        "pattern_similarity": p.pattern_similarity,
                        "semantic_alignment": p.semantic_alignment
                    } for p in transfer_patterns[:10]  # Top 10
                ]
            }

        # Architecture Mapping Analysis
        for strategy_name in strategies_list:
            console.print(f"[blue]🗺️ Analyzing {strategy_name} mapping...[/blue]")

            try:
                strategy = getattr(MappingStrategy, strategy_name.upper() + "_MAPPING")

                with console.status(f"[bold green]Creating {strategy_name} mappings..."):
                    mappings = architecture_mapper.create_architecture_mapping(
                        source_model_obj, target_model_obj, strategy
                    )

                if mappings:
                    mapping_quality = architecture_mapper.evaluate_mapping_quality(mappings)
                    results[f"{strategy_name}_mapping"] = {
                        "num_mappings": len(mappings),
                        "quality_distribution": mapping_quality.get("quality_distribution", {}),
                        "average_compatibility": mapping_quality.get("average_compatibility", 0.0),
                        "high_quality_mappings": mapping_quality.get("high_quality_mappings", 0)
                    }

            except AttributeError:
                console.print(f"[yellow]⚠️  Strategy {strategy_name} not recognized[/yellow]")

        # Transfer Success Prediction
        console.print("[blue]📊 Predicting transfer success...[/blue]")
        success_prediction = transfer_analyzer.predict_transfer_success(source_model_obj, target_model_obj)
        results["success_prediction"] = success_prediction

        # Save results
        analysis_summary = {
            "source_model": source_model,
            "target_model": target_model,
            "timestamp": pd.Timestamp.now().isoformat(),
            "configuration": {
                "mapping_strategies": strategies_list,
                "similarity_threshold": similarity_threshold,
                "vulnerability_transfer_analysis": vulnerability_transfer_analysis
            },
            "results": results
        }

        with open(output_path / "transfer_analysis.json", 'w') as f:
            import json
            json.dump(analysis_summary, f, indent=2, default=str)

        # Display summary
        table = Table(title="Transfer Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description")

        if "transfer_patterns" in results:
            table.add_row(
                "Transfer Patterns",
                str(results["transfer_patterns"]["num_patterns"]),
                "Transferable patterns found"
            )
            table.add_row(
                "Average Transferability",
                f"{results['transfer_patterns']['average_transferability']:.3f}",
                "Mean transferability score"
            )

        if "success_prediction" in results:
            table.add_row(
                "Transfer Success Probability",
                f"{results['success_prediction']['transfer_success_probability']:.3f}",
                "Predicted likelihood of successful transfer"
            )
            table.add_row(
                "Performance Retention",
                f"{results['success_prediction']['expected_performance_retention']:.3f}",
                "Expected performance after transfer"
            )

        for strategy in strategies_list:
            strategy_key = f"{strategy}_mapping"
            if strategy_key in results:
                table.add_row(
                    f"{strategy.title()} Mappings",
                    str(results[strategy_key]["num_mappings"]),
                    f"Successful {strategy} mappings created"
                )

        console.print(table)
        console.print(f"[bold green]✅ Transfer analysis saved to: {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Transfer analysis failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()