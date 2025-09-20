"""Phase A, B, C security analysis commands."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import yaml
from typing import Optional
import pandas as pd

from ...core.models import LambdaLabsLLMManager
from ...core.data import create_sample_data, create_data_loader
from ...sensitivity.security_analyzer import SecurityWeightAnalyzer
from ...security.targeted_attacks import TargetedAttackSimulator
from ...security.fault_injection import FaultInjector
from ...security.defense_mechanisms import DefenseManager
from ...security.weight_protection import CriticalWeightProtector
from ...utils.logging import setup_logging

phase_app = typer.Typer(help="Phase A/B/C security analysis commands")
console = Console()


@phase_app.command("phase-a")
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


@phase_app.command("phase-b")
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


@phase_app.command("phase-c")
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


@phase_app.command("run-complete-pipeline")
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