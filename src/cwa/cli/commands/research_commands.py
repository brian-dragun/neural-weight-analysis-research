"""Research-focused commands for super weight analysis."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
import yaml
import pandas as pd
import logging
from typing import Optional, Dict, Any
import ast

from ...core.models import LambdaLabsLLMManager
from ...research.super_weight_analyzer import SuperWeightAnalyzer
from ...utils.logging import setup_logging

research_app = typer.Typer(help="Research and super weight analysis commands")
console = Console()


@research_app.command("extract-critical-weights")
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
    """
    console.print(f"[bold blue]🔬 Research Mode: Critical Weight Extraction[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Mode: {mode}")
    console.print(f"Sensitivity Threshold: {sensitivity_threshold}")
    console.print(f"Top-K Percentage: {top_k_percent}%")
    console.print(f"Layer Focus: {layer_focus}")

    try:
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        setup_logging(output_path)

        # Load model
        console.print("[yellow]Loading model for research analysis...[/yellow]")
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model()

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
        if 'error' in stats:
            table.add_row("Analysis Status", stats['error'])
            table.add_row("Total Weights Analyzed", "0")
            table.add_row("Critical Weights Found", "0")
            table.add_row("Discovery Rate", "0.000000%")
            table.add_row("Average Sensitivity", "0.000000")
            table.add_row("Layer Coverage", "0 layers")
        else:
            table.add_row("Total Weights Analyzed", f"{stats.get('total_weights_analyzed', 0):,}")
            table.add_row("Critical Weights Found", f"{stats.get('critical_weights_found', 0)}")
            table.add_row("Discovery Rate", f"{stats.get('discovery_rate', 0):.6f}%")
            table.add_row("Average Sensitivity", f"{stats.get('avg_sensitivity', 0):.6f}")
            table.add_row("Layer Coverage", f"{stats.get('layer_coverage', 0)} layers")
            table.add_row("Component Diversity", f"{stats.get('component_diversity', 'N/A')}")

        console.print(table)

        # Research findings
        analysis = research_data.get("analysis_results", {})
        console.print(f"\n[bold yellow]🔍 Key Research Findings:[/bold yellow]")

        # Safely access nested analysis results
        architectural = analysis.get('architectural', {})
        behavioral = analysis.get('behavioral', {})

        if architectural and behavioral:
            console.print(f"• Early Layer Concentration: {architectural.get('early_layer_concentration', 0):.1%}")
            console.print(f"• Avg Perplexity Impact: {behavioral.get('avg_perplexity_impact', 0):.1f}x")
            console.print(f"• Max Perplexity Impact: {behavioral.get('max_perplexity_impact', 0):.1f}x")
            console.print(f"• Significant Weights: {behavioral.get('significant_weights', 0)}/{behavioral.get('tested_weights', 0)}")
        else:
            console.print("❌ Critical weight extraction failed: No analysis results available")
            if stats.get('critical_weights_found', 0) == 0:
                console.print("💡 Suggestion: Try lowering sensitivity threshold or increasing top-k percentage")

    except Exception as e:
        console.print(f"[bold red]❌ Critical weight extraction failed: {e}[/bold red]")
        raise


@research_app.command("validate-super-weights")
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
    """
    console.print(f"[bold blue]🧪 Super Weight Validation[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Perplexity Threshold: {perplexity_threshold}x")

    try:
        # Parse coordinates
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
        model = model_manager.load_model()

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


@research_app.command("research-extract")
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
    """
    console.print(f"[bold blue]🎓 PhD Research Extraction Mode[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"Research Focus: {focus}")
    console.print(f"Analysis Types: {analysis_types}")

    try:
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
        model = model_manager.load_model()

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


@research_app.command("spectral-analysis")
def spectral_analysis(
    model_name: str,
    target_layers: Optional[str] = typer.Option(None, help="Comma-separated layer names to analyze (None for all)"),
    analysis_types: str = typer.Option("signatures,transitions,stability,correlations", help="Types of spectral analysis"),
    top_k: int = typer.Option(10, help="Number of top critical configurations to identify"),
    output_dir: str = typer.Option("spectral_analysis_results", help="Output directory"),
    device: str = typer.Option("cuda", help="Device to use (cuda/cpu/auto)"),
    include_pac_bounds: bool = typer.Option(True, help="Include PAC-Bayesian bounds"),
    confidence_level: float = typer.Option(0.95, help="Confidence level for PAC bounds")
):
    """
    Advanced spectral vulnerability analysis using eigenvalue-based methods.

    Analyzes neural network weight matrices using spectral learning to identify
    critical configurations, phase transitions, and vulnerability patterns.
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
        from ...sensitivity.spectral_analyzer import SpectralVulnerabilityAnalyzer

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
            "Overall Vulnerability",
            f"{summary.get('overall_vulnerability_score', 0.0):.3f}",
            "Average vulnerability score"
        )
        table.add_row(
            "Critical Configurations",
            f"{len(critical_configs)}",
            "Eigenvalue configurations requiring attention"
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

    except Exception as e:
        console.print(f"[bold red]❌ Spectral analysis failed: {e}[/bold red]")
        raise