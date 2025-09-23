"""Advanced analysis commands - game theory and transfer learning."""

import typer
from rich.console import Console
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from ...core.models import LambdaLabsLLMManager
from ...core.data import create_sample_data

analysis_app = typer.Typer(help="Advanced analysis commands")
console = Console()


@analysis_app.command("analyze-game-theory")
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
        from ...game_theory.neurogame_analyzer import NeuroGameAnalyzer
        from ...game_theory.game_theoretic_analyzer import GameTheoreticWeightAnalyzer, GameConfiguration
        from ...game_theory.cooperative_analyzer import CooperativeGameAnalyzer, CooperativeGameConfig
        from ...game_theory.evolutionary_analyzer import EvolutionaryStabilityAnalyzer, EvolutionaryConfig

        console.print(f"[bold green]🎮 Starting game-theoretic analysis for {model_name}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"game_theory_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        config = {"name": model_name, "device": device}
        model_manager = LambdaLabsLLMManager(config)
        model = model_manager.load_model()
        tokenizer = model_manager.tokenizer

        # Create sample inputs
        sample_texts = create_sample_data(num_samples=10)
        # Tokenize the first sample for analysis
        inputs = tokenizer(sample_texts[0], return_tensors="pt", padding=True, truncation=True)

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

        console.print(f"[bold green]✅ Game theory analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Game-theoretic analysis failed: {e}[/red]")
        raise typer.Exit(1)


@analysis_app.command("analyze-transfer")
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
        from ...transfer.transfer_analyzer import TransferAnalyzer, TransferConfig
        from ...transfer.architecture_mapper import ArchitectureMapper, MappingConfig, MappingStrategy

        console.print(f"[bold green]🔄 Analyzing transfer: {source_model} → {target_model}[/bold green]")

        # Setup output directory
        if not output_dir:
            output_dir = f"transfer_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load models
        console.print(f"[blue]Loading source model: {source_model}[/blue]")
        source_config = {"name": source_model, "device": device}
        source_model_manager = LambdaLabsLLMManager(source_config)
        source_model_obj = source_model_manager.load_model()
        source_tokenizer = source_model_manager.tokenizer

        console.print(f"[blue]Loading target model: {target_model}[/blue]")
        target_config = {"name": target_model, "device": device}
        target_model_manager = LambdaLabsLLMManager(target_config)
        target_model_obj = target_model_manager.load_model()
        target_tokenizer = target_model_manager.tokenizer

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
        transfer_patterns = transfer_analyzer.analyze_transfer_potential(
            source_model_obj, target_model_obj, source_model, target_model
        )

        if transfer_patterns:
            results["transfer_patterns"] = {
                "num_patterns": len(transfer_patterns),
                "average_transferability": np.mean([p.transferability_score for p in transfer_patterns]),
                "high_quality_patterns": len([p for p in transfer_patterns if p.transferability_score > 0.8])
            }

        console.print(f"[bold green]✅ Transfer analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Transfer analysis failed: {e}[/red]")
        raise typer.Exit(1)