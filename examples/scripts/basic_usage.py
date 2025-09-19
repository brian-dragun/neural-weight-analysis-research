"""Basic usage example for Critical Weight Analysis tool."""

from pathlib import Path
from cwa.core.config import ExperimentConfig, ModelConfig
from cwa.core.models import LambdaLabsLLMManager
from cwa.core.data import create_sample_data, create_data_loader
from cwa.sensitivity.registry import get_sensitivity_metric


def main():
    """Run a basic sensitivity analysis example."""
    # Create configuration
    config = ExperimentConfig(
        name="basic_example",
        model=ModelConfig(
            name="microsoft/DialoGPT-small",
            model_size="small",
            device="cuda",
            torch_dtype="float16"
        ),
        data_samples=10
    )

    print(f"Running example: {config.name}")

    # Load model
    print("Loading model...")
    model_manager = LambdaLabsLLMManager(config.model.dict())
    model = model_manager.load_model()

    # Create sample data
    print("Creating sample data...")
    sample_texts = create_sample_data(config.data_samples)
    data_loader = create_data_loader(sample_texts, model_manager.tokenizer)

    # Run sensitivity analysis
    print("Running sensitivity analysis...")
    sensitivity_func = get_sensitivity_metric(config.sensitivity.metric)
    results = sensitivity_func(
        model=model,
        data_loader=data_loader,
        top_k=config.sensitivity.top_k
    )

    # Display results
    print(f"\\nResults:")
    print(f"- Metric: {results.metric_name}")
    print(f"- Top weights found: {len(results.top_k_weights)}")
    if results.top_k_weights:
        print(f"- Highest score: {results.top_k_weights[0][2]:.6f}")
        print(f"- Most critical layer: {results.top_k_weights[0][0]}")

    print("\\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()