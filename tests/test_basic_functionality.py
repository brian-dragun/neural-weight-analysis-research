"""Basic functionality tests."""

import pytest
import torch
from pathlib import Path
import tempfile

from cwa.core.config import ExperimentConfig, ModelConfig
from cwa.core.models import LambdaLabsLLMManager
from cwa.core.data import create_sample_data


def test_config_loading():
    """Test configuration loading."""
    config = ExperimentConfig(name="test")
    assert config.name == "test"
    assert config.model.name == "microsoft/DialoGPT-small"  # default


def test_sample_data_creation():
    """Test sample data creation."""
    texts = create_sample_data(5)
    assert len(texts) == 5
    assert all(isinstance(text, str) for text in texts)


def test_lambda_model_manager_init():
    """Test Lambda Labs model manager initialization."""
    config = {
        "name": "microsoft/DialoGPT-small",
        "model_size": "small",
        "device": "cuda",
        "torch_dtype": "float16",
        "cache_dir": "/tmp/hf_cache"
    }
    manager = LambdaLabsLLMManager(config)
    assert manager.model_name == "microsoft/DialoGPT-small"
    assert manager.model_size == "small"
    assert manager.device == "cuda"


def test_lambda_gpu_detection():
    """Test Lambda Labs GPU detection."""
    config = {
        "name": "microsoft/DialoGPT-small",
        "model_size": "small",
        "device": "cuda"
    }
    manager = LambdaLabsLLMManager(config)

    if torch.cuda.is_available():
        gpu_config = manager._detect_lambda_gpu_config()
        assert "gpu_count" in gpu_config
        assert gpu_config["gpu_count"] > 0


def test_basic_experiment_config():
    """Test creating a basic experiment configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = ExperimentConfig(
            name="test_experiment",
            model=ModelConfig(name="microsoft/DialoGPT-small"),
            data_samples=3,
            output_dir=temp_dir
        )

        # Test that we can create the config
        assert config.name == "test_experiment"
        assert Path(temp_dir).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lambda_model_loading():
    """Test Lambda Labs model loading (only if CUDA is available)."""
    config = {
        "name": "microsoft/DialoGPT-small",
        "model_size": "small",
        "device": "cuda",
        "torch_dtype": "float16",
        "cache_dir": "/tmp/hf_cache"
    }
    manager = LambdaLabsLLMManager(config)
    model = manager.load_model()

    assert model is not None

    info = manager.get_model_info()
    assert "total_parameters" in info
    assert info["total_parameters"] > 0
    assert info["model_size_category"] == "small"