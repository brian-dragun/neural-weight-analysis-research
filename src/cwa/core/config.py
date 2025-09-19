"""Configuration management using Pydantic."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


class ModelConfig(BaseModel):
    """Hugging Face LLM model configuration optimized for Lambda Labs GPUs."""
    name: str = "microsoft/DialoGPT-small"  # Default to small model for testing
    model_size: str = "small"  # small, medium, large
    device: str = "cuda"  # Lambda Labs: Default to CUDA (GPU-first)
    torch_dtype: str = "float16"  # Lambda Labs: Default to FP16 for GPU efficiency
    max_length: int = 512
    quantization: Optional[str] = None  # None, "8bit", "4bit"
    device_map: Optional[str] = "auto"  # Lambda Labs: Auto device mapping
    max_memory: Optional[Dict[str, str]] = None  # e.g., {"0": "20GiB", "cpu": "50GiB"}
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = False

    # Lambda Labs GPU optimizations
    use_flash_attention_2: bool = True  # Enable Flash Attention on supported models
    torch_compile: bool = False  # PyTorch 2.0 compilation (can add later)

    # Hugging Face specific options
    use_auth_token: Optional[str] = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = "/tmp/hf_cache"  # Lambda Labs: Use faster local storage


class SensitivityConfig(BaseModel):
    """Sensitivity analysis configuration."""
    metric: str = "basic_gradient"
    top_k: int = 100
    mode: str = "global"  # global or per_layer


class PerturbationConfig(BaseModel):
    """Perturbation configuration."""
    methods: List[str] = ["zero", "noise"]
    scales: Dict[str, float] = {"noise": 0.1}


class SecurityConfig(BaseModel):
    """Basic security configuration."""
    enabled: bool = True
    vulnerability_threshold: float = 0.5


class ExperimentConfig(BaseModel):
    """Main experiment configuration."""
    name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    perturbation: PerturbationConfig = Field(default_factory=PerturbationConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    data_samples: int = 100
    random_seed: int = 42
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)