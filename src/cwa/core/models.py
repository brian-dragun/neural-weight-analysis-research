"""Lambda Labs optimized Hugging Face LLM model management utilities."""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from typing import Dict, Any, Optional, Union
import logging
import psutil
from pathlib import Path

# Lambda Labs specific imports
try:
    import nvidia_ml_py as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class LambdaLabsLLMManager:
    """Manages Hugging Face LLM loading optimized for Lambda Labs GPU VMs."""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config.get("name", "microsoft/DialoGPT-small")
        self.model_size = config.get("model_size", "small")
        self.device = config.get("device", "cuda")
        self.torch_dtype = config.get("torch_dtype", "float16")
        self.quantization = config.get("quantization")
        self.device_map = config.get("device_map", "auto")
        self.max_memory = config.get("max_memory")
        self.low_cpu_mem_usage = config.get("low_cpu_mem_usage", True)
        self.trust_remote_code = config.get("trust_remote_code", False)
        self.max_length = config.get("max_length", 512)
        self.use_flash_attention_2 = config.get("use_flash_attention_2", True)
        self.cache_dir = config.get("cache_dir", "/tmp/hf_cache")

        self.model = None
        self.tokenizer = None
        self.model_info = {}

        # Initialize NVIDIA monitoring for Lambda Labs
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_enabled = True
                logger.info("NVIDIA monitoring enabled for Lambda Labs")
            except Exception:
                self.nvml_enabled = False
                logger.warning("NVIDIA monitoring not available")
        else:
            self.nvml_enabled = False

    def _detect_lambda_gpu_config(self) -> Dict[str, Any]:
        """Detect Lambda Labs GPU configuration and optimize accordingly."""
        gpu_info = {"gpu_count": 0, "total_gpu_memory": 0, "gpu_names": []}

        if torch.cuda.is_available():
            gpu_info["gpu_count"] = torch.cuda.device_count()

            if self.nvml_enabled:
                try:
                    for i in range(gpu_info["gpu_count"]):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_info["gpu_names"].append(name)
                        gpu_info["total_gpu_memory"] += memory_info.total / (1024**3)  # GB

                        logger.info(f"Detected Lambda Labs GPU {i}: {name} with {memory_info.total / (1024**3):.1f} GB")

                except Exception as e:
                    logger.warning(f"Failed to get detailed GPU info: {e}")

            # Lambda Labs optimization based on detected hardware
            if gpu_info["gpu_count"] >= 2:
                # Multi-GPU setup - use device map auto
                self.device_map = "auto"
                logger.info("Multi-GPU Lambda setup detected - using auto device mapping")

            # Memory optimization for Lambda Labs VMs
            if gpu_info["total_gpu_memory"] > 70:  # A100 80GB or similar
                self.max_memory = {"0": "70GiB", "cpu": "100GiB"}
                logger.info("High-memory Lambda GPU detected - optimizing for large models")
            elif gpu_info["total_gpu_memory"] > 40:  # A100 40GB or similar
                self.max_memory = {"0": "35GiB", "cpu": "50GiB"}
                logger.info("Lambda A100 40GB detected - optimizing memory allocation")
            elif gpu_info["total_gpu_memory"] > 20:  # RTX 4090 or similar
                self.max_memory = {"0": "20GiB", "cpu": "30GiB"}
                logger.info("Lambda RTX GPU detected - conservative memory allocation")

        return gpu_info

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration optimized for Lambda Labs GPUs."""
        if self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0  # Optimized for Lambda GPUs
            )
        elif self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8  # Lambda Labs optimization
            )
        return None

    def _log_lambda_memory_usage(self, stage: str):
        """Log memory usage with Lambda Labs specific details."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
                logger.info(f"{stage} - Lambda GPU {i}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")

        cpu_memory = psutil.virtual_memory()
        logger.info(f"{stage} - Lambda VM CPU Memory: {cpu_memory.percent:.1f}% used ({cpu_memory.used / 1024**3:.1f}GB / {cpu_memory.total / 1024**3:.1f}GB)")

    def load_model(self) -> torch.nn.Module:
        """Load Hugging Face LLM optimized for Lambda Labs GPU VMs."""
        try:
            logger.info(f"Loading model on Lambda Labs: {self.model_name} (Size: {self.model_size})")
            self._log_lambda_memory_usage("Before loading")

            # Verify GPU availability on Lambda Labs
            if not torch.cuda.is_available():
                logger.error("CUDA not available - Lambda Labs GPU not detected!")
                raise RuntimeError("Lambda Labs GPU environment required but CUDA not available")

            logger.info(f"Lambda Labs GPU detected: {torch.cuda.get_device_name()}")

            # Load tokenizer first (lightweight)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir
            )

            # Add special tokens if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "[PAD]"

            # Determine Lambda Labs optimized loading strategy
            load_kwargs = self._determine_device_strategy()

            logger.info(f"Loading model with Lambda Labs optimization: {load_kwargs}")

            # Try to load as causal LM first (most LLMs), fallback to base model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )
                self.model_info["model_type"] = "CausalLM"
            except Exception as e:
                logger.warning(f"Failed to load as CausalLM, trying base model: {e}")
                # Remove Flash Attention for base models
                if "use_flash_attention_2" in load_kwargs:
                    load_kwargs.pop("use_flash_attention_2")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **load_kwargs
                )
                self.model_info["model_type"] = "BaseModel"

            # Set to evaluation mode
            self.model.eval()

            # Lambda Labs: Optimize for inference
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()

            self._log_lambda_memory_usage("After loading")

            # Gather model information
            self._collect_model_info()

            logger.info(f"Lambda Labs model loaded successfully: {self.get_model_info()}")
            return self.model

        except Exception as e:
            logger.error(f"Failed to load model on Lambda Labs {self.model_name}: {e}")
            raise

    def _determine_device_strategy(self) -> Dict[str, Any]:
        """Determine optimal device strategy for Lambda Labs GPU VMs."""
        # Detect Lambda Labs GPU configuration first
        gpu_config = self._detect_lambda_gpu_config()

        load_kwargs = {
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir
        }

        # Lambda Labs: Default to float16 for GPU efficiency
        if self.torch_dtype == "auto":
            load_kwargs["torch_dtype"] = torch.float16  # Lambda default
        else:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16
            }
            load_kwargs["torch_dtype"] = dtype_map.get(self.torch_dtype, torch.float16)

        # Lambda Labs: Always use device mapping for optimal GPU utilization
        load_kwargs["device_map"] = self.device_map
        if self.max_memory:
            load_kwargs["max_memory"] = self.max_memory

        # Add quantization for medium/large models on Lambda Labs
        quantization_config = self._get_quantization_config()
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config

        # Lambda Labs: Enable Flash Attention 2 for supported models
        if self.use_flash_attention_2:
            load_kwargs["use_flash_attention_2"] = True

        return load_kwargs

    def _collect_model_info(self):
        """Collect comprehensive model information."""
        if self.model is None:
            return

        # Basic parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.model_info.update({
            "model_name": self.model_name,
            "model_size_category": self.model_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameters_in_millions": round(total_params / 1_000_000, 2),
            "architecture": self.model.config.model_type if hasattr(self.model, 'config') else 'unknown',
            "hidden_size": getattr(self.model.config, 'hidden_size', 'unknown'),
            "num_layers": getattr(self.model.config, 'num_hidden_layers', 'unknown'),
            "vocab_size": getattr(self.model.config, 'vocab_size', len(self.tokenizer) if self.tokenizer else 'unknown'),
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'unknown'),
            "device_placement": str(next(self.model.parameters()).device) if self.model.parameters() else 'unknown',
            "dtype": str(next(self.model.parameters()).dtype) if self.model.parameters() else 'unknown',
            "quantized": self.quantization is not None,
        })

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return self.model_info.copy()

    def get_named_parameters(self) -> Dict[str, torch.Tensor]:
        """Get named parameters for analysis."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return dict(self.model.named_parameters())

    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for the model."""
        if self.model is None:
            return {"error": "Model not loaded"}

        model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        model_size_gb = model_size_bytes / (1024**3)

        return {
            "model_size_gb": model_size_gb,
            "estimated_inference_gb": model_size_gb * 1.2,  # Rough estimate
            "current_gpu_usage_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        }


# Keep backwards compatibility
HuggingFaceLLMManager = LambdaLabsLLMManager
ModelManager = LambdaLabsLLMManager