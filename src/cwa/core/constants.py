"""Common constants for Critical Weight Analysis."""

# Default model parameters
DEFAULT_VOCAB_SIZE = 50257
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_BATCH_SIZE = 8

# Analysis thresholds
ACTIVATION_THRESHOLD = 1e3
PERPLEXITY_THRESHOLD = 100
SENSITIVITY_PERCENTILE = 99.999  # Top 0.001%

# Layer ranges
EARLY_LAYER_RANGE = (0, 4)  # Layers 0-3 as per research

# Device settings
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float16"

# Cache directories
HF_CACHE_DIR = "/tmp/hf_cache"

# File extensions
SUPPORTED_MODEL_FORMATS = [".pt", ".pth", ".safetensors"]
SUPPORTED_CONFIG_FORMATS = [".yaml", ".yml", ".json"]