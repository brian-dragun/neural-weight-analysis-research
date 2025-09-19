# Critical Weight Analysis Tool - Phase 1 Foundation

A research toolkit for analyzing critical weights in **Hugging Face Large Language Models** with integrated cybersecurity analysis capabilities, **optimized specifically for Lambda Labs GPU VMs**.

## 🚀 Phase 1 Features

This is the **foundation build** with clean, well-organized code that provides:

- **Lambda Labs GPU optimization** - Automatic GPU detection and memory management
- **Multi-scale model support** - From 124M to 70B+ parameter models
- **Basic sensitivity analysis** - Gradient-based weight importance scoring
- **Clean CLI interface** - Easy-to-use command line tools
- **Extensible architecture** - Ready for Phase 2 security enhancements

## 🛠️ Installation on Lambda Labs

```bash
# Clone the repository on your Lambda Labs VM
git clone <repo-url> critical-weight-analysis-v2
cd critical-weight-analysis-v2

# Install with uv (Python 3.12+)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# IMPORTANT: Install PyTorch with Lambda Labs CUDA 12.6
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install the tool
uv pip install -e .
```

## ⚡ Quick Start

```bash
# Verify Lambda Labs GPU setup
cwa info  # Should show your Lambda GPU details

# List available models optimized for Lambda Labs
cwa list-models

# Create a configuration for small model testing
cwa create-config --name my_test --model "microsoft/DialoGPT-small" --model-size small

# Run basic sensitivity analysis
cwa run my_test_config.yaml

# Get detailed model information
cwa model-info "microsoft/DialoGPT-small"
```

## 📋 Lambda Labs VM Recommendations by Model Size

### Small Models (Perfect for any Lambda GPU)
- **Models**: GPT-2 (124M), DialoGPT-small (124M), Phi-2 (2.7B)
- **Lambda Instance**: Any GPU instance (even RTX 4090)
- **Settings**: FP16, no quantization needed
- **Memory**: < 2GB GPU memory

### Medium Models (Ideal for Lambda A100 40GB)
- **Models**: Mistral-7B, LLaMA-2-7B, Phi-3-Mini
- **Lambda Instance**: A100 40GB recommended
- **Settings**: FP16 + 8-bit quantization
- **Memory**: ~20GB GPU memory

### Large Models (Requires Lambda A100 80GB or Multi-GPU)
- **Models**: LLaMA-2-13B, Mixtral-8x7B, LLaMA-2-70B
- **Lambda Instance**: A100 80GB or multi-GPU setup
- **Settings**: FP16 + 4-bit quantization + device mapping
- **Memory**: 40-70GB GPU memory

## 🧪 Example Workflows

### Single Lambda A100 40GB
```bash
# Medium model with 8-bit quantization
cwa create-config --name a100_mistral \\
    --model "mistralai/Mistral-7B-v0.1" \\
    --model-size medium \\
    --quantization 8bit \\
    --device cuda

cwa run a100_mistral_config.yaml
```

### Multi-GPU Lambda Setup
```bash
# Large model with device mapping
cwa create-config --name multi_gpu_llama \\
    --model "meta-llama/Llama-2-13b-hf" \\
    --model-size large \\
    --quantization 4bit \\
    --device cuda

cwa run multi_gpu_llama_config.yaml
```

## 🏗️ Project Structure

```
critical-weight-analysis-v2/
├── pyproject.toml              # Project configuration
├── src/cwa/                    # Main source code
│   ├── core/                   # Core abstractions
│   │   ├── interfaces.py       # Data structures and protocols
│   │   ├── config.py          # Configuration management
│   │   ├── models.py          # Lambda Labs LLM management
│   │   └── data.py            # Data handling utilities
│   ├── sensitivity/           # Sensitivity analysis
│   │   ├── basic_sensitivity.py
│   │   └── registry.py        # Metric registration system
│   ├── perturbation/          # Weight perturbation methods
│   ├── utils/                 # Utilities and helpers
│   └── cli/                   # Command line interface
├── configs/                   # Configuration templates
│   ├── models/               # Model-specific configs
│   └── experiments/          # Experiment templates
└── tests/                    # Test suite
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Run only Lambda Labs specific tests
pytest tests/test_basic_functionality.py::test_lambda_model_loading -v
```

## 📊 Performance Benchmarks on Lambda Labs

- **GPT-2 (124M)**: <10 seconds to load + analyze on any Lambda GPU
- **Mistral-7B**: <60 seconds to load + analyze on A100 40GB
- **LLaMA-13B**: <120 seconds to load on A100 80GB with 4-bit quantization

## 🛡️ Security & Code Quality

```bash
# Code formatting
black src/
ruff check src/

# Security scanning (when security dependencies added)
bandit -r src/
safety check
```

## 🔄 Next Phases

**Phase 2**: Advanced Security Analysis
- Adversarial attack simulation (FGSM, PGD, TextFooler)
- Hardware fault injection testing
- Attack detection mechanisms

**Phase 3**: Protection & Defense
- Critical weight protection strategies
- Defense mechanism implementation
- Robustness evaluation

## 🎯 Research Applications

- **Weight Importance Analysis** - Identify critical parameters for model performance
- **Sensitivity Mapping** - Understand model vulnerability patterns
- **Foundation for Security Research** - Clean base for advanced cybersecurity analysis
- **Multi-Model Comparison** - Compare sensitivity across different architectures

## 🤝 Contributing

This is a research tool built with clean, extensible code. The Phase 1 foundation provides:

- Clear abstractions and interfaces
- Comprehensive logging and error handling
- Lambda Labs GPU optimization
- Extensible registry systems for metrics and methods

Ready for Phase 2 security enhancements while maintaining code quality and performance.

## 📄 License

Academic and research use. Please cite appropriately in academic publications.

---

**Phase 1 Status**: ✅ Complete foundation with Lambda Labs optimization
**Next**: Ready for Phase 2 cybersecurity feature implementation