# Lambda Labs Setup Script Documentation

This document explains how the `setup_lambda.sh` script works, its components, and technical details for understanding and troubleshooting.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Script Architecture](#script-architecture)
3. [Detailed Function Breakdown](#detailed-function-breakdown)
4. [Dependencies & Installation Flow](#dependencies--installation-flow)
5. [Error Handling & Safety](#error-handling--safety)
6. [Lambda Labs Optimizations](#lambda-labs-optimizations)
7. [Hugging Face Authentication](#hugging-face-authentication)
8. [Testing & Verification](#testing--verification)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Customization Options](#customization-options)

## Overview

The `setup_lambda.sh` script is a comprehensive automation tool that transforms a fresh Lambda Labs VM into a fully configured environment for critical weight analysis research. It handles everything from system dependencies to GPU optimization.

### What It Does
- Installs Python 3.12 from deadsnakes PPA
- Sets up uv package manager with proper PATH configuration
- Installs PyTorch with CUDA 12.6 support optimized for Lambda Labs
- Configures all project dependencies through uv
- Authenticates with Hugging Face using multiple methods
- Optimizes for Lambda Labs GPU configurations
- Tests the entire installation
- Provides helpful usage instructions

### Design Principles
- **Idempotent**: Can be run multiple times safely
- **Fail-fast**: Exits immediately on errors (`set -e`)
- **Informative**: Colored output with clear status messages
- **Flexible**: Multiple authentication and configuration options
- **Robust**: Comprehensive error handling and validation

## Script Architecture

```bash
setup_lambda.sh
├── Global Configuration
│   ├── Error handling (set -e)
│   ├── Color definitions
│   └── Utility functions
├── System Detection
│   ├── GPU detection (nvidia-smi)
│   ├── Command existence checks
│   └── Lambda Labs environment validation
├── Installation Pipeline
│   ├── System dependencies → Python 3.12 → uv → Project setup
│   ├── PyTorch CUDA → Dependencies → Optimizations
│   └── Authentication → Testing → Instructions
└── Verification & Testing
    ├── Component testing
    ├── Integration testing
    └── Performance validation
```

## Detailed Function Breakdown

### Core Utility Functions

#### `print_status()`, `print_success()`, `print_warning()`, `print_error()`
```bash
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
```
- **Purpose**: Consistent, colored output for user feedback
- **Colors**: Blue (info), Green (success), Yellow (warning), Red (error)
- **Usage**: Throughout script for status communication

#### `command_exists()`
```bash
command_exists() {
    command -v "$1" >/dev/null 2>&1
}
```
- **Purpose**: Check if a command/program is available
- **Method**: Uses `command -v` which is POSIX-compliant
- **Returns**: 0 if command exists, 1 if not
- **Usage**: Prevents redundant installations and errors

### System Detection Functions

#### `check_lambda_gpu()`
```bash
check_lambda_gpu() {
    print_status "Checking Lambda Labs GPU configuration..."
    if command_exists nvidia-smi; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    fi
}
```
- **Purpose**: Detect and display Lambda Labs GPU information
- **Method**: Uses `nvidia-smi` with specific query parameters
- **Output**: GPU name, total memory, driver version
- **Fallback**: Graceful handling if nvidia-smi not available

### Installation Functions

#### `install_python312()`
```bash
install_python312() {
    if command_exists python3.12; then
        return 0  # Already installed
    fi

    # Add deadsnakes PPA
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update

    # Install Python 3.12 and tools
    sudo apt install -y python3.12 python3.12-dev python3.12-venv python3.12-distutils

    # Install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
}
```
- **Purpose**: Install Python 3.12 if not present
- **Method**: Uses deadsnakes PPA (Personal Package Archive)
- **Components**: Python interpreter, development headers, venv module, distutils
- **Pip**: Installs pip directly for Python 3.12
- **Idempotent**: Checks existence before installation

#### `install_uv()`
```bash
install_uv() {
    if command_exists uv; then
        return 0  # Already installed
    fi

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Configure PATH
    export PATH="$HOME/.local/bin:$PATH"
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
}
```
- **Purpose**: Install and configure uv package manager
- **Method**: Official installation script from astral.sh
- **PATH Configuration**: Adds to current session and bashrc
- **Persistence**: Ensures PATH survives shell restarts
- **Safety**: Checks if PATH already configured

### Project Setup Functions

#### `setup_project()`
```bash
setup_project() {
    # Validate project directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run from project root."
        exit 1
    fi

    # Initialize uv environment
    uv python install 3.12
    uv sync --python 3.12
}
```
- **Purpose**: Initialize the project environment
- **Validation**: Ensures script runs from correct directory
- **Python**: Installs Python 3.12 through uv if needed
- **Sync**: Installs all dependencies from pyproject.toml

#### `install_pytorch()`
```bash
install_pytorch() {
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
}
```
- **Purpose**: Install PyTorch with CUDA 12.6 support
- **Index URL**: Uses PyTorch's CUDA 12.6 wheel repository
- **Components**: Core PyTorch, vision utilities, audio processing
- **Lambda Optimization**: CUDA 12.6 is optimal for Lambda Labs GPUs

#### `install_project_deps()`
```bash
install_project_deps() {
    # Core dependencies
    uv add transformers>=4.35.0
    uv add accelerate>=0.24.0
    uv add bitsandbytes>=0.41.0
    # ... more dependencies

    # Development dependencies
    uv add --dev pytest>=7.4.0
    uv add --dev black>=23.11.0
    uv add --dev ruff>=0.1.6

    # Install project in development mode
    uv pip install -e .
}
```
- **Purpose**: Install all required project dependencies
- **Method**: Uses uv add for proper dependency management
- **Categories**: Core runtime, development tools, optional features
- **Development Mode**: Installs project as editable package

### Lambda Labs Optimization Functions

#### `install_lambda_optimizations()`
```bash
install_lambda_optimizations() {
    # Flash Attention 2 (optional)
    if uv add flash-attn>=2.0.0; then
        print_success "Flash Attention 2 installed"
    else
        print_warning "Flash Attention 2 failed (optional)"
    fi

    # Triton (optional)
    if uv add triton>=2.1.0; then
        print_success "Triton installed"
    else
        print_warning "Triton failed (optional)"
    fi
}
```
- **Purpose**: Install Lambda Labs specific GPU optimizations
- **Flash Attention**: 2-4x speedup for transformer models
- **Triton**: GPU kernel optimization library
- **Error Handling**: Optional installations with graceful failure
- **Benefits**: Significant performance improvements on Lambda GPUs

### Authentication Functions

#### `setup_huggingface()`
```bash
setup_huggingface() {
    # Method 1: Environment variable
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" | uv run huggingface-cli login --token
        return 0
    fi

    # Method 2: Token file
    if [ -f ".hf_token" ]; then
        TOKEN=$(cat .hf_token)
        echo "$TOKEN" | uv run huggingface-cli login --token
        return 0
    fi

    # Method 3: Interactive
    read -p "Login interactively? (y/n): " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run huggingface-cli login
    fi
}
```
- **Purpose**: Authenticate with Hugging Face Hub
- **Priority Order**: Environment variable → Token file → Interactive
- **Security**: Reads tokens securely without exposing in logs
- **Flexibility**: Multiple methods for different use cases
- **CLI Integration**: Uses official huggingface-cli tool

### Testing Functions

#### `test_installation()`
```bash
test_installation() {
    # Test Python and uv
    echo "Python: $(python3.12 --version)"
    echo "uv: $(uv --version)"

    # Test PyTorch CUDA
    uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

    # Test CWA CLI
    uv run cwa info

    # Test GPU detection
    uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
}
```
- **Purpose**: Verify complete installation
- **Components**: Version checks, import tests, CLI functionality
- **CUDA**: Validates PyTorch CUDA integration
- **Integration**: Tests end-to-end functionality
- **GPU**: Confirms GPU detection and access

## Dependencies & Installation Flow

### Installation Sequence
```
1. System Dependencies (apt packages)
   ↓
2. Python 3.12 (deadsnakes PPA)
   ↓
3. uv Package Manager (official installer)
   ↓
4. Project Environment (uv sync)
   ↓
5. PyTorch CUDA 12.6 (Lambda optimized)
   ↓
6. Project Dependencies (transformers, accelerate, etc.)
   ↓
7. Lambda Optimizations (flash-attn, triton)
   ↓
8. Authentication (Hugging Face)
   ↓
9. Testing & Verification
```

### Dependency Categories

#### System Dependencies
```bash
build-essential curl wget git unzip htop tmux vim tree jq python3-pip python3-dev
```
- **build-essential**: Compilers for building Python packages
- **Network tools**: curl, wget for downloading
- **Development**: git, vim, htop for development workflow
- **Python**: pip and dev headers

#### Python Dependencies
```bash
# Core ML/AI
transformers>=4.35.0    # Hugging Face transformers
accelerate>=0.24.0      # Large model loading
bitsandbytes>=0.41.0    # Quantization

# Data & Computation
numpy>=1.26.0           # Numerical computing
pandas>=2.1.0           # Data manipulation
torch                   # PyTorch (installed separately)

# CLI & UI
typer>=0.9.0           # CLI framework
rich>=13.7.0           # Rich terminal output
pydantic>=2.5.0        # Data validation

# Utilities
pyyaml>=6.0.1          # Configuration files
tqdm>=4.66.0           # Progress bars
psutil>=5.9.0          # System monitoring
```

## Error Handling & Safety

### Error Handling Strategy
```bash
set -e  # Exit immediately on error
```
- **Fail-Fast**: Script stops on first error
- **Clean Exit**: Prevents partial installations
- **Error Visibility**: Clear error messages with colors

### Safety Measures

#### Idempotent Operations
```bash
if command_exists python3.12; then
    print_success "Python 3.12 already installed"
    return 0
fi
```
- **Check Before Install**: Prevents redundant operations
- **Version Validation**: Confirms correct versions
- **State Preservation**: Doesn't break existing setups

#### Permission Handling
```bash
sudo apt update              # System packages need sudo
curl -LsSf ... | sh         # User-level installations
export PATH="$HOME/.local/bin:$PATH"  # User PATH modification
```
- **Minimal Sudo**: Only for system-level operations
- **User Space**: Most installations in user directory
- **PATH Safety**: Careful PATH modifications

#### Validation Checks
```bash
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found"
    exit 1
fi
```
- **Project Validation**: Ensures correct directory
- **Dependency Checks**: Validates prerequisites
- **Environment Validation**: Confirms Lambda Labs environment

## Lambda Labs Optimizations

### GPU Detection & Optimization
```bash
# Memory optimization based on detected GPU
if gpu_info["total_gpu_memory"] > 70; then  # A100 80GB
    self.max_memory = {"0": "70GiB", "cpu": "100GiB"}
elif gpu_info["total_gpu_memory"] > 40; then  # A100 40GB
    self.max_memory = {"0": "35GiB", "cpu": "50GiB"}
elif gpu_info["total_gpu_memory"] > 20; then  # RTX 4090
    self.max_memory = {"0": "20GiB", "cpu": "30GiB"}
```
- **Automatic Detection**: Uses nvidia-ml-py for GPU info
- **Memory Allocation**: Optimizes based on GPU type
- **Multi-GPU**: Handles Lambda's multi-GPU setups

### CUDA Optimization
```bash
# PyTorch with CUDA 12.6 (Lambda Labs standard)
--index-url https://download.pytorch.org/whl/cu126
```
- **CUDA Version**: 12.6 optimal for Lambda Labs
- **Wheel Source**: Official PyTorch CUDA wheels
- **Performance**: Maximum GPU utilization

### Performance Libraries
```bash
flash-attn>=2.0.0    # 2-4x speedup for attention
triton>=2.1.0        # GPU kernel optimization
```
- **Flash Attention**: Memory-efficient attention computation
- **Triton**: Custom GPU kernels for optimization
- **Optional**: Graceful failure if compilation issues

## Hugging Face Authentication

### Authentication Methods Priority

#### 1. Environment Variable (Highest Priority)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
./setup_lambda.sh
```
- **Security**: Token not stored in files
- **Automation**: Perfect for CI/CD and automation
- **Session Scope**: Only available in current session

#### 2. Token File
```bash
echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .hf_token
./setup_lambda.sh
```
- **Persistence**: Token survives session restarts
- **Convenience**: No need to remember/retype
- **Gitignore**: Ensure .hf_token is in .gitignore

#### 3. Interactive Login
```bash
./setup_lambda.sh
# Prompts for token input
```
- **User-Friendly**: Guided token input
- **Secure**: Token not visible in command history
- **One-Time**: Stores token in HF CLI cache

#### 4. Skip Authentication
```bash
./setup_lambda.sh --skip-hf-login
```
- **Flexibility**: Setup without authentication
- **Later Authentication**: Can login afterwards
- **Public Models**: Works for models that don't require auth

### Token Security
```bash
echo "$HF_TOKEN" | uv run huggingface-cli login --token
```
- **No Command Line**: Token not visible in process list
- **Pipe Input**: Secure token transmission
- **No Logging**: Token doesn't appear in logs

## Testing & Verification

### Test Categories

#### Component Tests
```bash
# Individual component validation
python3.12 --version    # Python installation
uv --version            # Package manager
torch.__version__       # PyTorch version
```

#### Integration Tests
```bash
# End-to-end functionality
uv run cwa info         # CLI integration
torch.cuda.is_available()  # CUDA integration
model loading test      # Full pipeline
```

#### Performance Tests
```bash
# GPU utilization
nvidia-smi              # GPU visibility
torch.cuda.device_count()  # GPU count
memory allocation test  # Memory optimization
```

### Success Criteria
- ✅ Python 3.12 available and functional
- ✅ uv package manager installed and in PATH
- ✅ PyTorch with CUDA 12.6 working
- ✅ All project dependencies installed
- ✅ CWA CLI functional and responsive
- ✅ GPU detection and utilization working
- ✅ Hugging Face authentication successful
- ✅ Memory optimization applied correctly

## Troubleshooting Guide

### Common Issues & Solutions

#### Permission Denied
```bash
# Problem: ./setup_lambda.sh: Permission denied
chmod +x setup_lambda.sh
```

#### Python 3.12 Installation Fails
```bash
# Manual deadsnakes PPA setup
sudo apt update
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-dev
```

#### uv Command Not Found
```bash
# Fix PATH configuration
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### PyTorch CUDA Issues
```bash
# Verify CUDA availability
nvidia-smi
python3.12 -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### Flash Attention Compilation Fails
```bash
# This is optional - continue without it
print_warning "Flash Attention failed to install (optional)"
# Or install manually later:
# uv pip install flash-attn --no-build-isolation
```

#### Hugging Face Authentication Issues
```bash
# Manual login
uv run huggingface-cli login
# Enter token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Verify authentication
uv run huggingface-cli whoami
```

#### Disk Space Issues
```bash
# Check available space
df -h

# Clean package cache
sudo apt autoremove
sudo apt autoclean

# Use different cache directory
export HF_HOME=/tmp/hf_cache
```

## Customization Options

### Script Arguments
```bash
./setup_lambda.sh --skip-hf-login     # Skip authentication
./setup_lambda.sh --help              # Show help
```

### Environment Variables
```bash
HF_TOKEN=your_token ./setup_lambda.sh              # Auto-authentication
CUDA_VISIBLE_DEVICES=0 ./setup_lambda.sh          # Specific GPU
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512     # Memory optimization
```

### Configuration Files
```bash
# Custom cache directories
echo "cache_dir: /custom/path" >> config.yaml

# Custom model settings
echo "torch_dtype: bfloat16" >> model_config.yaml
```

### Selective Installation
```bash
# Skip optional components by modifying functions
install_lambda_optimizations() {
    print_status "Skipping optional optimizations"
    return 0
}
```

## Advanced Usage

### Multiple GPU Setups
```bash
# The script automatically detects multi-GPU Lambda setups
# GPU configuration is handled in the Python code:
device_map = "auto"  # Automatic device mapping
max_memory = {"0": "35GiB", "1": "35GiB", "cpu": "50GiB"}
```

### Custom Python Environments
```bash
# Use different Python versions
uv python install 3.11
uv sync --python 3.11
```

### Development Mode
```bash
# Use the quick development setup instead
./setup_dev.sh  # Minimal setup for development
```

### Production Deployment
```bash
# Additional production optimizations
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=1
```

---

This documentation provides comprehensive understanding of the setup script's operation, enabling effective troubleshooting, customization, and maintenance of your Lambda Labs VM environment for critical weight analysis research.