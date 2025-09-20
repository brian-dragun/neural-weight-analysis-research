#!/bin/bash

# Critical Weight Analysis - Lambda Labs VM Setup Script
# Automated setup for Python 3.12, uv, PyTorch with CUDA 12.6, and Hugging Face authentication

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get Lambda Labs GPU info
check_lambda_gpu() {
    print_status "Checking Lambda Labs GPU configuration..."

    if command_exists nvidia-smi; then
        echo -e "${GREEN}GPU Information:${NC}"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        echo ""
    else
        print_warning "nvidia-smi not found. GPU may not be available."
    fi
}

# Function to install Python 3.12 using uv
install_python312() {
    print_status "Installing Python 3.12 via uv..."

    # uv will install Python 3.12 automatically when we sync
    # Check if already available
    if uv python list | grep -q "3.12"; then
        print_success "Python 3.12 already available via uv"
        return 0
    fi

    # Install Python 3.12 via uv
    uv python install 3.12

    print_success "Python 3.12 installed successfully via uv"
}

# Function to install uv
install_uv() {
    print_status "Installing uv package manager..."

    if command_exists uv; then
        UV_VERSION=$(uv --version)
        print_success "uv already installed: $UV_VERSION"
        return 0
    fi

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Add uv to bashrc for future sessions
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi

    UV_VERSION=$(uv --version)
    print_success "uv installed successfully: $UV_VERSION"
}

# Function to install minimal system dependencies
install_system_deps() {
    print_status "Installing minimal system dependencies..."

    sudo apt update
    sudo apt install -y \
        build-essential \
        curl \
        git \
        htop \
        tmux \
        tree

    print_success "System dependencies installed"
}

# Function to setup project
setup_project() {
    print_status "Setting up Critical Weight Analysis project..."

    # Ensure we're in the project directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "pyproject.toml not found. Please run this script from the project root directory."
        exit 1
    fi

    # Initialize uv project with Python 3.12
    print_status "Initializing uv environment with Python 3.12..."
    uv sync --python 3.12

    print_success "Project environment initialized"
}

# Function to install PyTorch with CUDA 12.6
install_pytorch() {
    print_status "Installing PyTorch with CUDA 12.6 support for Lambda Labs..."

    # Install PyTorch with CUDA 12.6 (Lambda Labs optimized)
    uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    print_success "PyTorch with CUDA 12.6 installed"
}

# Function to install additional project dependencies
install_project_deps() {
    print_status "Installing additional project dependencies..."

    # Dependencies are already installed via uv sync
    # Install optional extras if they exist

    # Install development and optional dependencies
    print_status "Installing development dependencies..."
    if uv sync --extra dev; then
        print_success "Development dependencies installed"
    else
        print_warning "Some development dependencies failed"
    fi

    print_status "Installing quantization support..."
    if uv sync --extra quantization; then
        print_success "Quantization support installed"
    else
        print_warning "Quantization support failed (optional)"
    fi

    # Install the project in development mode (editable)
    print_status "Installing project in development mode..."
    uv pip install -e .

    print_success "Project dependencies installed"
}

# Function to install optional Lambda Labs optimizations
install_lambda_optimizations() {
    print_status "Installing Lambda Labs GPU optimizations..."

    # Install Lambda Labs optimization extras defined in pyproject.toml
    print_status "Installing lambda-optimized extras..."
    if uv sync --extra lambda-optimized; then
        print_success "Lambda Labs optimizations installed"
    else
        print_warning "Some Lambda Labs optimizations failed to install (optional)"
    fi

    # Try individual packages if extras fail
    if ! uv run python -c "import flash_attn" 2>/dev/null; then
        print_status "Trying to install Flash Attention individually..."
        if uv add flash-attn>=2.0.0; then
            print_success "Flash Attention 2 installed"
        else
            print_warning "Flash Attention 2 failed to install (optional)"
        fi
    fi

    if ! uv run python -c "import triton" 2>/dev/null; then
        print_status "Trying to install Triton individually..."
        if uv add triton>=2.1.0; then
            print_success "Triton installed"
        else
            print_warning "Triton failed to install (optional)"
        fi
    fi
}

# Function to setup Hugging Face authentication
setup_huggingface() {
    print_status "Setting up Hugging Face authentication..."

    # Check if HF_TOKEN is provided as environment variable
    if [ -n "$HF_TOKEN" ]; then
        print_status "Using HF_TOKEN from environment variable"
        echo "$HF_TOKEN" | uv run huggingface-cli login --token
        print_success "Logged in to Hugging Face using environment token"
        return 0
    fi

    # Check if token file exists
    if [ -f ".hf_token" ]; then
        print_status "Using token from .hf_token file"
        TOKEN=$(cat .hf_token)
        echo "$TOKEN" | uv run huggingface-cli login --token
        print_success "Logged in to Hugging Face using token file"
        return 0
    fi

    # Interactive login
    print_status "Please log in to Hugging Face..."
    echo "You can:"
    echo "1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here"
    echo "2. Create .hf_token file with your token"
    echo "3. Login interactively now"
    echo ""

    read -p "Do you want to login interactively? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uv run huggingface-cli login
        print_success "Hugging Face authentication completed"
    else
        print_warning "Skipping Hugging Face authentication. You can login later with: uv run huggingface-cli login"
    fi
}

# Function to create useful directories
create_directories() {
    print_status "Creating useful directories..."

    mkdir -p outputs
    mkdir -p experiments
    mkdir -p models_cache
    mkdir -p /tmp/hf_cache

    print_success "Directories created"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."

    # Test Python and uv
    echo "Python version: $(python3.12 --version)"
    echo "uv version: $(uv --version)"

    # Test PyTorch CUDA
    if uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available')"; then
        print_success "PyTorch CUDA test passed"
    else
        print_error "PyTorch CUDA test failed"
    fi

    # Test CWA CLI
    if uv run cwa info; then
        print_success "CWA CLI test passed"
    else
        print_error "CWA CLI test failed"
    fi

    # Test GPU access
    if uv run python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"; then
        print_success "GPU detection test passed"
    else
        print_warning "GPU detection test failed (GPU may not be available)"
    fi
}

# Function to display final instructions
show_final_instructions() {
    echo ""
    echo -e "${GREEN}🎉 Lambda Labs VM Setup Complete!${NC}"
    echo ""
    echo -e "${BLUE}Quick Start Commands:${NC}"
    echo "  uv run cwa info                                    # Check system status"
    echo "  uv run cwa list-models                            # List available models"
    echo "  uv run cwa create-config --name test --model gpt2  # Create test config"
    echo "  uv run cwa run test_config.yaml                   # Run analysis"
    echo ""
    echo -e "${BLUE}Useful Aliases (add to ~/.bashrc):${NC}"
    echo "  alias cwa='uv run cwa'"
    echo "  alias python='uv run python'"
    echo "  alias pip='uv pip'"
    echo ""
    echo -e "${BLUE}Environment Variables:${NC}"
    echo "  export HF_TOKEN=your_token_here                   # Hugging Face token"
    echo "  export CUDA_VISIBLE_DEVICES=0                     # Specify GPU"
    echo ""
    echo -e "${BLUE}Cache Directories:${NC}"
    echo "  /tmp/hf_cache/          # Hugging Face models cache"
    echo "  outputs/                # Experiment outputs"
    echo "  experiments/            # Custom experiments"
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Run: uv run cwa info"
    echo "  2. Test with small model: uv run cwa create-config --name test --model microsoft/DialoGPT-small"
    echo "  3. Run analysis: uv run cwa run test_config.yaml"
    echo ""
}

# Main setup function
main() {
    echo -e "${GREEN}🚀 Critical Weight Analysis - Lambda Labs VM Setup${NC}"
    echo -e "${BLUE}Setting up Python 3.12, uv, PyTorch with CUDA 12.6, and CWA tools${NC}"
    echo ""

    # Check Lambda Labs environment
    check_lambda_gpu

    # Install components in optimal order for uv
    install_system_deps
    install_uv
    install_python312
    setup_project           # This handles most dependencies via uv sync
    install_pytorch
    install_project_deps    # Additional optional dependencies
    install_lambda_optimizations
    create_directories
    setup_huggingface

    # Test everything
    test_installation

    # Show final instructions
    show_final_instructions
}

# Parse command line arguments
SKIP_HF_LOGIN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-hf-login)
            SKIP_HF_LOGIN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-hf-login    Skip Hugging Face authentication setup"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  HF_TOKEN          Hugging Face token for automatic login"
            echo ""
            echo "Examples:"
            echo "  $0                              # Full setup with interactive HF login"
            echo "  HF_TOKEN=hf_xxx $0             # Setup with automatic HF login"
            echo "  $0 --skip-hf-login             # Setup without HF authentication"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Skip HF login if requested
if [ "$SKIP_HF_LOGIN" = true ]; then
    setup_huggingface() {
        print_status "Skipping Hugging Face authentication (--skip-hf-login)"
    }
fi

# Run main setup
main

print_success "Lambda Labs VM setup completed successfully! 🎉"