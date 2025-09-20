#!/bin/bash

# Quick Development Setup Script - uv optimized
# For when you just need the project setup using uv's full capabilities

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_status "🚀 Quick Development Setup for Critical Weight Analysis (uv optimized)"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found. Please run from project root."
    exit 1
fi

# Install uv if not present
if ! command -v uv >/dev/null 2>&1; then
    print_status "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Setup project with Python 3.12 (uv will install Python if needed)
print_status "Setting up project environment with Python 3.12..."
uv sync --python 3.12

# Install PyTorch with CUDA using uv
print_status "Installing PyTorch with CUDA via uv..."
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install optional extras
print_status "Installing optional dependencies..."
if uv sync --extra quantization; then
    print_success "Quantization support installed"
else
    print_warning "Quantization support failed (optional)"
fi

if uv sync --extra lambda-optimized; then
    print_success "Lambda Labs optimizations installed"
else
    print_warning "Lambda Labs optimizations failed (optional)"
fi

# Install project in development mode
print_status "Installing project in development mode..."
uv pip install -e .

# Create useful directories
print_status "Creating directories..."
mkdir -p outputs experiments models_cache /tmp/hf_cache

# Test installation
print_status "Testing installation..."
if uv run cwa info; then
    print_success "✅ Development setup complete!"
else
    print_warning "Setup completed but CLI test failed"
fi

echo ""
echo -e "${GREEN}🎉 Quick commands:${NC}"
echo "  uv run cwa info                    # Check system"
echo "  uv run cwa list-models            # List models"
echo "  uv run cwa create-config --help   # Create config"
echo "  uv run python --version           # Check Python"
echo ""
echo -e "${BLUE}Convenience aliases:${NC}"
echo "  source lambda_aliases.sh          # Load all aliases"
echo "  alias cwa='uv run cwa'"
echo "  alias python='uv run python'"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Set HF_TOKEN: export HF_TOKEN=hf_xxx"
echo "  2. Login to HF: uv run huggingface-cli login"
echo "  3. Test model: uv run cwa create-config --name test --model gpt2"