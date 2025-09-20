#!/bin/bash

# Lambda Labs VM Useful Aliases for Critical Weight Analysis
# Source this file: source lambda_aliases.sh
# Or add to ~/.bashrc: echo "source $(pwd)/lambda_aliases.sh" >> ~/.bashrc

# Core CWA commands
alias cwa='uv run cwa'
alias python='uv run python'
alias pip='uv pip'
alias pytest='uv run pytest'

# Quick CWA commands
alias cwa-info='uv run cwa info'
alias cwa-models='uv run cwa list-models'
alias cwa-metrics='uv run cwa list-metrics'

# GPU monitoring
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits'

# Environment shortcuts
alias ll='ls -la'
alias la='ls -la'
alias tree='tree -I "__pycache__|*.pyc|.pytest_cache|.git"'

# Project shortcuts
alias cdoutputs='cd outputs && ls -la'
alias cdexperiments='cd experiments && ls -la'
alias logs='find outputs -name "*.log" -exec tail -f {} +'

# Quick model tests
alias test-small='uv run cwa create-config --name quick-test --model microsoft/DialoGPT-small --model-size small && uv run cwa run quick-test_config.yaml'
alias test-gpt2='uv run cwa create-config --name gpt2-test --model gpt2 --model-size small && uv run cwa run gpt2-test_config.yaml'

# Hugging Face shortcuts
alias hf-login='uv run huggingface-cli login'
alias hf-whoami='uv run huggingface-cli whoami'

# Development shortcuts
alias format='black src/ && ruff check src/ --fix'
alias test='uv run pytest tests/ -v'
alias test-gpu='uv run pytest tests/test_basic_functionality.py::test_lambda_model_loading -v'

# Lambda Labs specific
alias lambda-info='echo "=== Lambda Labs VM Info ===" && hostname && whoami && pwd && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && python --version && uv --version'

# Environment
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false

# Functions
cwa-quick() {
    local model_name=${1:-"microsoft/DialoGPT-small"}
    local config_name="quick-$(date +%s)"
    echo "Creating quick config for: $model_name"
    uv run cwa create-config --name "$config_name" --model "$model_name" --model-size small
    echo "Running analysis..."
    uv run cwa run "${config_name}_config.yaml"
}

cwa-medium() {
    local model_name=${1:-"mistralai/Mistral-7B-v0.1"}
    local config_name="medium-$(date +%s)"
    echo "Creating medium model config for: $model_name"
    uv run cwa create-config --name "$config_name" --model "$model_name" --model-size medium --quantization 8bit
    echo "Running analysis..."
    uv run cwa run "${config_name}_config.yaml"
}

# Print useful information when sourced
echo "🚀 Lambda Labs CWA aliases loaded!"
echo ""
echo "Quick commands:"
echo "  cwa-info         # System information"
echo "  cwa-models       # List available models"
echo "  gpu              # GPU status"
echo "  test-small       # Quick small model test"
echo "  cwa-quick [model] # Quick analysis with any model"
echo "  lambda-info      # Full Lambda Labs VM info"
echo ""
echo "Type 'alias' to see all available shortcuts"