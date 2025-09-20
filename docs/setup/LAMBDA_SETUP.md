# Lambda Labs VM Quick Setup Guide

## 🚀 One-Command Setup

```bash
# Clone repository and setup everything
git clone <your-repo-url> critical-weight-analysis-v2
cd critical-weight-analysis-v2
./setup_lambda.sh
```

## 🔐 Hugging Face Authentication Options

### Option 1: Environment Variable (Recommended)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
./setup_lambda.sh
```

### Option 2: Token File
```bash
echo "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" > .hf_token
./setup_lambda.sh
```

### Option 3: Interactive Login
```bash
./setup_lambda.sh
# Will prompt for interactive login
```

### Option 4: Skip Authentication (Setup Later)
```bash
./setup_lambda.sh --skip-hf-login
# Login later with: uv run huggingface-cli login
```

## 📋 What the Setup Script Does

1. **System Dependencies**: Installs build tools, git, curl, etc.
2. **Python 3.12**: Installs from deadsnakes PPA if needed
3. **uv Package Manager**: Latest version with PATH configuration
4. **PyTorch CUDA 12.6**: Lambda Labs optimized installation
5. **Project Dependencies**: All required packages via uv
6. **Lambda Optimizations**: Flash Attention 2, Triton (optional)
7. **Hugging Face Auth**: Automatic login with your token
8. **Directory Structure**: Creates outputs, cache directories
9. **Testing**: Verifies GPU access and CLI functionality

## ⚡ Quick Test After Setup

```bash
# Check system status
uv run cwa info

# List available models
uv run cwa list-models

# Create and run a test
uv run cwa create-config --name lambda_test --model microsoft/DialoGPT-small
uv run cwa run lambda_test_config.yaml
```

## 🛠️ Manual Steps (if needed)

### If automatic Python 3.12 installation fails:
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-dev python3.12-venv
```

### If PyTorch CUDA installation fails:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### If Hugging Face login fails:
```bash
uv run huggingface-cli login
# Enter your token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 📁 Important Paths

- **HF Cache**: `/tmp/hf_cache/` (fast local storage)
- **Outputs**: `./outputs/` (experiment results)
- **Logs**: `./outputs/*/experiment.log`
- **Models Cache**: `./models_cache/`

## 🎯 Lambda Labs Specific Optimizations

The setup script automatically:
- Detects Lambda GPU types (A100, H100, RTX)
- Configures optimal memory allocation
- Installs CUDA 12.6 optimized PyTorch
- Sets up Flash Attention 2 for supported models
- Configures device mapping for multi-GPU setups

## 🔧 Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Should show your Lambda GPU
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues
```bash
chmod +x setup_lambda.sh
sudo chown -R $USER:$USER ~/.local
```

### uv Command Not Found
```bash
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Flash Attention Installation Fails
This is optional - the tool works without it:
```bash
# Skip flash attention if it fails
uv pip uninstall flash-attn
```

## 📊 Expected Performance

After successful setup on Lambda Labs:

| Model Size | Lambda Instance | Load Time | Analysis Time |
|------------|----------------|-----------|---------------|
| GPT-2 (124M) | Any GPU | <10s | <2min |
| Mistral-7B | A100 40GB | <60s | <10min |
| LLaMA-13B | A100 80GB | <120s | <20min |

## 🎉 Success Indicators

✅ `uv run cwa info` shows your Lambda GPU details
✅ `uv run cwa list-models` displays model recommendations
✅ `nvidia-smi` shows GPU utilization during analysis
✅ PyTorch reports CUDA available and correct version
✅ Hugging Face models download to `/tmp/hf_cache/`

---

**Ready for Research!** Your Lambda Labs VM is now optimized for critical weight analysis with full GPU acceleration and Hugging Face integration.