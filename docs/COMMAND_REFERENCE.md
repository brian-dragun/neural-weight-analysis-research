# CWA Command Reference Guide

Complete guide to running Critical Weight Analysis commands individually and in workflows.

## 🚀 Quick Start

```bash
cd ~/nova/neural-weight-analysis-research
source .venv/bin/activate

# Show available commands
cwa --help

# Basic system info
cwa info

# List available models
cwa list-models
```

## 📋 Command Categories

### **Basic Commands** (`cwa basic`)

```bash
# Run basic experiment with config
cwa run config.yaml --output-dir outputs/results/gpt2/basic/

# Create experiment configuration
cwa basic create-config "My Experiment" --model gpt2 --output-path my_config.yaml

# Get model information
cwa basic model-info microsoft/DialoGPT-small

# List available sensitivity metrics
cwa basic list-metrics

# System information
cwa basic info
```

### **Security Analysis** (`cwa security`)

**Phase A: Critical Weight Discovery**
```bash
# Discover critical weights
cwa security phase-a microsoft/DialoGPT-small \
    --output-dir outputs/results/microsoft_DialoGPT-small/phase_a/ \
    --vulnerability-threshold 0.8 \
    --top-k 500

# Alternative: Top-level alias
cwa phase-a microsoft/DialoGPT-small --output-dir outputs/results/microsoft_DialoGPT-small/phase_a/
```

**Phase B: Attack Simulation**
```bash
# Run attack simulation
cwa security phase-b microsoft/DialoGPT-small \
    outputs/results/microsoft_DialoGPT-small/phase_a/critical_weights.yaml \
    --output-dir outputs/results/microsoft_DialoGPT-small/phase_b/ \
    --attack-methods "fgsm,pgd,bit_flip,fault_injection"

# Alternative: Top-level alias
cwa phase-b microsoft/DialoGPT-small \
    outputs/results/microsoft_DialoGPT-small/phase_a/critical_weights.yaml \
    --output-dir outputs/results/microsoft_DialoGPT-small/phase_b/
```

**Phase C: Protection & Defense**
```bash
# Implement protections
cwa security phase-c microsoft/DialoGPT-small \
    outputs/results/microsoft_DialoGPT-small/phase_a/critical_weights.yaml \
    outputs/results/microsoft_DialoGPT-small/phase_b/attack_results.yaml \
    --output-dir outputs/results/microsoft_DialoGPT-small/phase_c/ \
    --protection-methods "weight_redundancy,checksums,adversarial_training"

# Alternative: Top-level alias
cwa phase-c microsoft/DialoGPT-small \
    outputs/results/microsoft_DialoGPT-small/phase_a/critical_weights.yaml \
    outputs/results/microsoft_DialoGPT-small/phase_b/attack_results.yaml \
    --output-dir outputs/results/microsoft_DialoGPT-small/phase_c/
```

**Complete A→B→C Pipeline**
```bash
# Run entire security pipeline
cwa security run-complete-pipeline microsoft/DialoGPT-small \
    --output-dir outputs/results/microsoft_DialoGPT-small/complete_pipeline/ \
    --vulnerability-threshold 0.8 \
    --attack-methods "fgsm,pgd,bit_flip,fault_injection" \
    --protection-methods "weight_redundancy,checksums,adversarial_training"
```

### **Research Commands** (`cwa research`)

**Super Weight Extraction**
```bash
# Extract critical weights for research
cwa research extract-critical-weights gpt2 \
    --mode super_weight_discovery \
    --sensitivity-threshold 0.7 \
    --top-k-percent 0.001 \
    --layer-focus early \
    --output-dir outputs/results/gpt2/research/extraction/

# Alternative: Top-level alias
cwa extract-critical-weights gpt2 --output-dir outputs/results/gpt2/research/
```

**Super Weight Validation**
```bash
# Validate specific coordinates
cwa research validate-super-weights llama-7b \
    --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
    --perplexity-threshold 100 \
    --output-dir outputs/results/llama-7b/research/validation/
```

**Specialized Research Extraction**
```bash
# Focus on attention mechanisms
cwa research research-extract gpt2 \
    --focus attention-mechanisms \
    --export-format research-csv \
    --output-dir outputs/results/gpt2/research/attention/

# Focus on MLP components
cwa research research-extract gpt2 \
    --focus mlp-components \
    --export-format publication-ready \
    --output-dir outputs/results/gpt2/research/mlp/

# Comprehensive analysis
cwa research research-extract gpt2 \
    --focus comprehensive \
    --analysis-types "statistical,behavioral,architectural" \
    --output-dir outputs/results/gpt2/research/comprehensive/
```

**Spectral Analysis**
```bash
# Advanced spectral vulnerability analysis
cwa research spectral-analysis gpt2 \
    --analysis-types "signatures,transitions,stability" \
    --top-k 10 \
    --include-pac-bounds \
    --output-dir outputs/results/gpt2/research/spectral/
```

### **Monitoring Commands** (`cwa monitor`)

```bash
# Real-time security monitoring
cwa monitor monitor-realtime gpt2 \
    --detection-algorithms "statistical,gradient,activation,weight_drift" \
    --latency-target 1.0 \
    --output-dir outputs/results/gpt2/monitoring/realtime/
```

### **Advanced Analysis** (`cwa advanced`)

**Game-Theoretic Analysis**
```bash
# Game theory weight analysis
cwa advanced analyze-game-theory gpt2 \
    --game-types "nash_equilibrium,cooperative,evolutionary" \
    --max-players 50 \
    --output-dir outputs/results/gpt2/analysis/game_theory/
```

**Transfer Learning Analysis**
```bash
# Cross-architecture transfer analysis
cwa advanced analyze-transfer gpt2 microsoft/DialoGPT-small \
    --mapping-strategies "geometric,semantic,interpolation" \
    --similarity-threshold 0.7 \
    --output-dir outputs/results/transfer_gpt2_to_dialogpt/
```

## 🔄 **Complete Workflow Examples**

### **Single Model Complete Analysis**
```bash
# 1. Basic model info
cwa basic model-info gpt2

# 2. Run complete security pipeline
cwa security run-complete-pipeline gpt2 \
    --output-dir outputs/results/gpt2/security_pipeline/

# 3. Research extraction
cwa research extract-critical-weights gpt2 \
    --output-dir outputs/results/gpt2/research/

# 4. Validate discovered weights
cwa research validate-super-weights gpt2 \
    --coordinates "DISCOVERED_COORDINATES" \
    --output-dir outputs/results/gpt2/validation/

# 5. Advanced analysis
cwa advanced analyze-game-theory gpt2 \
    --output-dir outputs/results/gpt2/game_theory/
```

### **Multi-Model Comparison**
```bash
# For each model: gpt2, microsoft/DialoGPT-small, llama-7b
for model in "gpt2" "microsoft/DialoGPT-small" "llama-7b"; do
    model_clean=$(echo $model | tr '/' '_')

    echo "Analyzing $model..."

    # Security analysis
    cwa security run-complete-pipeline $model \
        --output-dir outputs/results/$model_clean/security/

    # Research extraction
    cwa research extract-critical-weights $model \
        --output-dir outputs/results/$model_clean/research/

    # Game theory analysis
    cwa advanced analyze-game-theory $model \
        --output-dir outputs/results/$model_clean/game_theory/
done
```

### **Research-Focused Workflow**
```bash
# PhD research workflow
MODEL="gpt2"

# 1. Comprehensive extraction
cwa research research-extract $MODEL \
    --focus comprehensive \
    --analysis-types "statistical,behavioral,architectural" \
    --output-dir outputs/results/$MODEL/research/comprehensive/

# 2. Attention-specific analysis
cwa research research-extract $MODEL \
    --focus attention-mechanisms \
    --export-format publication-ready \
    --output-dir outputs/results/$MODEL/research/attention/

# 3. MLP-specific analysis
cwa research research-extract $MODEL \
    --focus mlp-components \
    --export-format replication-data \
    --output-dir outputs/results/$MODEL/research/mlp/

# 4. Spectral analysis
cwa research spectral-analysis $MODEL \
    --include-pac-bounds \
    --output-dir outputs/results/$MODEL/research/spectral/
```

## 🎯 **Quick Reference Commands**

```bash
# Most commonly used commands
cwa info                                    # System info
cwa list-models                            # Available models
cwa extract-critical-weights MODEL         # Basic research extraction
cwa phase-a MODEL                          # Security discovery
cwa security run-complete-pipeline MODEL   # Full security analysis
```

## 📊 **Output Organization**

All commands automatically organize results in the `outputs/` directory:

```
outputs/results/MODEL_NAME/
├── phase_a/           # Security discovery results
├── phase_b/           # Attack simulation results
├── phase_c/           # Protection results
├── research/          # Research extraction results
├── monitoring/        # Real-time monitoring logs
└── analysis/          # Advanced analysis results
```

## 🔧 **Common Options**

- `--output-dir`: Specify output directory
- `--device`: Choose device (cuda/cpu)
- `--help`: Show command help
- `--verbose`: Enable verbose logging

## 💡 **Pro Tips**

1. **Always specify output directories** to keep results organized
2. **Use model name cleaning** for file paths: `echo "meta-llama/Llama-2-7b" | tr '/' '_'`
3. **Check system info first**: `cwa info` to verify GPU availability
4. **Start with small models** for testing: `microsoft/DialoGPT-small`
5. **Use complete pipelines** for comprehensive analysis
6. **Save configurations** with `cwa basic create-config` for reproducibility