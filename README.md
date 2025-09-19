# Critical Weight Analysis & Cybersecurity Research Tool

A comprehensive **A→B→C security research pipeline** for analyzing critical weights in **Large Language Models** with advanced **PhD-level research capabilities**, optimized for **Lambda Labs GPU infrastructure**.

## ⭐ Key Capabilities

🔍 **Security Analysis**: Complete A→B→C pipeline for vulnerability discovery, attack simulation, and defense implementation
🎓 **PhD Research**: Super weight discovery with 100× perplexity validation methodology
⚔️ **Attack Simulation**: FGSM, PGD, bit-flip, fault injection, and advanced adversarial methods
🛡️ **Defense Systems**: Weight redundancy, checksums, real-time monitoring, and fortress protection
📊 **Publication Ready**: Automated table generation, replication packages, and visualization
🚀 **Lambda Optimized**: Full GPU acceleration with CUDA 12.6 support

## 🚀 Quick Start

```bash
# 1. Clone and setup on Lambda Labs VM
git clone <repo-url> critical-weight-analysis
cd critical-weight-analysis
source .venv/bin/activate

# 2. Verify GPU setup
cwa info

# 3. Run complete security analysis (fastest start)
cwa run-complete-pipeline "microsoft/DialoGPT-small"

# 4. Discover super weights (research mode)
cwa extract-critical-weights "gpt2" --mode "super_weight_discovery"

# 5. Validate known super weights
cwa validate-super-weights "gpt2" --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]"
```

**Results**: Complete analysis in `complete_pipeline_results/` + research data in `research_output/`

## 🧠 Core Concepts

### A→B→C Security Pipeline

**Phase A**: **Critical Weight Discovery**
Identify the most vulnerable weights using security-aware sensitivity analysis.

**Phase B**: **Attack Simulation**
Test targeted attacks specifically on discovered critical weights.

**Phase C**: **Protection & Defense**
Implement and validate defense mechanisms against successful attacks.

### PhD-Level Research Features

**Super Weight Discovery**: Find critical parameters that cause 100× perplexity increases
**Validation Methodology**: Bit-level analysis with literature coordinate validation
**Publication Support**: Automated table generation and replication packages
**Research Analytics**: Statistical, behavioral, and architectural analysis

## 🛠️ Installation

### Lambda Labs Setup (Recommended)

```bash
# Clone repository
git clone <repo-url> critical-weight-analysis
cd critical-weight-analysis

# Install with uv (Python 3.12+)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# IMPORTANT: Install PyTorch with Lambda Labs CUDA 12.6
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install the tool
uv pip install -e .

# Verify Lambda Labs GPU setup
cwa info  # Should show your GPU details
```

### Hardware Requirements

| Model Size | Lambda Instance | GPU Memory | Pipeline Time |
|------------|----------------|------------|---------------|
| **Small** (GPT-2, DialoGPT) | Any GPU | < 2GB | < 10 min |
| **Medium** (Mistral-7B, Llama-7B) | A100 40GB | ~20GB | < 30 min |
| **Large** (Llama-13B+) | A100 80GB | 40-70GB | < 60 min |

## 📋 Command Reference

### Core A→B→C Pipeline

```bash
# Complete pipeline (recommended for most users)
cwa run-complete-pipeline MODEL_NAME [options]

# Individual phases
cwa phase-a MODEL_NAME [options]                    # Critical weight discovery
cwa phase-b MODEL_NAME WEIGHTS_FILE [options]       # Attack simulation
cwa phase-c MODEL_NAME WEIGHTS_FILE ATTACKS_FILE [options]  # Defense implementation
```

### Research Commands

```bash
# Super weight discovery and analysis
cwa extract-critical-weights MODEL_NAME [options]   # PhD-level weight extraction
cwa validate-super-weights MODEL_NAME COORDINATES [options]  # Validate known super weights
cwa research-extract MODEL_NAME [options]           # Specialized research extraction
```

### Utility Commands

```bash
cwa info                # System information
cwa list-models         # Available models
cwa list-metrics        # Sensitivity metrics
cwa create-config       # Configuration setup
```

## 💡 Usage Examples

### Security Analysis Workflows

#### **Basic Security Assessment**
```bash
# Quick security analysis
cwa run-complete-pipeline "gpt2"

# Advanced security analysis
cwa run-complete-pipeline "mistralai/Mistral-7B-v0.1" \
  --attack-methods "fgsm,pgd,bit_flip,fault_injection" \
  --protection-methods "weight_redundancy,layer11_attention_fortress,input_sanitization"
```

#### **Individual Phase Analysis**
```bash
# Phase A: Find critical weights
cwa phase-a "gpt2" \
  --metric "security_gradient" \
  --vulnerability-threshold 0.8 \
  --top-k 500

# Phase B: Test attacks
cwa phase-b "gpt2" "results/critical_weights.yaml" \
  --attack-methods "fgsm,pgd,bit_flip"

# Phase C: Implement defenses
cwa phase-c "gpt2" "results/critical_weights.yaml" "results/attack_results.yaml" \
  --protection-methods "weight_redundancy,checksums,adversarial_training"
```

### Research Workflows

#### **Super Weight Discovery**
```bash
# Discover new super weights
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.7 \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-format "csv,json,plots"
```

#### **Literature Validation**
```bash
# Validate known Llama-7B super weights
cwa validate-super-weights "llama" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --perplexity-threshold 100

# Validate known Mistral-7B super weights
cwa validate-super-weights "mistral" \
  --coordinates "[(1, 'mlp.down_proj', [2070, 7310])]"
```

#### **Publication-Ready Analysis**
```bash
# Generate publication tables
cwa research-extract "gpt2" \
  --focus "mlp-components" \
  --export-format "publication-ready"

# Generate replication package
cwa research-extract "gpt2" \
  --export-format "replication-data"
```

#### **Comparative Studies**
```bash
# Compare multiple models
for model in "gpt2" "distilgpt2" "microsoft/DialoGPT-small"; do
  cwa extract-critical-weights "$model" --output-dir "study/${model//\//_}"
done

# Parameter sensitivity study
for threshold in 0.5 0.6 0.7 0.8 0.9; do
  cwa extract-critical-weights "gpt2" \
    --sensitivity-threshold "$threshold" \
    --output-dir "thresholds/threshold_${threshold}"
done
```

### Combined Research + Security Analysis

```bash
# Step 1: Research discovery
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --output-dir "research_analysis"

# Step 2: Security pipeline
cwa run-complete-pipeline "gpt2" \
  --output-dir "security_analysis"

# Step 3: Validate discoveries
cwa validate-super-weights "gpt2" \
  --coordinates "$(cat research_analysis/discovered_coordinates.txt)" \
  --output-dir "validation_analysis"
```

## 📊 Detailed Command Options

### extract-critical-weights

```bash
cwa extract-critical-weights MODEL_NAME \
  --mode {super_weight_discovery,validation,comprehensive} \
  --sensitivity-threshold FLOAT \
  --top-k-percent FLOAT \
  --layer-focus {early,middle,late,all} \
  --output-format {csv,json,plots} \
  --output-dir PATH \
  --research-mode \
  --device {cuda,cpu}
```

**Research Methodology**:
- **Activation magnitude monitoring** (>1e3 threshold) in mlp.down_proj layers
- **Hessian-based sensitivity scoring** with approximation methods
- **Early layer focus** (layers 0-3) based on transformer robustness research
- **Top 0.001% extraction** for precision research
- **Excludes known super weights** to discover novel parameters

### validate-super-weights

```bash
cwa validate-super-weights MODEL_NAME COORDINATES \
  --perplexity-threshold FLOAT \
  --output-dir PATH \
  --export-results \
  --device {cuda,cpu}
```

**Validation Features**:
- **100× perplexity increase validation** using cross-entropy loss
- **Bit-level criticality analysis** (MSB→LSB ranking)
- **Multiple perturbation scales** (0.1× to 10× modifications)
- **Statistical significance testing** with baseline comparisons

### research-extract

```bash
cwa research-extract MODEL_NAME \
  --focus {attention-mechanisms,mlp-components,early-layers,layer-norms,comprehensive} \
  --threshold FLOAT \
  --export-format {research-csv,publication-ready,replication-data} \
  --include-metadata \
  --analysis-types {statistical,behavioral,architectural} \
  --output-dir PATH \
  --device {cuda,cpu}
```

## 📁 Output Structure

### Complete Pipeline Results
```
complete_pipeline_results/
├── phase_a/
│   └── critical_weights.yaml          # Critical weights and vulnerability analysis
├── phase_b/
│   └── attack_results.yaml            # Attack simulation results
├── phase_c/
│   └── protection_results.yaml        # Defense effectiveness results
└── pipeline_summary.yaml              # Overall pipeline summary
```

### Research Output
```
research_output/
├── research_data.json                 # Complete analysis with metadata
├── discovered_weights.csv             # Critical weight coordinates and scores
├── statistics.csv                     # Statistical analysis metrics
├── research_summary.md                # Publication-ready summary
├── visualizations/
│   ├── sensitivity_distribution.png   # Score distribution plots
│   ├── layer_distribution.png         # Layer-wise analysis
│   ├── sensitivity_vs_layer.png       # Correlation plots
│   └── coordinate_heatmap.png          # Weight position heatmap
└── logs/
    └── extraction.log                 # Detailed analysis logs
```

### Publication Materials
```
publication_data/
├── table1_discovery_statistics.csv    # Publication Table 1
├── table2_layer_distribution.csv      # Publication Table 2
├── table3_top_critical_weights.csv    # Publication Table 3
├── replication_package.json           # Complete methodology
├── discovered_coordinates.txt         # Coordinates for validation
└── reproduction_commands.sh           # Replication script
```

## 🔬 Research Applications

### Super Weight Research
- **Novel Discovery**: Identify previously unknown super weights in any model
- **Cross-Architecture Analysis**: Compare vulnerability patterns across models
- **Bit-Level Investigation**: Understand the criticality of individual bits
- **Perplexity Impact Studies**: Quantify weight-performance relationships

### Cybersecurity Research
- **Vulnerability Discovery**: Systematic identification of model weak points
- **Attack Effectiveness**: Quantitative analysis of attack methods
- **Defense Validation**: Measurable protection effectiveness
- **Threat Modeling**: Comprehensive security risk assessment

### Academic Publications
- **Critical Weight Analysis**: Novel methods for identifying vulnerable parameters
- **Attack Taxonomy**: Classification of weight-based attacks and effectiveness
- **Defense Mechanisms**: Protection strategies with performance trade-offs
- **Security Metrics**: Quantitative security assessment frameworks

## 🏗️ Technical Architecture

```
critical-weight-analysis/
├── src/cwa/
│   ├── core/                          # Core abstractions
│   │   ├── interfaces.py              # Data structures and protocols
│   │   ├── config.py                  # Configuration management
│   │   ├── models.py                  # Lambda Labs LLM management
│   │   └── data.py                    # Data handling utilities
│   ├── sensitivity/                   # Phase A: Critical Weight Discovery
│   │   ├── security_analyzer.py       # Main security analysis
│   │   ├── grad_x_weight.py          # Gradient × weight metrics
│   │   ├── hessian_diag.py           # Hessian diagonal analysis
│   │   └── registry.py               # Metric registration
│   ├── security/                      # Phase B & C: Attack & Defense
│   │   ├── adversarial.py            # Adversarial attacks (FGSM, PGD, TextFooler)
│   │   ├── targeted_attacks.py       # Targeted attack strategies
│   │   ├── fault_injection.py        # Hardware fault simulation
│   │   ├── defense_mechanisms.py     # Protection implementations
│   │   └── weight_protection.py      # Critical weight protection
│   ├── research/                      # PhD-Level Research Features
│   │   └── super_weight_analyzer.py  # Super weight discovery & validation
│   ├── utils/                        # Utilities and helpers
│   └── cli/                          # Command line interface
├── configs/                          # Configuration templates
├── tests/                           # Comprehensive test suite
└── examples/                        # Research examples and notebooks
```

### Research Module Features

**SuperWeightAnalyzer**: Main research class implementing:
- Activation magnitude monitoring (>1e3 threshold)
- Hessian-based sensitivity scoring
- 100× perplexity validation methodology
- Bit-level criticality analysis (MSB→LSB ranking)
- Statistical, behavioral, and architectural analysis
- Publication-ready output generation
- Complete replication package creation

**Known Super Weight Database**: Pre-configured coordinates for:
- Llama-7B: `[(2, 'mlp.down_proj', [3968, 7003])]`
- Mistral-7B: `[(1, 'mlp.down_proj', [2070, 7310])]`
- Extensible to any model architecture

## 📈 Performance Benchmarks

### Lambda Labs Performance

| Phase | Small Models (GPT-2) | Medium Models (Mistral-7B) | Large Models (Llama-13B) |
|-------|---------------------|----------------------------|-------------------------|
| **Phase A** | < 2 minutes | < 5 minutes (A100 40GB) | < 10 minutes (A100 80GB) |
| **Phase B** | < 3 minutes/method | < 5 minutes/method | < 8 minutes/method |
| **Phase C** | < 2 minutes | < 5 minutes | < 10 minutes |
| **Research** | < 5 minutes | < 15 minutes | < 30 minutes |
| **Complete** | < 10 minutes | < 30 minutes | < 60 minutes |

### Available Attack Methods
- **FGSM/PGD**: < 3 minutes per method
- **Fault Injection**: < 5 minutes comprehensive testing
- **TextFooler**: < 8 minutes semantic attacks
- **Bit Flip**: < 2 minutes hardware simulation

### Available Protection Methods
- **Weight Redundancy**: Multiple backup copies with majority voting
- **Layer11 Attention Fortress**: Specialized protection for most critical components
- **Checksums**: Cryptographic integrity verification
- **Input Sanitization**: Real-time input validation
- **Adversarial Training**: Noise resistance enhancement
- **Real-time Monitoring**: Continuous anomaly detection

## 🧪 Testing & Validation

```bash
# Run all tests
pytest tests/ -v

# Test individual components
pytest tests/test_phase_a.py -v
pytest tests/test_phase_b.py -v
pytest tests/test_phase_c.py -v
pytest tests/test_research.py -v

# Test Lambda Labs integration
pytest tests/test_lambda_integration.py -v

# Test complete pipeline
pytest tests/test_complete_pipeline.py -v

# Test coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔒 Code Quality & Security

```bash
# Code formatting and linting
black src/
ruff check src/

# Security scanning
bandit -r src/
safety check

# Type checking
mypy src/
```

## 🤝 Contributing

This research tool is built with:
- **Clean Architecture**: Clear separation of concerns across phases
- **Extensible Design**: Easy to add new metrics, attacks, and defenses
- **Lambda Labs Optimization**: Full GPU acceleration support
- **Comprehensive Testing**: Validated across multiple model architectures
- **Rich Documentation**: Complete API and usage documentation

## 🏆 Citation

If you use this tool in your research, please cite:

```bibtex
@software{critical_weight_analysis,
  title={Critical Weight Analysis \& Cybersecurity Research Tool},
  author={Your Research Team},
  year={2024},
  url={https://github.com/your-repo/critical-weight-analysis},
  note={A→B→C Security Pipeline for Transformer Models with Super Weight Discovery}
}
```

## 📄 License

Academic and research use. Please cite appropriately in academic publications.

---

**Status**: ✅ Complete A→B→C Security Pipeline + PhD Research Features
**Ready for**: Advanced cybersecurity and super weight research on transformer models
**Optimized for**: Lambda Labs GPU infrastructure with CUDA 12.6 support