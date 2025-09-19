# Critical Weight Analysis & Cybersecurity Research Tool

A complete **A→B→C security research pipeline** for analyzing critical weights in **Hugging Face Large Language Models** with advanced cybersecurity capabilities, **optimized for Lambda Labs GPU VMs**.

## 🚀 Complete A→B→C Security Pipeline

This tool implements a comprehensive three-phase cybersecurity research workflow:

- **Phase A**: Critical Weight Discovery & Vulnerability Analysis
- **Phase B**: Attack Simulation on Critical Weights
- **Phase C**: Protection & Defense Implementation

Perfect for PhD-level cybersecurity research on transformer models with Lambda Labs GPU acceleration.

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

# Verify Lambda Labs GPU setup
cwa info  # Should show your Lambda GPU details
```

## 📊 Three-Phase Research Methodology

### Phase A: Critical Weight Discovery 🔍

**Goal**: Identify "super weights" most vulnerable to attacks through security-aware sensitivity analysis.

**Available Metrics**:
- `security_gradient` - Security-focused gradient sensitivity
- `vulnerability_scanner` - Deep vulnerability scanning
- `grad_x_weight` - Gradient × weight analysis
- `hessian_diag` - Hessian diagonal with fault tolerance
- `fault_hessian` - Multi-fault-type Hessian analysis

**Output**:
- Critical weights ranked by vulnerability score
- Per-layer vulnerability maps
- Attack surface analysis
- Security risk assessment

### Phase B: Attack Simulation ⚔️

**Goal**: Test targeted attacks specifically on critical weights discovered in Phase A.

**Attack Methods**:
- `fgsm` - Fast Gradient Sign Method
- `pgd` - Projected Gradient Descent
- `textfooler` - Semantic text attacks
- `bit_flip` - Hardware bit-flip simulation
- `fault_injection` - Comprehensive fault injection

**Output**:
- Attack success rates per method
- Performance degradation analysis
- Critical failure identification
- Recovery time estimates

### Phase C: Protection & Defense 🛡️

**Goal**: Implement and test defense mechanisms to protect critical weights from successful attacks.

**Protection Methods**:
- `weight_redundancy` - Multiple backup copies with majority voting
- `checksums` - Cryptographic integrity verification
- `adversarial_training` - Noise resistance enhancement
- `input_sanitization` - Real-time input validation
- `real_time_monitoring` - Continuous anomaly detection
- `fault_tolerance` - Error correction codes

**Output**:
- Protection effectiveness scores
- Defense coverage analysis
- Performance overhead metrics
- Overall security score

## 🚀 Quick Start Examples

### Complete A→B→C Pipeline (Recommended)

```bash
# Run complete security analysis in one command
cwa run-complete-pipeline "microsoft/DialoGPT-small"

# Advanced complete pipeline with custom parameters
cwa run-complete-pipeline "mistralai/Mistral-7B-v0.1" \
  --output-dir "mistral_security_analysis" \
  --vulnerability-threshold 0.8 \
  --top-k 1000 \
  --attack-methods "fgsm,pgd,bit_flip,fault_injection" \
  --protection-methods "weight_redundancy,checksums,adversarial_training,input_sanitization" \
  --device "cuda"
```

### Individual Phase Execution

```bash
# Phase A: Discover critical weights
cwa phase-a "gpt2" \
  --output-dir "gpt2_analysis/phase_a" \
  --metric "security_gradient" \
  --vulnerability-threshold 0.8 \
  --top-k 500

# Phase B: Attack the critical weights
cwa phase-b "gpt2" \
  "gpt2_analysis/phase_a/critical_weights.yaml" \
  --output-dir "gpt2_analysis/phase_b" \
  --attack-methods "fgsm,pgd,bit_flip"

# Phase C: Protect and test defenses
cwa phase-c "gpt2" \
  "gpt2_analysis/phase_a/critical_weights.yaml" \
  "gpt2_analysis/phase_b/attack_results.yaml" \
  --output-dir "gpt2_analysis/phase_c" \
  --protection-methods "weight_redundancy,checksums,adversarial_training"
```

## 📋 Lambda Labs VM Recommendations

### Small Models (Any Lambda GPU)
```bash
# Perfect for testing and development
cwa run-complete-pipeline "microsoft/DialoGPT-small"
cwa run-complete-pipeline "gpt2"
cwa run-complete-pipeline "distilgpt2"
```
- **Lambda Instance**: Any GPU (RTX 4090, RTX 6000, etc.)
- **Memory**: < 2GB GPU memory
- **Speed**: Complete pipeline < 10 minutes

### Medium Models (Lambda A100 40GB)
```bash
# Recommended for serious research
cwa run-complete-pipeline "mistralai/Mistral-7B-v0.1" \
  --attack-methods "fgsm,pgd,textfooler,bit_flip,fault_injection"

cwa run-complete-pipeline "meta-llama/Llama-2-7b-hf" \
  --protection-methods "weight_redundancy,checksums,adversarial_training,input_sanitization,real_time_monitoring"
```
- **Lambda Instance**: A100 40GB recommended
- **Memory**: ~20GB GPU memory
- **Speed**: Complete pipeline < 30 minutes

### Large Models (Lambda A100 80GB or Multi-GPU)
```bash
# Advanced research with large models
cwa run-complete-pipeline "meta-llama/Llama-2-13b-hf" \
  --vulnerability-threshold 0.9 \
  --top-k 1000 \
  --attack-methods "fgsm,pgd,textfooler,bit_flip,fault_injection" \
  --protection-methods "weight_redundancy,checksums,adversarial_training,input_sanitization,fault_tolerance"
```
- **Lambda Instance**: A100 80GB or multi-GPU setup
- **Memory**: 40-70GB GPU memory
- **Speed**: Complete pipeline < 60 minutes

## 🧪 Research Workflows

### Comparative Security Analysis
```bash
# Compare security across multiple models
for model in "gpt2" "distilgpt2" "microsoft/DialoGPT-small"; do
  cwa run-complete-pipeline "$model" --output-dir "security_comparison/${model//\//_}"
done
```

### Vulnerability Threshold Analysis
```bash
# Test different vulnerability thresholds
for threshold in 0.6 0.7 0.8 0.9; do
  cwa phase-a "mistralai/Mistral-7B-v0.1" \
    --vulnerability-threshold $threshold \
    --output-dir "threshold_analysis/threshold_${threshold}"
done
```

### Attack Method Effectiveness Study
```bash
# Phase A: Find critical weights once
cwa phase-a "meta-llama/Llama-2-7b-hf" --output-dir "attack_study/phase_a"

# Phase B: Test each attack method separately
for attack in "fgsm" "pgd" "textfooler" "bit_flip" "fault_injection"; do
  cwa phase-b "meta-llama/Llama-2-7b-hf" \
    "attack_study/phase_a/critical_weights.yaml" \
    --attack-methods "$attack" \
    --output-dir "attack_study/phase_b_${attack}"
done
```

### Defense Mechanism Evaluation
```bash
# Test individual protection methods
critical_weights="results/phase_a/critical_weights.yaml"
attack_results="results/phase_b/attack_results.yaml"

for protection in "weight_redundancy" "checksums" "adversarial_training" "input_sanitization"; do
  cwa phase-c "mistralai/Mistral-7B-v0.1" \
    "$critical_weights" \
    "$attack_results" \
    --protection-methods "$protection" \
    --output-dir "defense_study/${protection}"
done
```

## 📁 Output Structure

### Complete Pipeline Output
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

### Individual Phase Outputs

**Phase A Output**:
```yaml
phase: "A"
model: "mistralai/Mistral-7B-v0.1"
critical_weights:
  - ["layers.0.self_attn.q_proj.weight", 1234, 0.95]  # [layer, index, score]
  - ["layers.0.self_attn.k_proj.weight", 5678, 0.92]
vulnerability_map:
  "layers.0.self_attn": 0.89
  "layers.1.mlp": 0.76
attack_surface:
  attack_vectors:
    - type: "attention_disruption"
      severity: "high"
  total_critical_weights: 500
```

**Phase B Output**:
```yaml
phase: "B"
attack_results:
  fgsm:
    success_rate: 0.75
    successful_attacks: 15
  pgd:
    success_rate: 0.82
    successful_attacks: 18
  fault_injection:
    injected_faults: 25
    performance_degradation: 0.34
```

**Phase C Output**:
```yaml
phase: "C"
protection_results:
  protection_coverage: 0.89
  performance_overhead: 0.03
  defense_effectiveness:
    weight_redundancy: 0.92
    checksums: 0.87
test_results:
  overall_security_score: 0.85
  attack_resistance:
    fgsm: 0.91
    pgd: 0.88
```

## 🔧 Advanced Configuration

### Custom Vulnerability Analysis
```bash
cwa phase-a "mistralai/Mistral-7B-v0.1" \
  --metric "grad_x_weight" \
  --vulnerability-threshold 0.85 \
  --top-k 1000 \
  --output-dir "custom_vulnerability"
```

### Comprehensive Attack Testing
```bash
cwa phase-b "meta-llama/Llama-2-7b-hf" \
  "phase_a/critical_weights.yaml" \
  --attack-methods "fgsm,pgd,textfooler,bit_flip,fault_injection" \
  --output-dir "comprehensive_attacks"
```

### Maximum Protection Deployment
```bash
cwa phase-c "mistralai/Mistral-7B-v0.1" \
  "phase_a/critical_weights.yaml" \
  "phase_b/attack_results.yaml" \
  --protection-methods "weight_redundancy,checksums,adversarial_training,input_sanitization,real_time_monitoring,fault_tolerance" \
  --output-dir "maximum_protection"
```

## 🧪 Testing & Validation

```bash
# Run all tests
pytest tests/ -v

# Test individual phases
pytest tests/test_phase_a.py -v
pytest tests/test_phase_b.py -v
pytest tests/test_phase_c.py -v

# Test Lambda Labs GPU functionality
pytest tests/test_lambda_integration.py -v

# Test complete pipeline
pytest tests/test_complete_pipeline.py -v
```

## 📊 Performance Benchmarks on Lambda Labs

### Phase A (Critical Weight Discovery)
- **Small models (GPT-2)**: < 2 minutes
- **Medium models (Mistral-7B)**: < 5 minutes on A100 40GB
- **Large models (LLaMA-13B)**: < 10 minutes on A100 80GB

### Phase B (Attack Simulation)
- **FGSM/PGD attacks**: < 3 minutes per method
- **Fault injection**: < 5 minutes for comprehensive testing
- **TextFooler**: < 8 minutes for semantic attacks

### Phase C (Protection & Defense)
- **Basic protections**: < 2 minutes implementation
- **Advanced monitoring**: < 5 minutes setup
- **Comprehensive testing**: < 10 minutes validation

### Complete Pipeline
- **Small models**: < 10 minutes total
- **Medium models**: < 30 minutes on A100 40GB
- **Large models**: < 60 minutes on A100 80GB

## 🔬 PhD-Level Research Features

### Super Weight Discovery & Validation

This tool implements cutting-edge research methodologies for discovering and validating "super weights" in transformer models - critical parameters that cause 100× perplexity increases when modified.

#### **1. Critical Weight Extraction**

```bash
# Discover new super weights using PhD-level methodology
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.7 \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-format "csv,json,plots"
```

**Research Methodology**:
- **Activation magnitude monitoring** (>1e3 threshold) in mlp.down_proj layers
- **Hessian-based sensitivity scoring** with FKeras integration
- **Focus on early layers (0-3)** based on transformer robustness research
- **Extract top 0.001%** of weights by sensitivity
- **Excludes known super weights** to discover novel critical parameters

**Output Formats**:
- `research_data.json` - Complete analysis with metadata
- `discovered_weights.csv` - Critical weight coordinates and scores
- `statistics.csv` - Statistical analysis metrics
- `research_summary.md` - Publication-ready summary
- Visualization plots (sensitivity distributions, layer analysis, heatmaps)

#### **2. Super Weight Validation**

```bash
# Validate known super weight coordinates using 100× perplexity methodology
cwa validate-super-weights "gpt2" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --perplexity-threshold 100 \
  --export-results
```

**Known Super Weight Coordinates**:
- **Llama-7B**: `[(2, 'mlp.down_proj', [3968, 7003])]`
- **Mistral-7B**: `[(1, 'mlp.down_proj', [2070, 7310])]`
- **Custom coordinates**: Support for any model architecture

**Validation Features**:
- **100× perplexity increase validation** using cross-entropy loss
- **Bit-level criticality analysis** (MSB→LSB ranking)
- **Multiple perturbation scales** (0.1× to 10× modifications)
- **Statistical significance testing** with baseline comparisons

#### **3. Specialized Research Extraction**

```bash
# Focus on specific research areas with publication-ready outputs
cwa research-extract "gpt2" \
  --focus "mlp-components" \
  --export-format "publication-ready" \
  --analysis-types "statistical,behavioral,architectural"
```

**Research Focus Areas**:
- `attention-mechanisms`: Attention layer analysis
- `mlp-components`: MLP layer focus (super weight specialty)
- `early-layers`: Layers 0-3 analysis
- `layer-norms`: Normalization component analysis
- `comprehensive`: Full model analysis

**Export Formats**:
- `research-csv`: Raw data for analysis
- `publication-ready`: Tables formatted for papers
- `replication-data`: Complete replication package

### Research Analysis Types

#### **Statistical Analysis**
- **Sensitivity distributions**: Mean, std, percentiles (25th, 50th, 75th, 95th, 99th)
- **Layer distributions**: Weight counts per layer, early layer bias calculation
- **Correlations**: Sensitivity vs layer position, component relationships

#### **Behavioral Analysis**
- **Perplexity impact testing**: 10× threshold for significance
- **Task performance changes**: Model capability degradation
- **Recovery analysis**: Restoration effectiveness testing

#### **Architectural Analysis**
- **Component vulnerability**: MLP vs attention susceptibility
- **Layer-wise patterns**: Early layer concentration analysis
- **Hotspot identification**: High-density vulnerability regions

### Publication-Ready Outputs

#### **Research Tables**
```bash
# Generate publication tables automatically
cwa research-extract "gpt2" --export-format "publication-ready"
```

**Generated Tables**:
- `table1_discovery_statistics.csv`: Discovery metrics and rates
- `table2_layer_distribution.csv`: Layer-wise vulnerability analysis
- `table3_top_critical_weights.csv`: Top 20 critical weights with coordinates

#### **Replication Package**
```bash
# Generate complete replication package
cwa research-extract "gpt2" --export-format "replication-data"
```

**Replication Contents**:
- `replication_package.json`: Complete methodology and parameters
- `discovered_coordinates.txt`: Exact coordinates for validation
- `reproduction_commands.sh`: Step-by-step reproduction commands

### Advanced Research Workflows

#### **Comparative Model Analysis**
```bash
# Compare super weight patterns across models
for model in "gpt2" "microsoft/DialoGPT-small" "distilgpt2"; do
  cwa extract-critical-weights "$model" \
    --mode "super_weight_discovery" \
    --output-dir "comparative_analysis/${model//\//_}"
done
```

#### **Threshold Sensitivity Study**
```bash
# Analyze discovery rates across sensitivity thresholds
for threshold in 0.5 0.6 0.7 0.8 0.9; do
  cwa extract-critical-weights "gpt2" \
    --sensitivity-threshold "$threshold" \
    --output-dir "threshold_study/threshold_${threshold}"
done
```

#### **Layer Focus Comparison**
```bash
# Compare early vs late layer vulnerabilities
for focus in "early" "middle" "late"; do
  cwa extract-critical-weights "gpt2" \
    --layer-focus "$focus" \
    --output-dir "layer_study/${focus}_layers"
done
```

#### **Known Super Weight Validation Study**
```bash
# Validate literature-reported super weights
cwa validate-super-weights "llama" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --output-dir "validation_llama"

cwa validate-super-weights "mistral" \
  --coordinates "[(1, 'mlp.down_proj', [2070, 7310])]" \
  --output-dir "validation_mistral"
```

### Research Output Structure

#### **Critical Weight Extraction Output**
```
research_output/
├── research_data.json                    # Complete analysis results
├── discovered_weights.csv               # Weight coordinates and scores
├── statistics.csv                       # Statistical metrics
├── research_summary.md                  # Publication summary
├── visualizations/
│   ├── sensitivity_distribution.png     # Score distribution plots
│   ├── layer_distribution.png          # Layer-wise analysis
│   ├── sensitivity_vs_layer.png        # Correlation plots
│   └── coordinate_heatmap.png           # Weight position heatmap
└── logs/
    └── extraction.log                   # Detailed analysis logs
```

#### **Validation Output**
```
validation_output/
├── validation_report.json              # Complete validation results
├── validation_results.csv              # Coordinate validation table
├── bit_analysis/
│   ├── bit_criticality_L2_mlp.json    # Bit-level impact analysis
│   └── msb_lsb_ranking.csv             # Bit importance ranking
└── perplexity_analysis/
    ├── perturbation_results.csv        # Scale vs impact analysis
    └── baseline_comparison.csv         # Statistical significance
```

#### **Research Extraction Output**
```
research_extract_output/
├── table1_discovery_statistics.csv     # Publication Table 1
├── table2_layer_distribution.csv       # Publication Table 2
├── table3_top_critical_weights.csv     # Publication Table 3
├── replication_package.json            # Complete methodology
├── discovered_coordinates.txt          # Coordinates for validation
└── reproduction_commands.sh            # Replication script
```

## 🎯 Research Applications

### Super Weight Research
- **Novel Discovery**: Identify previously unknown super weights in any model
- **Cross-Architecture Analysis**: Compare vulnerability patterns across models
- **Bit-Level Investigation**: Understand the criticality of individual bits
- **Perplexity Impact Studies**: Quantify the relationship between weights and model performance

### Cybersecurity Research
- **Vulnerability Discovery**: Systematic identification of model weak points
- **Attack Effectiveness**: Quantitative analysis of different attack methods
- **Defense Validation**: Measurable protection effectiveness
- **Threat Modeling**: Comprehensive security risk assessment

### Academic Publications
- **Critical Weight Analysis**: Novel methods for identifying vulnerable parameters
- **Attack Taxonomy**: Classification of weight-based attacks and their effectiveness
- **Defense Mechanisms**: Protection strategies with performance trade-offs
- **Security Metrics**: Quantitative security assessment frameworks

### Real-world Applications
- **Model Hardening**: Practical security improvements for deployment
- **Fault Tolerance**: Hardware fault resistance in production environments
- **Security Monitoring**: Continuous protection for deployed models
- **Risk Assessment**: Security evaluation for model selection

### Integration with Existing Pipeline

The research features seamlessly integrate with your existing A→B→C pipeline:

#### **Combined Research & Security Analysis**
```bash
# Step 1: Discover super weights using research methodology
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --output-dir "research_analysis"

# Step 2: Run traditional pipeline on discovered weights
cwa run-complete-pipeline "gpt2" \
  --output-dir "security_analysis"

# Step 3: Validate discovered super weights
cwa validate-super-weights "gpt2" \
  --coordinates "$(cat research_analysis/discovered_coordinates.txt)" \
  --output-dir "validation_analysis"
```

#### **Enhanced Phase A with Research Insights**
```bash
# Traditional Phase A
cwa phase-a "gpt2" --metric "security_gradient" --top-k 500

# Research-enhanced discovery
cwa extract-critical-weights "gpt2" --top-k-percent 0.001 --layer-focus "early"

# Combined analysis comparing both approaches
```

## 📋 Complete Command Reference

### Core A→B→C Pipeline
```bash
# Individual phases
cwa phase-a MODEL_NAME [options]                    # Critical weight discovery
cwa phase-b MODEL_NAME WEIGHTS_FILE [options]       # Attack simulation
cwa phase-c MODEL_NAME WEIGHTS_FILE ATTACKS_FILE [options]  # Defense implementation

# Complete pipeline
cwa run-complete-pipeline MODEL_NAME [options]      # Full A→B→C analysis
```

### Research Commands (New)
```bash
# Super weight discovery and analysis
cwa extract-critical-weights MODEL_NAME [options]   # PhD-level weight extraction
cwa validate-super-weights MODEL_NAME COORDINATES [options]  # Validate known super weights
cwa research-extract MODEL_NAME [options]           # Specialized research extraction
```

### Utility Commands
```bash
cwa info                                            # System information
cwa list-models                                     # Available models
cwa list-metrics                                    # Sensitivity metrics
cwa create-config                                   # Configuration setup
```

### Research Command Details

#### **extract-critical-weights**
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

#### **validate-super-weights**
```bash
cwa validate-super-weights MODEL_NAME COORDINATES \
  --perplexity-threshold FLOAT \
  --output-dir PATH \
  --export-results \
  --device {cuda,cpu}
```

#### **research-extract**
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

## 🏗️ Project Architecture

```
critical-weight-analysis-v2/
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
│   ├── research/                      # PhD-Level Research Features (NEW)
│   │   └── super_weight_analyzer.py  # Super weight discovery & validation
│   ├── utils/                        # Utilities and helpers
│   └── cli/                          # Command line interface
├── configs/                          # Configuration templates
├── tests/                           # Comprehensive test suite
└── examples/                        # Research examples and notebooks
```

### Research Module Features

The new `research/` module adds PhD-level capabilities:

- **SuperWeightAnalyzer**: Main research class implementing:
  - Activation magnitude monitoring (>1e3 threshold)
  - Hessian-based sensitivity scoring
  - 100× perplexity validation methodology
  - Bit-level criticality analysis (MSB→LSB ranking)
  - Statistical, behavioral, and architectural analysis
  - Publication-ready output generation
  - Complete replication package creation

- **Known Super Weight Database**: Pre-configured coordinates for:
  - Llama-7B: `[(2, 'mlp.down_proj', [3968, 7003])]`
  - Mistral-7B: `[(1, 'mlp.down_proj', [2070, 7310])]`
  - Extensible to any model architecture

- **Research Workflows**: Specialized analysis modes:
  - Super weight discovery with early layer focus
  - Cross-model vulnerability comparison
  - Publication table generation
  - Replication package creation

## 🔒 Security & Code Quality

```bash
# Code formatting and linting
black src/
ruff check src/

# Security scanning
bandit -r src/
safety check

# Type checking
mypy src/

# Test coverage
pytest tests/ --cov=src --cov-report=html
```

## 🤝 Contributing

This research tool is built with:
- **Clean Architecture**: Clear separation of concerns across phases
- **Extensible Design**: Easy to add new metrics, attacks, and defenses
- **Lambda Labs Optimization**: Full GPU acceleration support
- **Comprehensive Testing**: Validated across multiple model architectures
- **Rich Documentation**: Complete API and usage documentation

## 📄 License

Academic and research use. Please cite appropriately in academic publications.

## 🏆 Citation

If you use this tool in your research, please cite:

```bibtex
@software{critical_weight_analysis,
  title={Critical Weight Analysis \& Cybersecurity Research Tool},
  author={Your Research Team},
  year={2024},
  url={https://github.com/your-repo/critical-weight-analysis-v2},
  note={A→B→C Security Pipeline for Transformer Models}
}
```

---

**Status**: ✅ Complete A→B→C Security Pipeline Implementation
**Ready for**: Advanced cybersecurity research on transformer models
**Optimized for**: Lambda Labs GPU infrastructure with CUDA 12.6 support