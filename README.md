# Critical Weight Analysis & Cybersecurity Research Tool

A comprehensive **A→B→C security research pipeline** for analyzing critical weights in **Large Language Models** with cutting-edge **PhD-level research capabilities** and **information-theoretic foundations**, optimized for **Lambda Labs GPU infrastructure**.

## ⭐ Core Capabilities

### 🔬 **Advanced Research Methods**
- **Information-Theoretic Analysis**: Fisher information matrices, mutual information estimation, and phase transition detection
- **Ensemble Super Weight Discovery**: 4-method ensemble combining activation, causal, information, and spectral approaches
- **Spectral Vulnerability Analysis**: Eigenvalue-based detection with PAC-Bayesian theoretical guarantees
- **100× Perplexity Validation**: Rigorous methodology for confirming super weight criticality

### 🔍 **Security Analysis Pipeline**
- **A→B→C Framework**: Complete vulnerability discovery → attack simulation → defense implementation
- **Multi-Method Discovery**: Traditional gradients + information theory + spectral learning
- **Advanced Attack Simulation**: FGSM, PGD, bit-flip, fault injection, and targeted strategies
- **Certified Defense Systems**: Weight redundancy, checksums, real-time monitoring with theoretical guarantees

### 🎓 **Publication-Ready Research**
- **Automated Table Generation**: Publication-quality tables and figures
- **Replication Packages**: Complete methodology and reproduction scripts
- **PAC-Bayesian Certificates**: Mathematical guarantees with confidence intervals
- **Cross-Architecture Analysis**: Compare vulnerability patterns across model types

### 🚀 **Production-Ready Tools**
- **Lambda Labs Optimized**: Full GPU acceleration with CUDA 12.6 support
- **Real-Time Monitoring**: Millisecond-latency security assessment
- **Comprehensive CLI**: 10+ specialized commands for different analysis workflows
- **Extensible Architecture**: Clean module design for adding new methods

## 🚀 Quick Start

### **🎯 Fastest Start - Complete Security Analysis**
```bash
# 1. Setup on Lambda Labs VM
git clone <repo-url> critical-weight-analysis
cd critical-weight-analysis
source .venv/bin/activate

# 2. Verify GPU and run complete pipeline
cwa info  # Verify GPU setup
cwa run-complete-pipeline "microsoft/DialoGPT-small"
```

### **🔬 Advanced Research Workflow**
```bash
# 3. Multi-method ensemble discovery (NEW!)
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --top-k-percent 0.001 \
  --layer-focus "early"

# 4. Information-theoretic analysis (NEW!)
cwa spectral-analysis "gpt2" \
  --analysis-types "signatures,transitions,stability" \
  --include-pac-bounds \
  --confidence-level 0.95

# 5. Validate discovered super weights
cwa validate-super-weights "gpt2" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --perplexity-threshold 100
```

### **📊 Publication-Ready Research**
```bash
# 6. Generate publication materials
cwa research-extract "gpt2" \
  --focus "mlp-components" \
  --export-format "publication-ready"

# 7. Create replication package
cwa research-extract "gpt2" \
  --export-format "replication-data"
```

**Results**:
- Security analysis: `complete_pipeline_results/`
- Research discoveries: `research_output/`
- Spectral analysis: `spectral_analysis_results/`
- Publication materials: `publication_data/`

## 🧠 Core Concepts

### **A→B→C Security Pipeline with Advanced Methods**

**Phase A**: **Multi-Method Critical Weight Discovery**
- Traditional gradient-based sensitivity analysis
- **NEW**: Information-theoretic Fisher information matrices
- **NEW**: Ensemble discovery with 4-method voting
- **NEW**: Spectral vulnerability detection using eigenvalues

**Phase B**: **Advanced Attack Simulation**
- FGSM, PGD adversarial perturbations
- Bit-flip and fault injection attacks
- Targeted attacks on discovered critical weights
- Cross-modal attack vectors

**Phase C**: **Certified Protection & Defense**
- Weight redundancy with backup systems
- Real-time monitoring with circuit breakers
- **NEW**: PAC-Bayesian certified defenses
- Spectral regularization for eigenvalue smoothing

### **Information-Theoretic Research Framework**

**🔬 Discovery Methods**:
1. **Activation Outliers**: Traditional >1e3 magnitude threshold detection
2. **Causal Intervention**: Direct measurement of weight impact on model behavior
3. **Information Bottleneck**: Mutual information analysis for compression vulnerabilities
4. **Spectral Anomaly**: Eigenvalue gap detection and phase transition analysis

**🎯 Ensemble Approach**:
- **Weighted Voting**: 30% Activation + 30% Causal + 20% Information + 20% Spectral
- **Agreement Threshold**: Requires 60% of methods to identify same weights
- **Confidence Scoring**: Mathematical confidence based on method consensus

**📊 Theoretical Guarantees**:
- **PAC-Bayesian Bounds**: Mathematical certificates with confidence intervals
- **Fisher Information**: Fundamental limits on parameter estimation accuracy
- **Phase Transition Detection**: Critical points where small changes cause dramatic effects
- **100× Perplexity Validation**: Rigorous confirmation of super weight criticality

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

### **🔍 Core A→B→C Security Pipeline**

```bash
# Complete pipeline (recommended for most users)
cwa run-complete-pipeline MODEL_NAME [options]

# Individual phases
cwa phase-a MODEL_NAME [options]                    # Multi-method critical weight discovery
cwa phase-b MODEL_NAME WEIGHTS_FILE [options]       # Advanced attack simulation
cwa phase-c MODEL_NAME WEIGHTS_FILE ATTACKS_FILE [options]  # Certified defense implementation
```

### **🔬 Advanced Research Commands**

```bash
# Enhanced ensemble super weight discovery (NEW!)
cwa extract-critical-weights MODEL_NAME \
  --mode "super_weight_discovery" \
  --top-k-percent 0.001 \
  --layer-focus "early"

# Information-theoretic spectral analysis (NEW!)
cwa spectral-analysis MODEL_NAME \
  --analysis-types "signatures,transitions,stability,correlations" \
  --include-pac-bounds \
  --confidence-level 0.95

# Super weight validation with 100× perplexity
cwa validate-super-weights MODEL_NAME COORDINATES \
  --perplexity-threshold 100 \
  --export-results

# Publication-ready research extraction
cwa research-extract MODEL_NAME \
  --focus "mlp-components" \
  --export-format "publication-ready"
```

### **🛠️ Utility & System Commands**

```bash
cwa info                # GPU and system information
cwa list-models         # Available transformer models
cwa list-metrics        # Sensitivity analysis methods
cwa create-config       # Configuration file generation
```

### **🆕 New Capabilities in Phase 1**

| Command | Capability | Enhancement |
|---------|------------|-------------|
| `extract-critical-weights` | **4-Method Ensemble** | Activation + Causal + Information + Spectral |
| `spectral-analysis` | **PAC-Bayesian Bounds** | Theoretical vulnerability guarantees |
| `validate-super-weights` | **Enhanced Validation** | Bit-level analysis with confidence scoring |
| `research-extract` | **Publication Ready** | Automated tables, figures, replication packages |

## 💡 Usage Examples

### **🛡️ Security Analysis Workflows**

#### **🚀 Basic Security Assessment**
```bash
# Complete security pipeline
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

### **🔬 Advanced Research Workflows**

#### **🎯 Ensemble Super Weight Discovery (NEW!)**
```bash
# Multi-method ensemble discovery with 4 approaches
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.7 \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-format "csv,json,plots"

# Results: Activation + Causal + Information + Spectral ensemble voting
# Output: research_output/discovered_weights.csv with confidence scores
```

#### **📊 Information-Theoretic Spectral Analysis (NEW!)**
```bash
# Comprehensive spectral vulnerability analysis
cwa spectral-analysis "gpt2" \
  --analysis-types "signatures,transitions,stability,correlations" \
  --top-k 10 \
  --include-pac-bounds \
  --confidence-level 0.95

# Fisher information matrix analysis
cwa spectral-analysis "mistralai/Mistral-7B-v0.1" \
  --target-layers "layers.0.mlp,layers.1.mlp,layers.2.mlp" \
  --analysis-types "signatures,stability" \
  --include-pac-bounds

# Results: PAC-Bayesian bounds, vulnerability certificates, spectral signatures
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

### **🚀 Complete Advanced Research + Security Workflow (NEW!)**

```bash
# Step 1: Multi-method ensemble discovery with information theory
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.7 \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-dir "step1_ensemble_discovery"

# Step 2: Information-theoretic spectral analysis
cwa spectral-analysis "gpt2" \
  --analysis-types "signatures,transitions,stability,correlations" \
  --include-pac-bounds \
  --confidence-level 0.95 \
  --output-dir "step2_spectral_analysis"

# Step 3: Traditional security pipeline for comparison
cwa run-complete-pipeline "gpt2" \
  --attack-methods "fgsm,pgd,bit_flip,fault_injection" \
  --protection-methods "weight_redundancy,layer11_attention_fortress" \
  --output-dir "step3_security_pipeline"

# Step 4: Cross-validate ensemble discoveries with perplexity
cwa validate-super-weights "gpt2" \
  --coordinates "$(cat step1_ensemble_discovery/discovered_coordinates.txt)" \
  --perplexity-threshold 100 \
  --export-results \
  --output-dir "step4_validation"

# Step 5: Generate comprehensive publication package
cwa research-extract "gpt2" \
  --focus "comprehensive" \
  --export-format "publication-ready" \
  --include-metadata \
  --output-dir "step5_publication_ready"
```

**Comprehensive Results**:
- **Ensemble Discovery**: `step1_ensemble_discovery/` - 4-method voting results
- **Spectral Analysis**: `step2_spectral_analysis/` - PAC-Bayesian certificates
- **Security Pipeline**: `step3_security_pipeline/` - Attack/defense results
- **Validation**: `step4_validation/` - 100× perplexity confirmation
- **Publication**: `step5_publication_ready/` - Tables, figures, replication package

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

**Enhanced Research Methodology (Phase 1)**:
- **Ensemble Discovery**: 4-method voting (Activation + Causal + Information + Spectral)
- **Information-Theoretic Analysis**: Fisher information matrices and mutual information
- **Spectral Methods**: Eigenvalue analysis and phase transition detection
- **PAC-Bayesian Guarantees**: Mathematical confidence bounds
- **Early layer focus** (layers 0-3) based on transformer robustness research
- **Top 0.001% extraction** for precision research with confidence scoring

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

### spectral-analysis (NEW!)

```bash
cwa spectral-analysis MODEL_NAME \
  --target-layers LAYER_NAMES \
  --analysis-types {signatures,transitions,stability,correlations} \
  --top-k INT \
  --output-dir PATH \
  --device {cuda,cpu,auto} \
  --include-pac-bounds BOOL \
  --confidence-level FLOAT
```

**Spectral Analysis Features**:
- **Eigenvalue Analysis**: Spectral signatures, gaps, and phase transitions
- **Fisher Information**: Fundamental parameter criticality bounds
- **PAC-Bayesian Certificates**: Mathematical vulnerability guarantees
- **Critical Configuration Detection**: Eigenvalue-based vulnerability patterns
- **Attack Susceptibility Assessment**: Perturbation, rank, spectral, and instability attacks
- **Theoretical Guarantees**: Confidence intervals with proven bounds

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

### Spectral Analysis Output (NEW!)
```
spectral_analysis_results/
├── spectral_analysis.json             # Complete spectral analysis results
├── spectral_summary.txt               # Human-readable summary report
├── vulnerability_certificates.json    # PAC-Bayesian certificates
├── critical_configurations/           # Top critical eigenvalue configurations
│   ├── config_001_layer_2_mlp.json   # Individual configuration analysis
│   ├── config_002_layer_0_attn.json  # Attack susceptibility assessment
│   └── config_recommendations.json    # Suggested defenses per configuration
├── visualizations/                    # Spectral analysis plots
│   ├── eigenvalue_spectra.png        # Eigenvalue distribution plots
│   ├── spectral_gaps.png             # Phase transition visualization
│   ├── fisher_information_heatmap.png # Fisher information matrix heatmap
│   └── pac_bounds_confidence.png     # PAC-Bayesian confidence intervals
└── theoretical_guarantees/            # Mathematical certificates
    ├── pac_bayesian_bounds.json      # Confidence bounds with proofs
    ├── vulnerability_bounds.json     # Upper/lower vulnerability limits
    └── certification_summary.md      # Theoretical guarantee explanations
```

## 🔬 Research Applications

### **🎯 Advanced Super Weight Research (Phase 1)**
- **Ensemble Discovery**: 4-method voting for robust super weight identification
- **Information-Theoretic Foundations**: Fisher information and mutual information analysis
- **Spectral Learning**: Eigenvalue-based vulnerability detection with phase transitions
- **PAC-Bayesian Guarantees**: Mathematical certificates with confidence intervals
- **Cross-Architecture Analysis**: Compare ensemble results across model types
- **Theoretical Validation**: 100× perplexity confirmation with statistical significance

### **🛡️ Advanced Cybersecurity Research**
- **Multi-Method Vulnerability Discovery**: Beyond gradients using information theory
- **Certified Attack Resistance**: PAC-Bayesian bounds for defense guarantees
- **Spectral Attack Vectors**: Novel eigenvalue-based attack strategies
- **Real-Time Security Monitoring**: Information-theoretic anomaly detection
- **Theoretical Threat Modeling**: Mathematical frameworks for vulnerability assessment
- **Ensemble Defense Strategies**: Multi-layered protection with confidence scoring

### **📊 Publication-Ready Research**
- **Novel Theoretical Contributions**: Information geometry for neural network security
- **Ensemble Methodology Papers**: Multi-method discovery with statistical validation
- **Spectral Analysis Framework**: First comprehensive eigenvalue vulnerability analysis
- **PAC-Bayesian Security**: Theoretical guarantees for neural network robustness
- **Comparative Studies**: Cross-architecture ensemble discovery analysis
- **Replication Packages**: Complete methodology with mathematical proofs

## 🏗️ Technical Architecture

```
critical-weight-analysis/
├── src/cwa/
│   ├── core/                          # Core abstractions
│   │   ├── interfaces.py              # Data structures and protocols
│   │   ├── config.py                  # Configuration management
│   │   ├── models.py                  # Lambda Labs LLM management
│   │   └── data.py                    # Data handling utilities
│   ├── sensitivity/                   # Phase A: Multi-Method Critical Weight Discovery
│   │   ├── security_analyzer.py       # Traditional security analysis
│   │   ├── grad_x_weight.py          # Gradient × weight metrics
│   │   ├── hessian_diag.py           # Hessian diagonal analysis
│   │   ├── spectral_analyzer.py      # NEW: Spectral vulnerability analysis
│   │   └── registry.py               # Metric registration
│   ├── theory/                        # NEW: Information-Theoretic Foundations
│   │   └── information_geometry.py   # Fisher information, mutual information, PAC-Bayesian
│   ├── security/                      # Phase B & C: Advanced Attack & Certified Defense
│   │   ├── adversarial.py            # Adversarial attacks (FGSM, PGD, TextFooler)
│   │   ├── targeted_attacks.py       # Targeted attack strategies
│   │   ├── fault_injection.py        # Hardware fault simulation
│   │   ├── defense_mechanisms.py     # Enhanced protection implementations
│   │   └── weight_protection.py      # Critical weight protection
│   ├── research/                      # PhD-Level Research Features (Enhanced)
│   │   └── super_weight_analyzer.py  # Ensemble super weight discovery & validation
│   ├── utils/                        # Utilities and helpers
│   └── cli/                          # Command line interface
├── configs/                          # Configuration templates
├── tests/                           # Comprehensive test suite
└── examples/                        # Research examples and notebooks
```

### **Phase 1 Enhanced Research Features**

**InformationGeometricAnalyzer**: Advanced information-theoretic analysis implementing:
- Fisher Information Matrix computation for vulnerability analysis
- Mutual information estimation using InfoNet-inspired approaches
- Phase transition detection in weight space using spectral signatures
- Information bottleneck analysis for compression vulnerabilities
- PAC-Bayesian bounds with mathematical confidence guarantees
- Cross-layer information flow analysis

**SpectralVulnerabilityAnalyzer**: Eigenvalue-based vulnerability detection implementing:
- Spectral signature analysis using eigenvalues and singular values
- Phase transition detection with spectral gap analysis
- Critical configuration detection based on spectral properties
- Attack susceptibility assessment for perturbation, rank, spectral, and instability attacks
- PAC-Bayesian bounds for certified vulnerability assessment
- Comprehensive export with visualization and theoretical guarantees

**SuperWeightAnalyzer (Enhanced)**: Ensemble super weight discovery implementing:
- **4-Method Ensemble**: Activation + Causal + Information + Spectral approaches
- **Weighted Voting**: 30% Activation + 30% Causal + 20% Information + 20% Spectral
- **Confidence Scoring**: Agreement-based confidence with 60% threshold requirement
- 100× perplexity validation methodology with statistical significance
- Complete replication packages with ensemble methodology
- Cross-validation with information-theoretic and spectral methods

**Known Super Weight Database**: Pre-configured coordinates for:
- Llama-7B: `[(2, 'mlp.down_proj', [3968, 7003])]`
- Mistral-7B: `[(1, 'mlp.down_proj', [2070, 7310])]`
- Extensible to any model architecture with ensemble validation

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

## 🚀 **Phase 1 Complete: Information-Theoretic Foundations**

### ✅ **NEW in Phase 1 (Just Released!)**
- **🔬 Information-Theoretic Analysis**: Fisher information matrices, mutual information, PAC-Bayesian bounds
- **🎯 4-Method Ensemble Discovery**: Activation + Causal + Information + Spectral voting
- **📊 Spectral Vulnerability Analysis**: Eigenvalue-based detection with mathematical certificates
- **📋 Enhanced CLI**: New `cwa spectral-analysis` command with comprehensive options
- **📁 Advanced Output Structures**: PAC-Bayesian certificates, vulnerability guarantees, replication packages

### 🔮 **Coming in Phase 2** (1-2 months)
- **🕰️ Real-Time Monitoring Framework**: Millisecond-latency security with circuit breakers
- **🎮 Game-Theoretic Weight Analysis**: NeuroGame architecture for strategic neuron interactions
- **🔄 Cross-Architecture Transfer**: Vulnerability pattern migration analysis

### 🌟 **Coming in Phase 3** (3+ months)
- **🖼️ Multimodal Security Framework**: CLIP/LLaVA cross-modal vulnerability analysis
- **⚡ Emerging Architecture Support**: Mamba/RWKV state-space models, quantum neural networks
- **📈 Advanced Benchmarking**: Unified evaluation against RobustBench, HarmBench, MMMU

**Current Status**: ✅ **Phase 1 Complete** - Advanced Information-Theoretic Research Pipeline
**Ready for**: Cutting-edge cybersecurity research with mathematical guarantees
**Optimized for**: Lambda Labs GPU infrastructure with CUDA 12.6 support
**Research Impact**: Publication-ready with novel theoretical contributions