# Critical Weight Analysis & Cybersecurity Research Tool

A comprehensive **A→B→C security research pipeline** for analyzing critical weights in **Large Language Models** with cutting-edge **PhD-level research capabilities** and **information-theoretic foundations**, optimized for **Lambda Labs GPU infrastructure**.

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repo-url> critical-weight-analysis
cd critical-weight-analysis

# Install with dependencies
pip install -e .

# Verify installation and GPU setup
cwa info
```

### Fastest Start - Traditional Security Analysis
```bash
# Run complete security pipeline
cwa security run-complete-pipeline "gpt2"

# Or use individual phases
cwa phase-a "gpt2" --output-dir outputs/phase_a
cwa phase-b "gpt2" outputs/phase_a/critical_weights.yaml --output-dir outputs/phase_b
cwa phase-c "gpt2" outputs/phase_a/critical_weights.yaml outputs/phase_b/attack_results.yaml --output-dir outputs/phase_c
```

### Advanced Research Workflows

Pre-built automation scripts are available in the `scripts/` directory:

```bash
# Phase 1: Information-theoretic research workflow
./scripts/run_phase1_workflow.sh

# Phase 2: Game theory and real-time monitoring workflow
./scripts/run_phase1_phase2_workflow.sh

# Complete integrated analysis (all capabilities)
./scripts/run_complete_integrated_workflow.sh

# Test with small models for development
./scripts/run_small_model_test.sh
```

These scripts automatically run the proper command sequences with the correct modular CLI structure.

## 📋 Available Commands

The tool uses a modular CLI structure. Check all available commands:

```bash
cwa --help                    # Main help
cwa basic --help             # Basic operations
cwa security --help          # A/B/C Security phases
cwa research --help          # Research & super weight analysis
cwa monitor --help           # Real-time monitoring
cwa advanced --help          # Game theory & transfer analysis
```

### Core Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `cwa info` | System & GPU information | `cwa info` |
| `cwa basic list-models` | Available models | `cwa basic list-models` |
| `cwa phase-a` | Critical weight discovery | `cwa phase-a "gpt2"` |
| `cwa extract-critical-weights` | Research extraction | `cwa extract-critical-weights "gpt2"` |

### Modular Command Structure

The tool uses a hierarchical command structure for better organization:

```bash
cwa basic        # Basic operations (info, list-models, etc.)
cwa security     # A/B/C Security phases (phase-a, phase-b, phase-c, run-complete-pipeline)
cwa research     # Research commands (extract-critical-weights, spectral-analysis, validate-super-weights)
cwa monitor      # Real-time monitoring (monitor-realtime)
cwa advanced     # Game theory & transfer analysis (analyze-game-theory, analyze-transfer)
```

### Research Commands

```bash
# Super weight discovery with ensemble methods
cwa research extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.7 \
  --output-dir "research_output"

# Spectral vulnerability analysis
cwa research spectral-analysis "gpt2" \
  --analysis-types "signatures,transitions,stability" \
  --include-pac-bounds \
  --output-dir "spectral_results"

# Super weight validation
cwa research validate-super-weights "gpt2" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --perplexity-threshold 100 \
  --output-dir "validation_results"
```

### Advanced Security Commands

```bash
# Real-time monitoring
cwa monitor monitor-realtime "gpt2" \
  --detection-algorithms "statistical,gradient,activation" \
  --latency-target 1.0 \
  --output-dir "monitoring_results"

# Game-theoretic analysis
cwa advanced analyze-game-theory "gpt2" \
  --game-types "nash_equilibrium,cooperative" \
  --max-players 50 \
  --output-dir "game_theory_results"

# Cross-architecture transfer analysis
cwa advanced analyze-transfer "gpt2" "distilgpt2" \
  --mapping-strategies "geometric,semantic" \
  --output-dir "transfer_results"
```

## 🏗️ Architecture Overview

### Core Phases

**Phase A: Critical Weight Discovery**
- Traditional gradient-based analysis
- Information-theoretic Fisher information matrices
- Ensemble discovery with 4-method voting
- Spectral vulnerability detection

**Phase B: Attack Simulation**
- FGSM, PGD adversarial attacks
- Bit-flip and fault injection
- Targeted attacks on critical weights

**Phase C: Protection & Defense**
- Weight redundancy systems
- Real-time monitoring with circuit breakers
- PAC-Bayesian certified defenses

### Advanced Research Features

**Information-Theoretic Analysis (Phase 1)**
- 4-method ensemble: Activation + Causal + Information + Spectral
- Fisher information matrices and mutual information
- PAC-Bayesian theoretical guarantees
- 100× perplexity validation methodology

**Game Theory & Real-Time Security (Phase 2)**
- Nash equilibrium modeling of neural security
- Cooperative game theory for weight coalitions
- Evolutionary stability analysis (ESS)
- Sub-millisecond real-time monitoring
- Cross-architecture vulnerability transfer

### Project Structure

```
critical-weight-analysis/
├── src/cwa/
│   ├── cli/                    # Modular command-line interface
│   ├── core/                   # Core abstractions & models
│   ├── sensitivity/            # Phase A: Critical weight discovery
│   ├── security/               # Phase B & C: Attacks & defenses
│   ├── research/               # Super weight research features
│   ├── game_theory/            # Phase 2: Game-theoretic analysis
│   ├── monitoring/             # Phase 2: Real-time monitoring
│   └── evaluation/             # Evaluation metrics
├── scripts/                    # Workflow automation scripts
├── configs/                    # Configuration templates
├── tests/                      # Comprehensive test suite
└── outputs/                    # Analysis results
```

## 📊 Output Structure

### Traditional Security Pipeline
```
outputs/
├── phase_a/
│   └── critical_weights.yaml          # Discovered critical weights
├── phase_b/
│   └── attack_results.yaml            # Attack simulation results
└── phase_c/
    └── protection_results.yaml        # Defense effectiveness
```

### Research Output
```
research_output/
├── research_data.json                 # Complete analysis with metadata
├── discovered_weights.csv             # Critical weight coordinates
├── statistics.csv                     # Statistical analysis
├── visualizations/                    # Analysis plots
└── publication_data/                  # Publication-ready materials
```

### Advanced Analysis Output
```
outputs/
├── spectral_analysis_results/          # Spectral vulnerability analysis
├── game_theory_analysis_results/       # Game-theoretic modeling
├── realtime_monitoring_results/        # Real-time security monitoring
└── transfer_analysis_results/          # Cross-architecture analysis
```

## 🔬 Research Applications

### Core Research Areas

**Traditional Security Research**
- Multi-method vulnerability discovery beyond gradients
- Certified attack resistance with PAC-Bayesian bounds
- Theoretical threat modeling frameworks

**Information-Theoretic Security**
- Fisher information analysis for parameter criticality
- Information bottleneck vulnerability detection
- Phase transition analysis in weight space

**Game-Theoretic Security Modeling**
- Strategic modeling of neural networks as multi-player games
- Nash equilibrium configurations for security
- Cooperative coalitions for defensive strategies
- Evolutionary dynamics of long-term security

**Real-Time Production Security**
- Sub-millisecond anomaly detection systems
- Circuit breaker patterns for fail-safe operation
- Adaptive security thresholds with mathematical guarantees

### Publication-Ready Features

- Automated table and figure generation
- Complete replication packages
- Mathematical certificates with confidence intervals
- Cross-architecture comparative studies

## 💻 Hardware Requirements

| Model Size | Recommended GPU | Memory | Pipeline Time |
|------------|----------------|--------|---------------|
| **Small** (GPT-2, DialoGPT) | Any GPU | < 2GB | < 10 min |
| **Medium** (Mistral-7B, Llama-7B) | A100 40GB | ~20GB | < 30 min |
| **Large** (Llama-13B+) | A100 80GB | 40-70GB | < 60 min |

## 🧪 Testing & Development

```bash
# Run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_phase_a.py -v
pytest tests/test_research.py -v

# Code quality checks
black src/
ruff check src/
mypy src/
```

## 🤝 Contributing

This tool is built with:
- **Clean Architecture**: Modular separation of concerns
- **Extensible Design**: Easy to add new metrics and methods
- **Lambda Labs Optimization**: Full GPU acceleration support
- **Comprehensive Testing**: Validated across model architectures

## 🏆 Citation

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

## 🚀 Current Status

✅ **Phase 1 Complete**: Information-theoretic foundations with ensemble super weight discovery
✅ **Phase 2 Complete**: Game-theoretic analysis and real-time monitoring
🔄 **Ready for Research**: Publication-ready with novel theoretical contributions

**Optimized for**: Lambda Labs GPU infrastructure with CUDA support
**Research Impact**: Cutting-edge cybersecurity research with mathematical guarantees