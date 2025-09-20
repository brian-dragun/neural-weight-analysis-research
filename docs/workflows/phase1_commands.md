# Phase 1 & 2 Available Commands

## Quick Setup
```bash
# Install the package
pip install -e .

# Verify setup
python test_setup.py

# Check CLI is working
cwa --help
```

## Phase 1: Information-Theoretic Analysis

### 1. System Information
```bash
cwa info                    # GPU and system information
cwa list-models            # Available models
cwa list-metrics           # Available sensitivity metrics
```

### 2. Ensemble Super Weight Discovery
```bash
# Multi-method ensemble discovery (Phase 1)
cwa extract-critical-weights "gpt2" \
  --mode "super_weight_discovery" \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-dir "discovery_results"
```

### 3. Information-Theoretic Spectral Analysis
```bash
# Advanced spectral analysis with PAC-Bayesian bounds
cwa spectral-analysis "gpt2" \
  --analysis-types "signatures,transitions,stability,correlations" \
  --include-pac-bounds \
  --confidence-level 0.95 \
  --output-dir "spectral_results"
```

### 4. Super Weight Validation
```bash
# Validate discovered super weights with 100× perplexity
cwa validate-super-weights "gpt2" \
  --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
  --perplexity-threshold 100 \
  --export-results \
  --output-dir "validation_results"
```

### 5. Publication-Ready Research
```bash
# Generate publication materials
cwa research-extract "gpt2" \
  --focus "comprehensive" \
  --export-format "publication-ready" \
  --include-metadata \
  --output-dir "publication_results"
```

## Traditional Security Pipeline

### 6. Complete Security Pipeline
```bash
# Run full A→B→C security analysis
cwa run-complete-pipeline "gpt2" \
  --attack-methods "fgsm,pgd,bit_flip,fault_injection" \
  --protection-methods "weight_redundancy,checksums" \
  --output-dir "security_results"
```

### 7. Individual Security Phases
```bash
# Phase A: Critical weight discovery
cwa phase-a "gpt2" \
  --metric "security_gradient" \
  --vulnerability-threshold 0.8 \
  --top-k 500 \
  --output-dir "phase_a_results"

# Phase B: Attack simulation
cwa phase-b "gpt2" "phase_a_results/critical_weights.yaml" \
  --attack-methods "fgsm,pgd,bit_flip" \
  --output-dir "phase_b_results"

# Phase C: Defense implementation
cwa phase-c "gpt2" "phase_a_results/critical_weights.yaml" "phase_b_results/attack_results.yaml" \
  --protection-methods "weight_redundancy,checksums" \
  --output-dir "phase_c_results"
```

## Phase 2: Game Theory & Real-Time Monitoring (NEW!)

### 8. Real-Time Security Monitoring
```bash
# Production-ready security monitoring with circuit breakers
cwa monitor-realtime "gpt2" \
  --detection-algorithms "statistical,gradient,activation,weight_drift" \
  --circuit-breaker-config "auto" \
  --latency-target 1.0 \
  --anomaly-thresholds "adaptive" \
  --output-dir "monitoring_results"
```

### 9. Game-Theoretic Weight Analysis
```bash
# Complete game-theoretic analysis (Nash, cooperative, evolutionary)
cwa analyze-game-theory "gpt2" \
  --game-types "nash_equilibrium,cooperative,evolutionary" \
  --max-players 50 \
  --strategy-space-size 10 \
  --convergence-threshold 1e-6 \
  --output-dir "game_theory_results"

# Just Nash equilibrium analysis
cwa analyze-game-theory "gpt2" \
  --game-types "nash_equilibrium" \
  --max-players 30 \
  --output-dir "nash_results"

# Cooperative coalition analysis with Shapley values
cwa analyze-game-theory "gpt2" \
  --game-types "cooperative" \
  --coalition-analysis "shapley_values,core_analysis" \
  --max-players 20 \
  --output-dir "cooperative_results"

# Evolutionary stability analysis
cwa analyze-game-theory "gpt2" \
  --game-types "evolutionary" \
  --dynamics-type "replicator" \
  --output-dir "evolutionary_results"
```

### 10. Cross-Architecture Transfer Analysis
```bash
# Comprehensive transfer analysis between models
cwa analyze-transfer "gpt2" "distilgpt2" \
  --mapping-strategies "geometric,semantic,interpolation" \
  --transfer-types "architecture,vulnerability" \
  --similarity-threshold 0.7 \
  --output-dir "transfer_results"

# Just geometric mapping
cwa analyze-transfer "gpt2" "microsoft/DialoGPT-small" \
  --mapping-strategies "geometric" \
  --similarity-threshold 0.8 \
  --output-dir "geometric_mapping_results"

# Vulnerability transfer focus
cwa analyze-transfer "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-7b-hf" \
  --mapping-strategies "semantic,interpolation" \
  --vulnerability-transfer-analysis true \
  --output-dir "vulnerability_transfer_results"
```

## Model Options

### Small Models (Fast testing)
- `"gpt2"` - GPT-2 base model
- `"distilgpt2"` - DistilGPT-2
- `"microsoft/DialoGPT-small"` - Dialog model

### Medium Models (Lambda Labs A100)
- `"mistralai/Mistral-7B-v0.1"` - Mistral 7B
- `"microsoft/DialoGPT-medium"` - Medium dialog model

### Large Models (Lambda Labs A100 80GB)
- `"meta-llama/Llama-2-7b-hf"` - Llama 2 7B
- `"meta-llama/Llama-2-13b-hf"` - Llama 2 13B

## Example Workflows

### Complete Phase 1 + 2 Workflow
```bash
# Full integrated Phase 1 & 2 analysis
./run_phase1_phase2_workflow.sh

# Or just Phase 1
./run_phase1_workflow.sh
```

### Individual Phase 2 Commands
```bash
# Quick Phase 2 test
cwa info
cwa monitor-realtime "gpt2" --latency-target 1.0
cwa analyze-game-theory "gpt2" --game-types "nash_equilibrium"
cwa analyze-transfer "gpt2" "distilgpt2" --mapping-strategies "geometric"
```

### Step-by-Step Complete Analysis
```bash
# Phase 1: Information-theoretic foundations
cwa extract-critical-weights "gpt2" --mode "super_weight_discovery" --top-k-percent 0.001
cwa spectral-analysis "gpt2" --include-pac-bounds --confidence-level 0.95

# Phase 2: Advanced security & game theory
cwa monitor-realtime "gpt2" --detection-algorithms "statistical,gradient,activation"
cwa analyze-game-theory "gpt2" --game-types "nash_equilibrium,cooperative"
cwa analyze-transfer "gpt2" "distilgpt2" --mapping-strategies "semantic,geometric"

# Traditional security pipeline
cwa run-complete-pipeline "gpt2"

# Publication materials
cwa research-extract "gpt2" --export-format "publication-ready"
```

## Output Structure

### Phase 1 + 2 Complete Results
```
phase1_phase2_analysis_TIMESTAMP/
├── step2_ensemble_discovery/     # Phase 1: Multi-method ensemble results
├── step3_spectral_analysis/      # Phase 1: PAC-Bayesian spectral analysis
├── step4_realtime_monitoring/    # Phase 2: Real-time security monitoring
├── step5_game_theory/            # Phase 2: Game-theoretic analysis
├── step6_transfer_analysis/      # Phase 2: Cross-architecture transfer
├── step7_security_pipeline/      # Traditional attack/defense baseline
├── step8_validation/             # Super weight validation
└── step9_publication/            # Publication-ready materials
```

### Individual Analysis Results
```
# Real-time monitoring
monitoring_results/
├── monitoring_session.json      # Complete session data
├── circuit_breaker_logs.json    # Circuit breaker events
└── anomaly_alerts/              # Detected anomalies

# Game theory analysis
game_theory_results/
├── game_theory_analysis.json    # Complete analysis
├── nash_equilibrium/            # Nash equilibrium results
├── cooperative_analysis/        # Coalition & Shapley values
└── evolutionary_stability/      # ESS and replicator dynamics

# Transfer analysis
transfer_results/
├── transfer_analysis.json       # Complete transfer analysis
├── pattern_extraction/          # Cross-architecture patterns
├── architecture_mapping/        # Weight mapping strategies
└── vulnerability_transfer/      # Vulnerability migration analysis
```