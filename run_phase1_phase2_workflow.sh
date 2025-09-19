#!/bin/bash
# Complete Phase 1 + Phase 2 Research Workflow Script
# Runs comprehensive information-theoretic, game-theoretic, and real-time security analysis

set -e  # Exit on error

MODEL_NAME="gpt2"  # Change this to your preferred model
COMPARE_MODEL="distilgpt2"  # For transfer analysis
OUTPUT_BASE="phase1_phase2_analysis_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Complete Phase 1 + Phase 2 Research Workflow"
echo "Primary model: $MODEL_NAME"
echo "Comparison model: $COMPARE_MODEL"
echo "Output directory: $OUTPUT_BASE"

# Create base output directory
mkdir -p "$OUTPUT_BASE"

echo ""
echo "📊 Step 1: System Information"
cwa info

echo ""
echo "🔍 Step 2: Multi-Method Ensemble Discovery (Phase 1)"
cwa extract-critical-weights "$MODEL_NAME" \
  --mode "super_weight_discovery" \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-dir "$OUTPUT_BASE/step2_ensemble_discovery"

echo ""
echo "🌊 Step 3: Information-Theoretic Spectral Analysis (Phase 1)"
cwa spectral-analysis "$MODEL_NAME" \
  --analysis-types "signatures,transitions,stability,correlations" \
  --include-pac-bounds \
  --confidence-level 0.95 \
  --output-dir "$OUTPUT_BASE/step3_spectral_analysis"

echo ""
echo "🕰️ Step 4: Real-Time Security Monitoring (Phase 2 - NEW!)"
cwa monitor-realtime "$MODEL_NAME" \
  --detection-algorithms "statistical,gradient,activation,weight_drift" \
  --circuit-breaker-config "auto" \
  --latency-target 1.0 \
  --output-dir "$OUTPUT_BASE/step4_realtime_monitoring"

echo ""
echo "🎮 Step 5: Game-Theoretic Vulnerability Analysis (Phase 2 - NEW!)"
cwa analyze-game-theory "$MODEL_NAME" \
  --game-types "nash_equilibrium,cooperative,evolutionary" \
  --max-players 50 \
  --strategy-space-size 10 \
  --convergence-threshold 1e-6 \
  --output-dir "$OUTPUT_BASE/step5_game_theory"

echo ""
echo "🔄 Step 6: Cross-Architecture Transfer Analysis (Phase 2 - NEW!)"
cwa analyze-transfer "$MODEL_NAME" "$COMPARE_MODEL" \
  --mapping-strategies "geometric,semantic,interpolation" \
  --transfer-types "architecture,vulnerability" \
  --similarity-threshold 0.7 \
  --output-dir "$OUTPUT_BASE/step6_transfer_analysis"

echo ""
echo "🛡️ Step 7: Traditional Security Pipeline (Baseline Comparison)"
cwa run-complete-pipeline "$MODEL_NAME" \
  --attack-methods "fgsm,pgd,bit_flip" \
  --protection-methods "weight_redundancy,checksums" \
  --output-dir "$OUTPUT_BASE/step7_security_pipeline"

echo ""
echo "✅ Step 8: Cross-Validate Discovered Weights"
# Check if coordinates file exists
COORDS_FILE="$OUTPUT_BASE/step2_ensemble_discovery/discovered_coordinates.txt"
if [ -f "$COORDS_FILE" ]; then
    cwa validate-super-weights "$MODEL_NAME" \
      --coordinates "$(cat $COORDS_FILE)" \
      --perplexity-threshold 100 \
      --export-results \
      --output-dir "$OUTPUT_BASE/step8_validation"
else
    echo "⚠️  Coordinates file not found, using known super weights for validation"
    # Use known coordinates as example
    cwa validate-super-weights "$MODEL_NAME" \
      --coordinates "[(0, 'transformer.h.0.mlp.c_fc', [100, 200])]" \
      --perplexity-threshold 100 \
      --export-results \
      --output-dir "$OUTPUT_BASE/step8_validation"
fi

echo ""
echo "📚 Step 9: Generate Comprehensive Publication Package"
cwa research-extract "$MODEL_NAME" \
  --focus "comprehensive" \
  --export-format "publication-ready" \
  --include-metadata \
  --output-dir "$OUTPUT_BASE/step9_publication"

echo ""
echo "🎉 Complete Phase 1 + Phase 2 Workflow Finished!"
echo ""
echo "📁 Results located in: $OUTPUT_BASE/"
echo ""
echo "📊 Key outputs:"
echo "  📈 Phase 1 Results:"
echo "    - Ensemble discovery: $OUTPUT_BASE/step2_ensemble_discovery/"
echo "    - Spectral analysis: $OUTPUT_BASE/step3_spectral_analysis/"
echo "  ⚡ Phase 2 Results:"
echo "    - Real-time monitoring: $OUTPUT_BASE/step4_realtime_monitoring/"
echo "    - Game theory analysis: $OUTPUT_BASE/step5_game_theory/"
echo "    - Transfer analysis: $OUTPUT_BASE/step6_transfer_analysis/"
echo "  🛡️ Security & Validation:"
echo "    - Security pipeline: $OUTPUT_BASE/step7_security_pipeline/"
echo "    - Weight validation: $OUTPUT_BASE/step8_validation/"
echo "    - Publication materials: $OUTPUT_BASE/step9_publication/"
echo ""
echo "🔬 Ready for cutting-edge research publication!"
echo "🚀 All Phase 1 & 2 capabilities now fully operational!"