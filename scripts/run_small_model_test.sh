#!/bin/bash
# Small Model Test Workflow - Fast execution for testing all capabilities
# Optimized for quick testing and validation of the complete pipeline

set -e  # Exit on error

MODEL_NAME="${1:-microsoft/DialoGPT-small}"  # Default to DialoGPT-small
COMPARE_MODEL="${2:-distilgpt2}"  # Comparison model for transfer
OUTPUT_BASE="small_model_test_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Small Model Test Workflow"
echo "🎯 Primary model: $MODEL_NAME"
echo "🔄 Comparison model: $COMPARE_MODEL"
echo "📁 Output directory: $OUTPUT_BASE"
echo "⚡ Optimized for fast execution"
echo ""

# Create base output directory
mkdir -p "$OUTPUT_BASE"

echo "📊 Step 1: System Information"
echo "=============================="
cwa info
echo ""

echo "🛡️ Step 2: Quick Traditional Security Test"
echo "=========================================="
cwa run-complete-pipeline "$MODEL_NAME" \
  --attack-methods "fgsm,pgd" \
  --protection-methods "weight_redundancy,checksums" \
  --output-dir "$OUTPUT_BASE/traditional_security"
echo "✅ Traditional security complete"
echo ""

echo "🔍 Step 3: Phase 1 - Ensemble Discovery (Fast)"
echo "=============================================="
cwa extract-critical-weights "$MODEL_NAME" \
  --mode "super_weight_discovery" \
  --sensitivity-threshold 0.5 \
  --top-k-percent 0.01 \
  --layer-focus "early" \
  --output-dir "$OUTPUT_BASE/ensemble_discovery"
echo "✅ Ensemble discovery complete"
echo ""

echo "🌊 Step 4: Phase 1 - Spectral Analysis (Core)"
echo "============================================="
cwa spectral-analysis "$MODEL_NAME" \
  --analysis-types "signatures,stability" \
  --include-pac-bounds \
  --confidence-level 0.90 \
  --top-k 5 \
  --output-dir "$OUTPUT_BASE/spectral_analysis"
echo "✅ Spectral analysis complete"
echo ""

echo "🕰️ Step 5: Phase 2 - Real-Time Monitoring (Quick)"
echo "================================================"
cwa monitor-realtime "$MODEL_NAME" \
  --detection-algorithms "statistical,gradient" \
  --circuit-breaker-config "auto" \
  --latency-target 2.0 \
  --output-dir "$OUTPUT_BASE/realtime_monitoring"
echo "✅ Real-time monitoring complete"
echo ""

echo "🎮 Step 6: Phase 2 - Game Theory (Nash Only)"
echo "============================================"
cwa analyze-game-theory "$MODEL_NAME" \
  --game-types "nash_equilibrium" \
  --max-players 20 \
  --strategy-space-size 5 \
  --convergence-threshold 1e-4 \
  --output-dir "$OUTPUT_BASE/game_theory"
echo "✅ Game theory analysis complete"
echo ""

echo "🔄 Step 7: Phase 2 - Transfer Analysis (Geometric)"
echo "================================================="
cwa analyze-transfer "$MODEL_NAME" "$COMPARE_MODEL" \
  --mapping-strategies "geometric" \
  --similarity-threshold 0.6 \
  --output-dir "$OUTPUT_BASE/transfer_analysis"
echo "✅ Transfer analysis complete"
echo ""

echo "✅ Step 8: Super Weight Validation (Sample)"
echo "==========================================="
# Use sample coordinates for validation
cwa validate-super-weights "$MODEL_NAME" \
  --coordinates "[(0, 'transformer.h.0.mlp.c_fc', [50, 100])]" \
  --perplexity-threshold 50 \
  --export-results \
  --output-dir "$OUTPUT_BASE/validation"
echo "✅ Validation complete"
echo ""

echo "📚 Step 9: Quick Publication Package"
echo "===================================="
cwa research-extract "$MODEL_NAME" \
  --focus "mlp-components" \
  --export-format "publication-ready" \
  --output-dir "$OUTPUT_BASE/publication"
echo "✅ Publication package complete"
echo ""

echo "🎉 SMALL MODEL TEST WORKFLOW COMPLETE!"
echo "======================================"
echo ""
echo "📁 Results located in: $OUTPUT_BASE/"
echo ""
echo "📊 Quick Analysis Summary:"
echo "  🛡️  Traditional Security: $OUTPUT_BASE/traditional_security/"
echo "  🔍 Ensemble Discovery: $OUTPUT_BASE/ensemble_discovery/"
echo "  🌊 Spectral Analysis: $OUTPUT_BASE/spectral_analysis/"
echo "  🕰️  Real-time Monitoring: $OUTPUT_BASE/realtime_monitoring/"
echo "  🎮 Game Theory: $OUTPUT_BASE/game_theory/"
echo "  🔄 Transfer Analysis: $OUTPUT_BASE/transfer_analysis/"
echo "  ✅ Validation: $OUTPUT_BASE/validation/"
echo "  📚 Publication: $OUTPUT_BASE/publication/"
echo ""
echo "⚡ Performance optimized for small models:"
echo "  ✓ Reduced complexity parameters"
echo "  ✓ Faster convergence thresholds"
echo "  ✓ Limited analysis scope for speed"
echo "  ✓ Core functionality demonstration"
echo ""
echo "🚀 All capabilities tested successfully on small model!"

# Generate summary
cat > "$OUTPUT_BASE/SMALL_MODEL_SUMMARY.md" << EOF
# Small Model Test Summary

**Model**: $MODEL_NAME
**Comparison Model**: $COMPARE_MODEL
**Test Date**: $(date)
**Execution Time**: Fast-optimized configuration

## Tests Completed

✅ **Traditional Security**: Attack/defense pipeline
✅ **Phase 1 Ensemble**: 4-method super weight discovery
✅ **Phase 1 Spectral**: PAC-Bayesian vulnerability analysis
✅ **Phase 2 Monitoring**: Real-time security with circuit breakers
✅ **Phase 2 Game Theory**: Nash equilibrium strategic analysis
✅ **Phase 2 Transfer**: Cross-architecture mapping analysis
✅ **Validation**: Super weight perplexity testing
✅ **Publication**: Research-ready output generation

## Optimizations Applied

- **Reduced Parameters**: Faster execution for testing
- **Core Algorithms**: Essential functionality without full complexity
- **Quick Convergence**: Lower thresholds for demonstration
- **Limited Scope**: Focused analysis for validation

## Next Steps

For production research, use:
- \`./run_complete_integrated_workflow.sh\` for full analysis
- Larger models for comprehensive results
- Full parameter sets for publication-quality analysis

EOF

echo ""
echo "📋 Test summary saved to: $OUTPUT_BASE/SMALL_MODEL_SUMMARY.md"