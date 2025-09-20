#!/bin/bash
# Phase 1 Advanced Research Workflow Script
# Run comprehensive information-theoretic weight analysis

set -e  # Exit on error

MODEL_NAME="gpt2"  # Change this to your preferred model
OUTPUT_BASE="phase1_analysis_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Phase 1 Advanced Research Workflow"
echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_BASE"

# Create base output directory
mkdir -p "$OUTPUT_BASE"

echo ""
echo "📊 Step 1: System Information"
cwa info

echo ""
echo "🔍 Step 2: Multi-Method Ensemble Discovery"
cwa extract-critical-weights "$MODEL_NAME" \
  --mode "super_weight_discovery" \
  --top-k-percent 0.001 \
  --layer-focus "early" \
  --output-dir "$OUTPUT_BASE/step2_ensemble_discovery"

echo ""
echo "🌊 Step 3: Information-Theoretic Spectral Analysis"
cwa spectral-analysis "$MODEL_NAME" \
  --analysis-types "signatures,transitions,stability,correlations" \
  --include-pac-bounds \
  --confidence-level 0.95 \
  --output-dir "$OUTPUT_BASE/step3_spectral_analysis"

echo ""
echo "🛡️ Step 4: Traditional Security Pipeline"
cwa run-complete-pipeline "$MODEL_NAME" \
  --attack-methods "fgsm,pgd,bit_flip" \
  --protection-methods "weight_redundancy,checksums" \
  --output-dir "$OUTPUT_BASE/step4_security_pipeline"

echo ""
echo "✅ Step 5: Cross-Validate Discovered Weights"
# Check if coordinates file exists
COORDS_FILE="$OUTPUT_BASE/step2_ensemble_discovery/discovered_coordinates.txt"
if [ -f "$COORDS_FILE" ]; then
    cwa validate-super-weights "$MODEL_NAME" \
      --coordinates "$(cat $COORDS_FILE)" \
      --perplexity-threshold 100 \
      --export-results \
      --output-dir "$OUTPUT_BASE/step5_validation"
else
    echo "⚠️  Coordinates file not found, using known super weights for validation"
    # Use known Llama-7B coordinates as example
    cwa validate-super-weights "$MODEL_NAME" \
      --coordinates "[(2, 'mlp.down_proj', [3968, 7003])]" \
      --perplexity-threshold 100 \
      --export-results \
      --output-dir "$OUTPUT_BASE/step5_validation"
fi

echo ""
echo "📚 Step 6: Generate Publication-Ready Research"
cwa research-extract "$MODEL_NAME" \
  --focus "comprehensive" \
  --export-format "publication-ready" \
  --include-metadata \
  --output-dir "$OUTPUT_BASE/step6_publication"

echo ""
echo "🎉 Phase 1 Workflow Complete!"
echo ""
echo "📁 Results located in: $OUTPUT_BASE/"
echo "📊 Key outputs:"
echo "  - Ensemble discovery: $OUTPUT_BASE/step2_ensemble_discovery/"
echo "  - Spectral analysis: $OUTPUT_BASE/step3_spectral_analysis/"
echo "  - Security pipeline: $OUTPUT_BASE/step4_security_pipeline/"
echo "  - Validation results: $OUTPUT_BASE/step5_validation/"
echo "  - Publication materials: $OUTPUT_BASE/step6_publication/"
echo ""
echo "🔬 Ready for Phase 2 implementation!"