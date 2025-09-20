
# Super Weight Analysis Research Summary

## Model: microsoft/DialoGPT-small
## Analysis Mode: super_weight_discovery
## Date: 2025-09-20 22:24:30

## Key Findings

**Analysis Status**: No weights to analyze
- **Total Critical Weights Discovered**: 0
- **Discovery Rate**: 0.000000%
- **Average Sensitivity Score**: 0.000000
- **Layer Coverage**: 0 layers

## Layer Distribution

## Validation Results
- **Validated Super Weights**: 0

## Methodology
- Activation threshold: 1000.0
- Top-K percentage: 0.001%
- Layer focus: early
- Sensitivity threshold: 0.7

## Research Notes
This analysis uses PhD-level methodology for super weight discovery based on:
1. Activation magnitude monitoring (>1e3 threshold)
2. Hessian-based sensitivity scoring
3. 100× perplexity increase validation
4. Focus on early transformer layers (0-3)
5. Emphasis on mlp.down_proj components
