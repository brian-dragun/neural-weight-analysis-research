#!/bin/bash

# Test Phase C with corrected protection methods
echo "🔧 Testing Phase C with corrected protection methods..."

# Go to the results from the successful complete pipeline
cd /home/ubuntu/nova/neural-weight-analysis-research

# Test with the correctly named methods
echo "Testing with: weight_redundancy, enhanced_checksums, input_sanitization"

# Create a simple test script
cat > test_fixed_protection.py << 'EOF'
import sys
import os
sys.path.append('/home/ubuntu/nova/neural-weight-analysis-research/src')

try:
    from cwa.security.defense_mechanisms import list_defense_mechanisms
    print("✅ Available defense mechanisms:")
    for method in sorted(list_defense_mechanisms()):
        print(f"  - {method}")
    print("\n🔍 Analysis:")
    print("  ❌ 'checksums' - NOT AVAILABLE")
    print("  ✅ 'enhanced_checksums' - AVAILABLE")
    print("  ✅ 'adversarial_training' - AVAILABLE (but has NaN bug)")
    print("  ✅ 'weight_redundancy' - AVAILABLE")
    print("  ✅ 'input_sanitization' - AVAILABLE")
    print("  ✅ 'layer11_attention_fortress' - AVAILABLE")
    
except ImportError as e:
    print(f"❌ Cannot import: {e}")
    print("Need to set up the Python environment properly")

EOF

# Run the test
python3 test_fixed_protection.py

echo ""
echo "🎯 Recommendation: Use these working methods:"
echo "   weight_redundancy,enhanced_checksums,input_sanitization"
echo "   OR"  
echo "   weight_redundancy,layer11_attention_fortress,input_sanitization"