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

