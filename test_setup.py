#!/usr/bin/env python3
"""
Quick test to verify Phase 1 & 2 modules are working
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test that all Phase 1 & 2 modules can be imported."""
    try:
        print("🔍 Testing Phase 1 imports...")
        from src.cwa.theory.information_geometry import InformationGeometricAnalyzer
        from src.cwa.sensitivity.spectral_analyzer import SpectralVulnerabilityAnalyzer
        from src.cwa.research.super_weight_analyzer import SuperWeightAnalyzer
        print("✅ Phase 1 modules imported successfully")

        print("🔍 Testing Phase 2 imports...")
        from src.cwa.monitoring.realtime_monitor import RealtimeSecurityMonitor
        from src.cwa.game_theory.neurogame_analyzer import NeuroGameAnalyzer
        from src.cwa.game_theory.game_theoretic_analyzer import GameTheoreticWeightAnalyzer
        from src.cwa.transfer.transfer_analyzer import TransferAnalyzer
        print("✅ Phase 2 modules imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_torch_setup():
    """Test PyTorch and CUDA setup."""
    print(f"🔍 PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔍 CUDA devices: {torch.cuda.device_count()}")
        print(f"🔍 Current device: {torch.cuda.get_device_name()}")
    return True

def test_cli_entry():
    """Test that CLI entry point exists."""
    try:
        from src.cwa.cli.main import app
        print("✅ CLI entry point accessible")
        return True
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Critical Weight Analysis - Setup Verification")
    print("=" * 60)

    all_passed = True

    # Test imports
    all_passed &= test_imports()
    print()

    # Test PyTorch
    all_passed &= test_torch_setup()
    print()

    # Test CLI
    all_passed &= test_cli_entry()
    print()

    if all_passed:
        print("🎉 All tests passed! Ready to run Phase 1 & 2 workflows")
        print()
        print("📋 Next steps:")
        print("1. Install the package: pip install -e .")
        print("2. Run workflow: ./run_phase1_workflow.sh")
        print("3. Or run individual commands: cwa --help")
    else:
        print("❌ Some tests failed. Check imports and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()