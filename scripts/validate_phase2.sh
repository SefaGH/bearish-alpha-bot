#!/bin/bash
# Phase 2 Validation Script
# Runs all Phase 2 tests to validate implementation

set -e

echo "========================================================================"
echo " PHASE 2: MULTI-SYMBOL TRADING & SIGNAL ACCEPTANCE VALIDATION"
echo "========================================================================"
echo ""
echo "Running comprehensive validation tests..."
echo ""

# Change to repo root
cd "$(dirname "$0")/.."

# Test 1: Phase 2 Requirements
echo "▶ Test 1: Phase 2 Requirements Validation"
echo "------------------------------------------------------------------------"
python tests/validate_phase2_requirements.py
echo ""

# Test 2: Symbol-Specific Thresholds
echo "▶ Test 2: Symbol-Specific Thresholds"
echo "------------------------------------------------------------------------"
python tests/test_symbol_specific_thresholds.py
echo ""

# Test 3: Duplicate Prevention (includes time delays)
echo "▶ Test 3: Duplicate Prevention with Phase 2 Settings"
echo "------------------------------------------------------------------------"
echo "⏱️  Note: This test takes ~22 seconds due to cooldown validation"
python tests/test_duplicate_prevention_phase2.py
echo ""

echo "========================================================================"
echo " ✅ ALL PHASE 2 VALIDATION TESTS COMPLETED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Phase 2 Features Validated:"
echo "  ✅ Duplicate prevention optimized (0.05%, 20s cooldown)"
echo "  ✅ Multi-symbol trading (BTC/ETH/SOL with custom thresholds)"
echo "  ✅ Debug logging comprehensive ([STR-DEBUG] format)"
echo "  ✅ Symbol-independent tracking"
echo "  ✅ Price delta bypass working"
echo ""
echo "Documentation:"
echo "  📖 docs/PHASE2_MULTI_SYMBOL_TRADING.md"
echo "  📖 PHASE2_IMPLEMENTATION_SUMMARY.md"
echo ""
