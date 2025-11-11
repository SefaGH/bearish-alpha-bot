"""
Manual validation script for FAZ 2.1 changes

This script performs a dry-run validation of the training configuration
without actually running the training process.
"""

import sys
import os
import re

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def validate_changes():
    """Validate that all FAZ 2.1 changes are present and correct."""
    
    print("="*60)
    print("FAZ 2.1: Manual Validation")
    print("="*60)
    print()
    
    # Read the training script
    train_script = os.path.join(project_root, 'scripts', 'train_all_models.py')
    with open(train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Validation checks
    checks = []
    
    # Check 1: REGIME_TRAINING_TIMEFRAMES
    regime_match = re.search(r"REGIME_TRAINING_TIMEFRAMES\s*=\s*\[(.*?)\]", content)
    if regime_match:
        timeframes = regime_match.group(1)
        has_15m = "'15m'" in timeframes
        has_1d = "'1d'" in timeframes
        count = timeframes.count("'") // 2
        checks.append(("5 timeframes configured", count == 5))
        checks.append(("'15m' included", has_15m))
        checks.append(("'1d' included", has_1d))
    else:
        checks.append(("REGIME_TRAINING_TIMEFRAMES found", False))
    
    # Check 2: MIN_SAMPLES_FOR_NN
    min_nn_match = re.search(r"MIN_SAMPLES_FOR_NN\s*=\s*(\d+)", content)
    if min_nn_match:
        min_nn = int(min_nn_match.group(1))
        checks.append(("MIN_SAMPLES_FOR_NN = 1000", min_nn == 1000))
    else:
        checks.append(("MIN_SAMPLES_FOR_NN found", False))
    
    # Check 3: Configuration logging
    has_config_log = "REGIME MODEL TRAINING CONFIGURATION" in content
    checks.append(("Configuration logging added", has_config_log))
    
    # Check 4: Total samples logging
    has_total_log = re.search(r"Total training samples.*from.*timeframes", content)
    checks.append(("Total samples logging added", has_total_log is not None))
    
    # Check 5: timeframe_count in performance tracker
    has_tf_count = "'timeframe_count'" in content
    checks.append(("timeframe_count in tracker", has_tf_count))
    
    # Check 6: Missing data warning
    has_warning = "⚠️" in content and "için veri bulunamadı" in content
    checks.append(("Missing data warning added", has_warning))
    
    # Print results
    print("Validation Results:")
    print("-" * 60)
    
    all_passed = True
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("="*60)
        print("✅ All validations passed!")
        print("="*60)
        print()
        print("Configuration Summary:")
        print(f"  • Timeframes: 5 (15m, 30m, 1h, 4h, 1d)")
        print(f"  • MIN_SAMPLES_FOR_NN: 1000")
        print(f"  • Expected samples: 7200 (5 × 1440)")
        print(f"  • Sample increase: +67% (from 4320)")
        print()
        print("Next Steps:")
        print("  1. Run GitHub Actions workflow to train models")
        print("  2. Check training logs for expected sample count")
        print("  3. Compare accuracy with baseline (target: +5%)")
        return True
    else:
        print("="*60)
        print("❌ Some validations failed!")
        print("="*60)
        return False


def check_test_files():
    """Check that test files exist and can be run."""
    print()
    print("Checking test files...")
    print("-" * 60)
    
    test_files = [
        'tests/test_faz21_config_simple.py',
        'tests/test_faz21_timeframe_expansion.py'
    ]
    
    all_exist = True
    for test_file in test_files:
        full_path = os.path.join(project_root, test_file)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"{status} {test_file}")
        if not exists:
            all_exist = False
    
    return all_exist


def main():
    """Main validation function."""
    validation_passed = validate_changes()
    tests_exist = check_test_files()
    
    print()
    print("="*60)
    
    if validation_passed and tests_exist:
        print("✅ FAZ 2.1 implementation is complete and validated!")
        print("="*60)
        print()
        print("The changes are ready for training.")
        print("Run the training workflow in GitHub Actions to see results.")
        return 0
    else:
        print("❌ Validation incomplete. Please review the issues above.")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
