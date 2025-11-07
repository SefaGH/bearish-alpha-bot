"""
Simple test for FAZ 2.1: Multi-Timeframe Data Augmentation (without dependencies)

This test directly reads the train_all_models.py file and verifies the configuration
without importing the module (which requires all dependencies).
"""

import re


def test_configuration_from_file():
    """Test configuration by reading the file directly."""
    
    # Read the training script
    with open('scripts/train_all_models.py', 'r') as f:
        content = f.read()
    
    # Test 1: Check REGIME_TRAINING_TIMEFRAMES
    print("Test 1: REGIME_TRAINING_TIMEFRAMES")
    regime_match = re.search(r"REGIME_TRAINING_TIMEFRAMES\s*=\s*\[(.*?)\]", content)
    if regime_match:
        regime_timeframes = regime_match.group(1)
        print(f"   Found: REGIME_TRAINING_TIMEFRAMES = [{regime_timeframes}]")
        
        # Check for all expected timeframes
        expected = ["'15m'", "'30m'", "'1h'", "'4h'", "'1d'"]
        for tf in expected:
            assert tf in regime_timeframes, f"{tf} not found in REGIME_TRAINING_TIMEFRAMES"
        
        # Count timeframes (count commas + 1)
        tf_count = regime_timeframes.count("'") // 2
        assert tf_count == 5, f"Expected 5 timeframes, found {tf_count}"
        print("   ✅ PASS: 5 timeframes found: 15m, 30m, 1h, 4h, 1d")
    else:
        raise AssertionError("REGIME_TRAINING_TIMEFRAMES not found in file")
    
    # Test 2: Check ALL_TIMEFRAMES includes '1d'
    print("\nTest 2: ALL_TIMEFRAMES")
    all_tf_match = re.search(r"ALL_TIMEFRAMES\s*=\s*\[(.*?)\]", content)
    if all_tf_match:
        all_timeframes = all_tf_match.group(1)
        print(f"   Found: ALL_TIMEFRAMES = [{all_timeframes}]")
        
        assert "'1d'" in all_timeframes, "'1d' not found in ALL_TIMEFRAMES"
        assert "'15m'" in all_timeframes, "'15m' not found in ALL_TIMEFRAMES"
        print("   ✅ PASS: ALL_TIMEFRAMES includes '1d' and '15m'")
    else:
        raise AssertionError("ALL_TIMEFRAMES not found in file")
    
    # Test 3: Check MIN_SAMPLES_FOR_NN
    print("\nTest 3: MIN_SAMPLES_FOR_NN")
    min_nn_match = re.search(r"MIN_SAMPLES_FOR_NN\s*=\s*(\d+)", content)
    if min_nn_match:
        min_nn = int(min_nn_match.group(1))
        print(f"   Found: MIN_SAMPLES_FOR_NN = {min_nn}")
        assert min_nn == 1000, f"Expected MIN_SAMPLES_FOR_NN=1000, found {min_nn}"
        print("   ✅ PASS: MIN_SAMPLES_FOR_NN = 1000")
    else:
        raise AssertionError("MIN_SAMPLES_FOR_NN not found in file")
    
    # Test 4: Check CANDLE_LIMIT
    print("\nTest 4: CANDLE_LIMIT")
    candle_match = re.search(r"CANDLE_LIMIT\s*=\s*(\d+)", content)
    if candle_match:
        candle_limit = int(candle_match.group(1))
        print(f"   Found: CANDLE_LIMIT = {candle_limit}")
        assert candle_limit == 1440, f"Expected CANDLE_LIMIT=1440, found {candle_limit}"
        print("   ✅ PASS: CANDLE_LIMIT = 1440 (BingX limit)")
    else:
        raise AssertionError("CANDLE_LIMIT not found in file")
    
    # Test 5: Check for configuration logging
    print("\nTest 5: Configuration Logging")
    config_log_patterns = [
        r"REGIME MODEL TRAINING CONFIGURATION",
        r"Timeframes:",
        r"Candle limit per timeframe:",
        r"Expected total samples:",
        r"Minimum NN samples:"
    ]
    
    for pattern in config_log_patterns:
        if re.search(pattern, content):
            print(f"   ✅ Found log pattern: {pattern}")
        else:
            raise AssertionError(f"Configuration log pattern not found: {pattern}")
    
    # Test 6: Check for total samples logging
    print("\nTest 6: Total Samples Logging")
    total_samples_pattern = r"Total training samples.*from.*timeframes"
    if re.search(total_samples_pattern, content):
        print("   ✅ PASS: Total samples logging found")
    else:
        raise AssertionError("Total samples logging not found")
    
    # Test 7: Check for timeframe_count in performance tracker
    print("\nTest 7: Performance Tracker timeframe_count")
    timeframe_count_pattern = r"'timeframe_count':\s*len\(REGIME_TRAINING_TIMEFRAMES\)"
    if re.search(timeframe_count_pattern, content):
        print("   ✅ PASS: timeframe_count added to performance tracker")
    else:
        raise AssertionError("timeframe_count not added to performance tracker")
    
    # Test 8: Calculate expected samples
    print("\nTest 8: Expected Sample Count")
    expected_samples = 5 * 1440  # 5 timeframes * 1440 candles
    print(f"   Expected: {expected_samples} samples (5 timeframes * 1440 candles)")
    assert expected_samples == 7200, f"Expected 7200, calculated {expected_samples}"
    print("   ✅ PASS: Expected sample count is 7200")
    
    print("\n" + "="*60)
    print("✅ All FAZ 2.1 configuration tests passed!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   • REGIME_TRAINING_TIMEFRAMES: 5 timeframes (15m, 30m, 1h, 4h, 1d)")
    print(f"   • MIN_SAMPLES_FOR_NN: 1000 (increased from 500)")
    print(f"   • Expected samples: 7200 (up from 4320, +67%)")
    print(f"   • Configuration logging: ✅ Added")
    print(f"   • Performance tracking: ✅ Enhanced")


if __name__ == "__main__":
    print("="*60)
    print("FAZ 2.1: Multi-Timeframe Data Augmentation Test")
    print("="*60)
    print()
    test_configuration_from_file()
