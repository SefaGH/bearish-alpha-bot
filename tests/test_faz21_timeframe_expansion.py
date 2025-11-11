"""
Test for FAZ 2.1: Multi-Timeframe Data Augmentation

This test verifies that the training configuration has been updated correctly
to use 5 timeframes instead of 3, and that MIN_SAMPLES_FOR_NN has been increased.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_regime_training_timeframes():
    """Test that REGIME_TRAINING_TIMEFRAMES contains exactly 5 timeframes."""
    from scripts.train_all_models import REGIME_TRAINING_TIMEFRAMES
    
    # Expected timeframes
    expected_timeframes = ['15m', '30m', '1h', '4h', '1d']
    
    # Check that we have 5 timeframes
    assert len(REGIME_TRAINING_TIMEFRAMES) == 5, \
        f"Expected 5 timeframes, got {len(REGIME_TRAINING_TIMEFRAMES)}"
    
    # Check that the timeframes are correct
    assert REGIME_TRAINING_TIMEFRAMES == expected_timeframes, \
        f"Expected {expected_timeframes}, got {REGIME_TRAINING_TIMEFRAMES}"
    
    print("✅ REGIME_TRAINING_TIMEFRAMES is correct: 5 timeframes")
    print(f"   Timeframes: {REGIME_TRAINING_TIMEFRAMES}")


def test_all_timeframes():
    """Test that ALL_TIMEFRAMES includes '1d' for data fetching."""
    from scripts.train_all_models import ALL_TIMEFRAMES
    
    # Check that '1d' is included
    assert '1d' in ALL_TIMEFRAMES, \
        f"'1d' should be in ALL_TIMEFRAMES for data fetching, got {ALL_TIMEFRAMES}"
    
    # Check that '15m' is included
    assert '15m' in ALL_TIMEFRAMES, \
        f"'15m' should be in ALL_TIMEFRAMES for regime training, got {ALL_TIMEFRAMES}"
    
    print("✅ ALL_TIMEFRAMES includes '1d' and '15m'")
    print(f"   Timeframes: {ALL_TIMEFRAMES}")


def test_min_samples_for_nn():
    """Test that MIN_SAMPLES_FOR_NN has been increased to 1000."""
    from scripts.train_all_models import MIN_SAMPLES_FOR_NN
    
    # Check that MIN_SAMPLES_FOR_NN is 1000
    assert MIN_SAMPLES_FOR_NN == 1000, \
        f"Expected MIN_SAMPLES_FOR_NN=1000, got {MIN_SAMPLES_FOR_NN}"
    
    print("✅ MIN_SAMPLES_FOR_NN is correct: 1000")


def test_expected_sample_count():
    """Test that the expected sample count is calculated correctly."""
    from scripts.train_all_models import REGIME_TRAINING_TIMEFRAMES, CANDLE_LIMIT
    
    # Calculate expected samples
    expected_samples = len(REGIME_TRAINING_TIMEFRAMES) * CANDLE_LIMIT
    
    # With 5 timeframes and 1440 candles per timeframe
    assert expected_samples == 7200, \
        f"Expected 7200 samples (5 * 1440), got {expected_samples}"
    
    print(f"✅ Expected sample count is correct: {expected_samples}")
    print(f"   ({len(REGIME_TRAINING_TIMEFRAMES)} timeframes * {CANDLE_LIMIT} candles)")


def test_timeframe_sequence():
    """Test that timeframes are in the correct order (ascending duration)."""
    from scripts.train_all_models import REGIME_TRAINING_TIMEFRAMES
    
    # Define the correct order
    expected_order = ['15m', '30m', '1h', '4h', '1d']
    
    # Check order
    assert REGIME_TRAINING_TIMEFRAMES == expected_order, \
        f"Timeframes should be in ascending order: {expected_order}, got {REGIME_TRAINING_TIMEFRAMES}"
    
    print("✅ Timeframes are in correct ascending order")


def test_candle_limit():
    """Test that CANDLE_LIMIT remains at 1440 (BingX limit)."""
    from scripts.train_all_models import CANDLE_LIMIT
    
    # Check that CANDLE_LIMIT is 1440
    assert CANDLE_LIMIT == 1440, \
        f"Expected CANDLE_LIMIT=1440 (BingX limit), got {CANDLE_LIMIT}"
    
    print("✅ CANDLE_LIMIT is correct: 1440 (BingX API limit)")


def test_min_samples_rf():
    """Test that MIN_SAMPLES_FOR_RF remains at 100."""
    from scripts.train_all_models import MIN_SAMPLES_FOR_RF
    
    # Check that MIN_SAMPLES_FOR_RF is 100
    assert MIN_SAMPLES_FOR_RF == 100, \
        f"Expected MIN_SAMPLES_FOR_RF=100, got {MIN_SAMPLES_FOR_RF}"
    
    print("✅ MIN_SAMPLES_FOR_RF is correct: 100")


if __name__ == "__main__":
    print("="*60)
    print("Testing FAZ 2.1: Multi-Timeframe Data Augmentation")
    print("="*60)
    print()
    
    # Run all tests
    test_regime_training_timeframes()
    test_all_timeframes()
    test_min_samples_for_nn()
    test_expected_sample_count()
    test_timeframe_sequence()
    test_candle_limit()
    test_min_samples_rf()
    
    print()
    print("="*60)
    print("✅ All FAZ 2.1 configuration tests passed!")
    print("="*60)
