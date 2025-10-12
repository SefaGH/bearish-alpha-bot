import sys
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.sizing import position_size_usdt

def test_long_uses_real_distance_when_stop_below_entry():
    entry, stop, risk = 100.0, 99.0, 10.0  # distance 1.0
    qty = position_size_usdt(entry, stop, risk, side='long')
    assert math.isclose(qty, risk / (entry - stop), rel_tol=1e-9)

def test_short_uses_real_distance_when_stop_above_entry():
    entry, stop, risk = 100.0, 101.0, 10.0  # distance 1.0
    qty = position_size_usdt(entry, stop, risk, side='short')
    assert math.isclose(qty, risk / (stop - entry), rel_tol=1e-9)

def test_min_distance_floor_applies_when_stop_on_wrong_side_long():
    entry, stop, risk = 100.0, 101.0, 10.0  # wrong side -> floor 0.1% of entry
    qty = position_size_usdt(entry, stop, risk, side='long')
    assert math.isclose(qty, risk / (entry * 0.001), rel_tol=1e-9)

def test_min_distance_floor_applies_when_stop_on_wrong_side_short():
    entry, stop, risk = 100.0, 99.0, 10.0  # wrong side -> floor 0.1% of entry
    qty = position_size_usdt(entry, stop, risk, side='short')
    assert math.isclose(qty, risk / (entry * 0.001), rel_tol=1e-9)