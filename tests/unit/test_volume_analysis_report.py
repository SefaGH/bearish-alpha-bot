import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.analyze_volume_buckets import aggregate_trades, load_trades_from_files


FIXTURE = Path(__file__).parent.parent / "fixtures" / "volume" / "sample_trades.jsonl"


def test_volume_bucket_aggregation_counts_and_means():
    trades = load_trades_from_files([FIXTURE], run_id="run-123")
    report = aggregate_trades(trades)

    overall = report["overall"]
    assert overall["n_trades"] == 5
    assert overall["n_wins"] == 3
    assert overall["n_losses"] == 2
    assert overall["win_rate"] == 0.6
    assert overall["avg_pnl"] == 26.0
    assert overall["avg_rr"] == pytest.approx(1.06)  # (2.0 -0.5 + 1.2 -0.4 + 3.0)/5

    high = report["by_volume_bucket"]["HIGH"]
    assert high["n_trades"] == 2
    assert high["n_wins"] == 1
    assert high["n_losses"] == 1
    assert high["win_rate"] == 0.5
    assert high["avg_pnl"] == 12.5
    assert high["avg_rr"] == pytest.approx(0.75)

    normal = report["by_volume_bucket"]["NORMAL"]
    assert normal["n_trades"] == 2
    assert normal["n_wins"] == 1
    assert normal["n_losses"] == 1
    assert normal["win_rate"] == 0.5
    assert normal["avg_pnl"] == 2.5
    assert normal["avg_rr"] == pytest.approx(0.4)

    extreme = report["by_volume_bucket"]["EXTREME"]
    assert extreme["n_trades"] == 1
    assert extreme["n_wins"] == 1
    assert extreme["n_losses"] == 0
    assert extreme["win_rate"] == 1.0
    assert extreme["avg_pnl"] == 100.0
    assert extreme["avg_rr"] == pytest.approx(3.0)

    strat_high_alpha = report["by_bucket_and_strategy"]["HIGH"]["alpha"]
    assert strat_high_alpha["n_trades"] == 2
    assert strat_high_alpha["n_wins"] == 1
    assert strat_high_alpha["n_losses"] == 1
    assert strat_high_alpha["win_rate"] == 0.5
    assert strat_high_alpha["avg_pnl"] == 12.5
    assert strat_high_alpha["avg_rr"] == pytest.approx(0.75)

    strat_normal_beta = report["by_bucket_and_strategy"]["NORMAL"]["beta"]
    assert strat_normal_beta["n_trades"] == 2
    assert strat_normal_beta["win_rate"] == 0.5
    assert strat_normal_beta["avg_pnl"] == 2.5
    assert strat_normal_beta["avg_rr"] == pytest.approx(0.4)

    strat_extreme_alpha = report["by_bucket_and_strategy"]["EXTREME"]["alpha"]
    assert strat_extreme_alpha["n_trades"] == 1
    assert strat_extreme_alpha["win_rate"] == 1.0
    assert strat_extreme_alpha["avg_pnl"] == 100.0
    assert strat_extreme_alpha["avg_rr"] == pytest.approx(3.0)


def test_timeframe_filtering_keeps_matching_records_only():
    trades = load_trades_from_files([FIXTURE], run_id="run-123", timeframe="5m")
    assert len(trades) == 5
    trades_none = load_trades_from_files([FIXTURE], run_id="run-123", timeframe="1h")
    assert len(trades_none) == 0
