import json
from pathlib import Path

import pytest

from scripts.canary_go_no_go_report import (
    Thresholds,
    _apply_filters,
    _compute_metrics,
    _load_events,
)


def _write_lines(path: Path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_canary_report_parses_and_computes_extended_metrics(tmp_path: Path):
    log_file = tmp_path / "live_trading_test.log"
    trace_1 = {
        "event": "order_decision_trace",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "policy_applied": True,
        "policy_decision": "applied",
        "fallback_reason": None,
        "effective_order_type": "limit",
        "bucket": "EXTREME",
        "atr_age_ms": 1200,
        "env_forced_order_type": None,
    }
    trace_2 = {
        "event": "order_decision_trace",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "policy_applied": False,
        "policy_decision": "skipped:missing_atr_force_market",
        "fallback_reason": "missing_atr_force_market",
        "effective_order_type": "market",
        "bucket": "NORMAL",
        "atr_age_ms": 100,
        "env_forced_order_type": None,
    }
    outcome_1 = {
        "event": "order_decision_outcome",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "success": True,
        "reason": None,
        "fallback_reason": None,
        "entry_slippage_bps": 3.0,
        "entry_notional_usd": 1000.0,
        "time_to_fill_ms": 120.0,
    }
    outcome_2 = {
        "event": "order_decision_outcome",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "success": False,
        "reason": "ABORT:NO_FILL_TIMEOUT",
        "fallback_reason": "limit_timeout_market_fallback_disabled:extreme_bucket",
        "entry_slippage_bps": 8.0,
        "entry_notional_usd": 5000.0,
        "time_to_fill_ms": 480.0,
    }
    trade_1 = {
        "event": "TRADE_CLOSED",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "stop_overshoot_bps": 0.0,
        "rr_after_fill": 1.20,
        "rr_achieved": 1.00,
    }
    trade_2 = {
        "event": "TRADE_CLOSED",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "stop_overshoot_bps": 12.0,
        "rr_after_fill": 1.40,
        "rr_achieved": 1.10,
    }

    lines = [
        f"order_decision_trace {trace_1}",
        f"order_decision_trace {trace_2}",
        f"order_decision_outcome {outcome_1}",
        f"order_decision_outcome {outcome_2}",
        f"TRADE_CLOSED {json.dumps(trade_1)}",
        f"TRADE_CLOSED {json.dumps(trade_2)}",
        "[RECON-WATCHDOG] stale_removed=0 orphans_detected=0 orphans_adopted=0 active_positions=1",
    ]
    _write_lines(log_file, lines)

    traces, outcomes, mgr, trades, recon = _load_events([log_file])
    traces, outcomes, mgr, trades = _apply_filters(
        traces=traces,
        outcomes=outcomes,
        mgr=mgr,
        trades=trades,
        symbol="BTC/USDT:USDT",
        strategy="adaptive_ob",
    )
    report = _compute_metrics(
        traces=traces,
        outcomes=outcomes,
        trades=trades,
        recon_events=recon,
        thresholds=Thresholds(
            smart_entry_applied_min_rate=0.40,
            missing_atr_force_market_max_rate=0.60,
            extreme_market_max_rate=0.10,
            atr_age_threshold_ms=5000,
            missed_fill_increase_max_pct=20.0,
            max_stop_abort_cancel_unconfirmed=0,
            max_recon_orphans_detected=0,
            max_recon_stale_removed=0,
            max_recon_orphans_adopted=0,
            require_recon_watchdog_events=True,
        ),
        baseline=None,
    )

    m = report["metrics"]
    assert m["total_traces"] == 2
    assert m["smart_entry_applied_rate"] == pytest.approx(0.5, rel=1e-9)
    assert m["missing_atr_force_market_count"] == 1
    assert m["abort_no_fill_timeout_count"] == 1
    assert m["entry_slippage_trade_weighted_p90_bps"] == pytest.approx(7.5, rel=1e-9)
    assert m["entry_slippage_trade_weighted_p95_bps"] == pytest.approx(7.75, rel=1e-9)
    assert m["entry_slippage_notional_weighted_p90_bps"] == pytest.approx(8.0, rel=1e-9)
    assert m["time_to_fill_ms_p50"] == pytest.approx(300.0, rel=1e-9)
    assert m["time_to_fill_ms_p90"] == pytest.approx(444.0, rel=1e-9)
    assert m["time_to_fill_ms_p95"] == pytest.approx(462.0, rel=1e-9)
    assert m["stop_overshoot_p90_bps"] == pytest.approx(10.8, rel=1e-9)
    assert m["stop_overshoot_p95_bps"] == pytest.approx(11.4, rel=1e-9)
    assert m["planned_vs_realized_rr_drift_abs_p90"] == pytest.approx(0.29, rel=1e-9)
    assert m["planned_vs_realized_rr_drift_abs_p95"] == pytest.approx(0.295, rel=1e-9)
    assert m["recon_watchdog_events_count"] == 1
    assert m["recon_stale_removed_total"] == 0
    assert m["recon_orphans_detected_total"] == 0
    assert m["recon_orphans_adopted_total"] == 0
    assert m["abort_stop_hit_cancel_unconfirmed_count"] == 0
    assert m["abort_stop_hit_cancel_unconfirmed_rate"] == pytest.approx(0.0, rel=1e-9)
    assert m["filled_during_stop_abort_count"] == 0
    assert m["filled_during_stop_abort_rate"] == pytest.approx(0.0, rel=1e-9)

    gates = report["go_no_go"]
    assert gates["stop_abort_cancel_unconfirmed_ok"] is True
    assert gates["recon_events_present"] is True
    assert gates["recon_orphans_detected_ok"] is True
    assert gates["recon_stale_removed_ok"] is True
    assert gates["recon_orphans_adopted_ok"] is True


def test_canary_report_fails_when_recon_orphans_exceed_threshold(tmp_path: Path):
    log_file = tmp_path / "live_trading_recon_fail.log"
    trace = {
        "event": "order_decision_trace",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "policy_applied": True,
        "policy_decision": "applied",
        "fallback_reason": None,
        "effective_order_type": "limit",
        "bucket": "NORMAL",
        "atr_age_ms": 100,
        "env_forced_order_type": None,
    }
    outcome = {
        "event": "order_decision_outcome",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "success": True,
        "reason": None,
        "fallback_reason": None,
        "entry_slippage_bps": 1.0,
        "entry_notional_usd": 100.0,
        "time_to_fill_ms": 100.0,
    }
    trade = {
        "event": "TRADE_CLOSED",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "stop_overshoot_bps": 1.0,
        "planned_vs_realized_rr_drift": 0.1,
    }
    lines = [
        f"order_decision_trace {trace}",
        f"order_decision_outcome {outcome}",
        f"TRADE_CLOSED {json.dumps(trade)}",
        "[RECON-WATCHDOG] stale_removed=0 orphans_detected=2 orphans_adopted=0 active_positions=1",
    ]
    _write_lines(log_file, lines)

    traces, outcomes, mgr, trades, recon = _load_events([log_file])
    traces, outcomes, mgr, trades = _apply_filters(
        traces=traces,
        outcomes=outcomes,
        mgr=mgr,
        trades=trades,
        symbol="BTC/USDT:USDT",
        strategy="adaptive_ob",
    )
    report = _compute_metrics(
        traces=traces,
        outcomes=outcomes,
        trades=trades,
        recon_events=recon,
        thresholds=Thresholds(
            smart_entry_applied_min_rate=0.1,
            missing_atr_force_market_max_rate=1.0,
            extreme_market_max_rate=1.0,
            atr_age_threshold_ms=5000,
            missed_fill_increase_max_pct=100.0,
            max_stop_abort_cancel_unconfirmed=0,
            max_recon_orphans_detected=0,
            max_recon_stale_removed=5,
            max_recon_orphans_adopted=0,
            require_recon_watchdog_events=True,
        ),
        baseline=None,
    )

    assert report["status"] == "NO_GO"
    assert report["go_no_go"]["recon_orphans_detected_ok"] is False
    assert "recon_orphans_detected_ok" in (report.get("failed_gates") or [])


def test_canary_report_flags_stop_abort_cancel_unconfirmed_signal(tmp_path: Path):
    log_file = tmp_path / "live_trading_stop_abort_signal.log"
    trace = {
        "event": "order_decision_trace",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "policy_applied": True,
        "policy_decision": "applied",
        "fallback_reason": None,
        "effective_order_type": "limit",
        "bucket": "NORMAL",
        "atr_age_ms": 100,
        "env_forced_order_type": None,
    }
    outcome_1 = {
        "event": "order_decision_outcome",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "success": False,
        "reason": "ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED",
        "fallback_reason": None,
        "entry_slippage_bps": 1.0,
        "entry_notional_usd": 100.0,
        "time_to_fill_ms": 200.0,
    }
    outcome_2 = {
        "event": "order_decision_outcome",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "success": True,
        "reason": "FILLED_DURING_STOP_ABORT",
        "fallback_reason": None,
        "entry_slippage_bps": 1.0,
        "entry_notional_usd": 120.0,
        "time_to_fill_ms": 180.0,
    }
    trade = {
        "event": "TRADE_CLOSED",
        "symbol": "BTC/USDT:USDT",
        "strategy_name": "adaptive_ob",
        "stop_overshoot_bps": 1.0,
        "planned_vs_realized_rr_drift": 0.1,
    }
    lines = [
        f"order_decision_trace {trace}",
        f"order_decision_outcome {outcome_1}",
        f"order_decision_outcome {outcome_2}",
        f"TRADE_CLOSED {json.dumps(trade)}",
        "[RECON-WATCHDOG] stale_removed=0 orphans_detected=0 orphans_adopted=0 active_positions=1",
    ]
    _write_lines(log_file, lines)

    traces, outcomes, mgr, trades, recon = _load_events([log_file])
    traces, outcomes, mgr, trades = _apply_filters(
        traces=traces,
        outcomes=outcomes,
        mgr=mgr,
        trades=trades,
        symbol="BTC/USDT:USDT",
        strategy="adaptive_ob",
    )
    report = _compute_metrics(
        traces=traces,
        outcomes=outcomes,
        trades=trades,
        recon_events=recon,
        thresholds=Thresholds(
            smart_entry_applied_min_rate=0.1,
            missing_atr_force_market_max_rate=1.0,
            extreme_market_max_rate=1.0,
            atr_age_threshold_ms=5000,
            missed_fill_increase_max_pct=100.0,
            max_stop_abort_cancel_unconfirmed=0,
            max_recon_orphans_detected=0,
            max_recon_stale_removed=5,
            max_recon_orphans_adopted=0,
            require_recon_watchdog_events=True,
        ),
        baseline=None,
    )

    metrics = report["metrics"]
    assert metrics["abort_stop_hit_cancel_unconfirmed_count"] == 1
    assert metrics["abort_stop_hit_cancel_unconfirmed_rate"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["filled_during_stop_abort_count"] == 1
    assert metrics["filled_during_stop_abort_rate"] == pytest.approx(0.5, rel=1e-9)
    assert metrics["outcome_reason_counts"]["ABORT:STOP_HIT_BEFORE_ENTRY_CANCEL_UNCONFIRMED"] == 1
    assert metrics["outcome_reason_counts"]["FILLED_DURING_STOP_ABORT"] == 1

    gates = report["go_no_go"]
    assert gates["stop_abort_cancel_unconfirmed_ok"] is False
    assert "stop_abort_cancel_unconfirmed_ok" in (report.get("failed_gates") or [])
