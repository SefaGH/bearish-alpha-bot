from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_mr_promote_z_tuning_script_includes_trade_labeled_sweep(tmp_path: Path):
    log_path = tmp_path / "sample.log"
    out_json = tmp_path / "out.json"

    req = {
        "event": "strategy_recheck_request",
        "pending_id": "p1",
        "symbol": "BTC/USDT:USDT",
        "side": "short",
        "condition_data": {"near": "upper", "adx": 10.0},
        "check_detail": {"fast_watch": {"touch_confirmed": True, "dist_to_band_bps": 1.0}},
    }
    ev = {
        "event": "mr_recheck_eval",
        "pending_id": "p1",
        "symbol": "BTC/USDT:USDT",
        "side": "short",
        "near": "upper",
        "z": 2.3,
        "dist_to_trigger_bps": 1.0,
        "action": "HOLD",
        "primary_gate_reason": "in_band",
    }
    closed = {
        "event": "TRADE_CLOSED",
        "strategy": "mean_reversion",
        "strategy_name": "mean_reversion",
        "symbol": "BTC/USDT:USDT",
        "side": "SELL",
        "pnl_usd": 12.0,
        "rr_achieved": 1.4,
        "exit_reason": "take_profit",
        "entry_metadata": {
            "signal_id": "sig-1",
            "promotion_override": {
                "candidate": True,
                "applied": True,
                "near": "upper",
                "touch_confirmed": True,
                "dist_bps": 1.0,
                "z": 2.3,
                "adx": 10.0,
            },
        },
    }

    log_path.write_text(
        "\n".join(
            [
                f"x strategy_recheck_request {json.dumps(req)}",
                f"x mr_recheck_eval {json.dumps(ev)}",
                f"x TRADE_CLOSED {json.dumps(closed)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/analyze_mr_promote_z_tuning.py",
        "--log",
        str(log_path),
        "--glob",
        "",
        "--thresholds",
        "2.0,2.2",
        "--touch-policy",
        "required",
        "--max-dist-bps",
        "2.0",
        "--max-adx",
        "20.0",
        "--out-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    cov = payload.get("trade_closed_coverage", {})
    assert cov.get("total_trade_closed_in_scope") == 1
    assert cov.get("trade_base_eligible_count") == 1

    trade_sweep = payload.get("trade_sweep", {})
    assert trade_sweep.get("2.00", {}).get("pass_count") == 1
    assert trade_sweep.get("2.20", {}).get("pass_count") == 1
