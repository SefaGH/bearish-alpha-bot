#!/usr/bin/env python3
"""
Offline incident replay for Crash Guard (adaptive_ob).

Goal:
- Recreate Crash Guard decisions for a historical UTC window by:
  - parsing the live trading log for the target position IDs
  - fetching 1m OHLCV via ccxt
  - building 5m candles (with a simulated "forming" bar)
  - feeding a stub MarketDataPipeline into StrategyCoordinator
  - asserting decisions + reason-codes (panic_veto_no_reversal / stop_loss_cooldown_active)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from unittest.mock import AsyncMock, MagicMock

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml  # repo-local minimal yaml shim (avoids PyYAML dependency issues)

from src.core.ccxt_client import CcxtClient
from src.core.indicators import add_indicators, rsi as rsi_series
from src.core.strategy_coordinator import StrategyCoordinator
import src.core.strategy_coordinator as sc_mod


TS_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - \[(?P<logger>[^\]]+)\] - (?P<level>[A-Z]+) - (?P<msg>.*)$"
)

POS_OPEN_RE = re.compile(r"Position opened:\s+(?P<pos_id>pos_[^ ]+)")


def _parse_log_ts(ts_str: str) -> datetime:
    # Log timestamps are in UTC (consistent with embedded Z timestamps elsewhere in the log).
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _parse_iso_utc(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _floor(dt: datetime, freq: str) -> datetime:
    return pd.Timestamp(dt).floor(freq).to_pydatetime().replace(tzinfo=timezone.utc)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        if getattr(out.index, "tz", None) is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    except Exception:
        out.index = pd.to_datetime(out.index, utc=True)
    return out


def _iter_log_records(path: Path) -> Iterable[Tuple[datetime, str, str, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            m = TS_LINE_RE.match(line)
            if not m:
                continue
            ts = _parse_log_ts(m.group("ts"))
            yield ts, m.group("logger"), m.group("level"), m.group("msg")


@dataclass(frozen=True)
class IncidentPosition:
    position_id: str
    open_ts: datetime
    entry_price: Optional[float]
    volume_bucket_at_entry: Optional[str]
    trade_closed_payload: Optional[Dict[str, Any]]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh.read()) or {}


def _normalize_inline_list(value: Any) -> Any:
    """
    Repo-local YAML shim does not parse inline YAML lists like:
      key: ["A", "B"]
    Normalize a small subset of these fields via JSON decoding.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return value
    try:
        parsed = json.loads(s)
        return parsed
    except Exception:
        return value


def _make_coordinator(*, market_data_pipeline: Any, config: Dict[str, Any]) -> StrategyCoordinator:
    pm = MagicMock()
    pm.cfg = config or {}
    pm.performance_monitor = None
    pm.exchange_clients = {}
    pm.get_strategy_allocation.return_value = 0.0
    pm.get_open_positions_for_symbol.return_value = []
    pm.get_open_positions.return_value = {}

    rm = MagicMock()
    coordinator = StrategyCoordinator(pm, rm, market_data_pipeline=market_data_pipeline, config=config)

    # Keep the coordinator minimal: bypass heavy pipeline stages, focus on Crash Guard.
    coordinator._validate_signal_format = MagicMock(return_value={"valid": True})
    coordinator.validate_duplicate = MagicMock(return_value=(True, "OK"))
    coordinator._check_signal_conflicts = AsyncMock(return_value={"has_conflict": False})
    coordinator._assess_signal_risk = AsyncMock(return_value={"acceptable": True, "position_size": 10.0, "metrics": {}})
    coordinator._route_signal = MagicMock(return_value={})
    coordinator._generate_signal_id = MagicMock(return_value="sig_replay")
    coordinator.signal_queue = AsyncMock()
    coordinator.signal_queue.put.return_value = (True, None)

    return coordinator


class ReplayMarketDataPipeline:
    def __init__(self, *, df_1m_raw: pd.DataFrame, df_1m_ind: pd.DataFrame, df_5m_ind: pd.DataFrame):
        self._df_1m_raw = _ensure_utc_index(df_1m_raw)
        self._df_1m_ind = _ensure_utc_index(df_1m_ind)
        self._df_5m_ind = _ensure_utc_index(df_5m_ind)
        self._now: datetime = datetime.now(timezone.utc)

    def set_now(self, now: datetime) -> None:
        self._now = now.astimezone(timezone.utc)

    async def get_latest_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        exchange: str = None,
        limit: int = None,
        include_forming: bool = False,
        hybrid_policy: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        tf = str(timeframe or "").strip().lower()
        if tf not in {"1m", "5m"}:
            return None

        if tf == "1m":
            cutoff = pd.Timestamp(_floor(self._now, "1min")).tz_convert("UTC")
            df = self._df_1m_ind.loc[self._df_1m_ind.index < cutoff].copy()
            if df.empty:
                return df
            if limit is not None:
                df = df.tail(int(limit))
            return df

        # 5m view with optional forming candle aggregated from 1m bars.
        bucket_start = pd.Timestamp(_floor(self._now, "5min")).tz_convert("UTC")
        closed = self._df_5m_ind.loc[self._df_5m_ind.index < bucket_start].copy()
        if closed.empty:
            return closed

        if include_forming:
            minute_cutoff = pd.Timestamp(_floor(self._now, "1min")).tz_convert("UTC")
            part = self._df_1m_raw.loc[(self._df_1m_raw.index >= bucket_start) & (self._df_1m_raw.index < minute_cutoff)]
            if not part.empty:
                forming_row = {
                    "open": float(part["open"].iloc[0]),
                    "high": float(part["high"].max()),
                    "low": float(part["low"].min()),
                    "close": float(part["close"].iloc[-1]),
                    "volume": float(part["volume"].sum()),
                }
                forming_df = pd.DataFrame([forming_row], index=[bucket_start])
                merged = pd.concat([closed, forming_df])
                merged = merged[~merged.index.duplicated(keep="last")]

                # Mimic MarketDataPipeline._merge_forming_candle: recompute RSI only; keep other indicators from closed bars.
                merged["rsi"] = rsi_series(merged["close"])
                indicator_cols = [
                    c for c in closed.columns if c not in {"open", "high", "low", "close", "volume", "rsi"}
                ]
                for col in indicator_cols:
                    merged[col] = merged[col].ffill()
                df_out = merged
            else:
                df_out = closed
        else:
            df_out = closed

        if limit is not None:
            df_out = df_out.tail(int(limit))
        return df_out


class _FrozenDateTime(datetime):
    _frozen_now: datetime = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._frozen_now.replace(tzinfo=None)
        return cls._frozen_now.astimezone(tz)


class ReplayClock:
    def __init__(self):
        self._orig_datetime = sc_mod.datetime
        self._orig_time_time = sc_mod.time.time

    def set(self, now: datetime) -> None:
        dt = now.astimezone(timezone.utc)
        _FrozenDateTime._frozen_now = dt
        sc_mod.datetime = _FrozenDateTime  # type: ignore[assignment]
        sc_mod.time.time = lambda: float(dt.timestamp())  # type: ignore[assignment]

    def restore(self) -> None:
        sc_mod.datetime = self._orig_datetime  # type: ignore[assignment]
        sc_mod.time.time = self._orig_time_time  # type: ignore[assignment]


def _build_5m_from_1m(df1m: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_utc_index(df1m)
    if df.empty:
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample("5min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _compute_reversal_meta(df_1m_ind: pd.DataFrame, *, now: datetime, hook_delta: float) -> Dict[str, Any]:
    cutoff = pd.Timestamp(_floor(now, "1min")).tz_convert("UTC")
    df = _ensure_utc_index(df_1m_ind).loc[_ensure_utc_index(df_1m_ind).index < cutoff]
    if df is None or df.empty or len(df) < 2:
        return {"rsi_hook": False, "bull_candle": False, "reclaim": False, "reversal_tf": "1m", "note": "insufficient_1m"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    try:
        rsi_now = float(last.get("rsi"))
        rsi_prev = float(prev.get("rsi"))
    except Exception:
        rsi_now = None
        rsi_prev = None

    rsi_hook = False
    try:
        if rsi_now is not None and rsi_prev is not None:
            rsi_hook = (rsi_now - rsi_prev) >= float(hook_delta)
    except Exception:
        rsi_hook = False

    bull_candle = False
    try:
        bull_candle = bool(float(last.get("close")) > float(last.get("open")) and float(last.get("close")) > float(prev.get("close")))
    except Exception:
        bull_candle = False

    reclaim = False
    try:
        reclaim = bool(float(last.get("close")) > float(last.get("ema_fast")))
    except Exception:
        reclaim = False

    return {
        "rsi_hook": bool(rsi_hook),
        "bull_candle": bool(bull_candle),
        "reclaim": bool(reclaim),
        "reversal_tf": "1m",
    }


def _extract_positions_from_log(
    log_path: Path,
    *,
    target_position_ids: List[str],
    window_start: datetime,
    window_end: datetime,
) -> List[IncidentPosition]:
    targets = set(target_position_ids)
    open_ts: Dict[str, datetime] = {}
    trade_closed: Dict[str, Dict[str, Any]] = {}

    for ts, logger_name, level, msg in _iter_log_records(log_path):
        if ts < window_start or ts > window_end:
            continue

        m = POS_OPEN_RE.search(msg)
        if m:
            pid = m.group("pos_id")
            if pid in targets:
                open_ts[pid] = ts

        if msg.startswith("TRADE_CLOSED "):
            try:
                payload = json.loads(msg[len("TRADE_CLOSED ") :])
            except Exception:
                payload = None
            if isinstance(payload, dict):
                pid = payload.get("position_id")
                if isinstance(pid, str) and pid in targets:
                    trade_closed[pid] = payload

    out: List[IncidentPosition] = []
    for pid in target_position_ids:
        ot = open_ts.get(pid)
        if not ot:
            continue
        payload = trade_closed.get(pid)
        entry_px = None
        vb = None
        if isinstance(payload, dict):
            try:
                entry_px = float(payload.get("entry_price")) if payload.get("entry_price") is not None else None
            except Exception:
                entry_px = None
            vb = payload.get("volume_bucket_at_entry") or payload.get("volume_bucket")
        out.append(
            IncidentPosition(
                position_id=pid,
                open_ts=ot,
                entry_price=entry_px,
                volume_bucket_at_entry=str(vb).upper() if vb is not None else None,
                trade_closed_payload=payload,
            )
        )
    return out


def _infer_stop_to_next_entry_pairs(positions: List[IncidentPosition]) -> List[Tuple[IncidentPosition, IncidentPosition]]:
    # For this incident: consecutive stop-loss churn is best represented by pairing
    # sorted opens (pos1->pos2->pos3). We'll verify each predecessor has stop_loss TRADE_CLOSED.
    ordered = sorted(positions, key=lambda p: p.open_ts)
    pairs: List[Tuple[IncidentPosition, IncidentPosition]] = []
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        payload = a.trade_closed_payload or {}
        if str(payload.get("exit_reason") or "").lower() == "stop_loss":
            pairs.append((a, b))
    return pairs


async def _replay_entry_decision(
    coordinator: StrategyCoordinator,
    mdp: ReplayMarketDataPipeline,
    clock: ReplayClock,
    *,
    symbol: str,
    now: datetime,
    volume_bucket: str,
    reversal_meta: Dict[str, Any],
) -> Dict[str, Any]:
    mdp.set_now(now)
    clock.set(now)

    coordinator._enrich_signal = AsyncMock(
        return_value={
            "symbol": symbol,
            "side": "long",
            "entry": 1.0,
            "stop": 0.99,
            "target": 1.01,
            "rr_ratio": 2.0,
            "volume_bucket": volume_bucket,
            "meta": dict(reversal_meta),
        }
    )
    return await coordinator.process_strategy_signal("adaptive_ob", {"symbol": symbol, "side": "long"})


async def compute_incident_metrics(
    *,
    log_path: Path,
    exchange: str,
    symbol: str,
    window_start: datetime,
    window_end: datetime,
    config_path: Path,
    position_ids: List[str],
    lookback_hours: int = 6,
) -> Dict[str, Any]:
    """
    Machine-readable incident replay summary.

    Returns:
      {
        "entry": {"attempts": int, "accepted": int, "blocked": int, "by_reason": {code: n}},
        "churn": {"pairs": int, "drop_count": int, "by_reason": {code: n}},
        "positions": [{"position_id": str, "open_ts": str, "status": str, "reason_code": str}],
      }
    """
    cfg = _load_config(config_path)
    ind_cfg = cfg.get("indicators", {}) or {}

    strat_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {})
    crash_cfg = (strat_cfg.get("crash_guard") or {}) if isinstance(strat_cfg, dict) else {}
    hook_delta = float(strat_cfg.get("hook_delta", 1.0) or 1.0)

    if isinstance(crash_cfg, dict):
        if "panic_volume_buckets" in crash_cfg:
            crash_cfg["panic_volume_buckets"] = _normalize_inline_list(crash_cfg.get("panic_volume_buckets"))
        if "cooldown_escalation_steps" in crash_cfg:
            crash_cfg["cooldown_escalation_steps"] = _normalize_inline_list(crash_cfg.get("cooldown_escalation_steps"))

    replay_cfg = cfg
    try:
        replay_cfg.setdefault("strategies", {}).setdefault("adaptive_ob", {}).setdefault("volume_filters", {})["enabled"] = False
        replay_cfg["strategies"]["adaptive_ob"].setdefault("crash_guard", {})["enabled"] = True
    except Exception:
        pass

    positions = _extract_positions_from_log(
        log_path,
        target_position_ids=list(position_ids),
        window_start=window_start,
        window_end=window_end,
    )
    if not positions:
        return {
            "entry": {"attempts": 0, "accepted": 0, "blocked": 0, "by_reason": {}},
            "churn": {"pairs": 0, "drop_count": 0, "by_reason": {}},
            "positions": [],
        }

    lookback = pd.Timedelta(hours=max(1, int(lookback_hours)))
    since_dt = (pd.Timestamp(window_start) - lookback).to_pydatetime().replace(tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)

    client = CcxtClient(str(exchange).lower())
    df_1m_raw = await client.ohlcv(symbol, "1m", limit=1440, add_indicators=False, since=since_ms)
    if df_1m_raw is None or df_1m_raw.empty:
        raise SystemExit("Failed to fetch 1m OHLCV; cannot replay.")
    df_1m_raw = _ensure_utc_index(df_1m_raw)
    df_1m_raw = df_1m_raw.loc[(df_1m_raw.index >= pd.Timestamp(since_dt)) & (df_1m_raw.index <= pd.Timestamp(window_end))]

    df_1m_ind = add_indicators(df_1m_raw, ind_cfg)
    df_5m_raw = _build_5m_from_1m(df_1m_raw)
    df_5m_ind = add_indicators(df_5m_raw, ind_cfg)

    mdp = ReplayMarketDataPipeline(df_1m_raw=df_1m_raw, df_1m_ind=df_1m_ind, df_5m_ind=df_5m_ind)
    coordinator = _make_coordinator(market_data_pipeline=mdp, config=replay_cfg)
    clock = ReplayClock()

    entry_attempts = 0
    entry_accepted = 0
    entry_blocked = 0
    entry_by_reason: Dict[str, int] = {}
    per_pos: List[Dict[str, Any]] = []

    for pos in positions:
        entry_attempts += 1
        now = pos.open_ts
        vb = pos.volume_bucket_at_entry or "EXTREME"
        reversal_meta = _compute_reversal_meta(df_1m_ind, now=now, hook_delta=hook_delta)
        result = await _replay_entry_decision(
            coordinator,
            mdp,
            clock,
            symbol=symbol,
            now=now,
            volume_bucket=vb,
            reversal_meta=reversal_meta,
        )
        status = str(result.get("status") or "").lower()
        reason_code = str(result.get("reason_code") or result.get("reason") or "unknown")
        if status in {"accepted", "ok"}:
            entry_accepted += 1
        else:
            entry_blocked += 1
            entry_by_reason[reason_code] = entry_by_reason.get(reason_code, 0) + 1
        per_pos.append(
            {
                "position_id": pos.position_id,
                "open_ts": now.isoformat(),
                "volume_bucket": str(vb),
                "status": status,
                "reason_code": reason_code,
                "reversal_meta": {
                    "rsi_hook": bool(reversal_meta.get("rsi_hook")),
                    "bull_candle": bool(reversal_meta.get("bull_candle")),
                    "reclaim": bool(reversal_meta.get("reclaim")),
                },
            }
        )

    churn_by_reason: Dict[str, int] = {}
    churn_drop_count = 0
    pairs = _infer_stop_to_next_entry_pairs(positions)
    for a, b in pairs:
        payload = a.trade_closed_payload or {}
        exit_ts = _parse_iso_utc(str(payload.get("exit_time"))) if payload.get("exit_time") else None
        if exit_ts is None:
            exit_ts = _parse_iso_utc(str(payload.get("timestamp"))) if payload.get("timestamp") else None
        if exit_ts is None:
            continue

        mdp.set_now(exit_ts)
        clock.set(exit_ts)
        await coordinator.handle_trade_closed(
            {
                "event": "TRADE_CLOSED",
                "strategy_name": "adaptive_ob",
                "symbol": symbol,
                "side": payload.get("side") or "LONG",
                "exit_reason": "stop_loss",
                "volume_bucket_at_entry": payload.get("volume_bucket_at_entry") or "EXTREME",
            }
        )

        reversal_meta = _compute_reversal_meta(df_1m_ind, now=b.open_ts, hook_delta=hook_delta)
        result = await _replay_entry_decision(
            coordinator,
            mdp,
            clock,
            symbol=symbol,
            now=b.open_ts,
            volume_bucket=str(payload.get("volume_bucket_at_entry") or "EXTREME"),
            reversal_meta=reversal_meta,
        )
        status = str(result.get("status") or "").lower()
        reason_code = str(result.get("reason_code") or result.get("reason") or "unknown")
        if status not in {"accepted", "ok"}:
            churn_drop_count += 1
            churn_by_reason[reason_code] = churn_by_reason.get(reason_code, 0) + 1

    clock.restore()
    return {
        "entry": {
            "attempts": int(entry_attempts),
            "accepted": int(entry_accepted),
            "blocked": int(entry_blocked),
            "by_reason": dict(sorted(entry_by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "churn": {
            "pairs": int(len(pairs)),
            "drop_count": int(churn_drop_count),
            "by_reason": dict(sorted(churn_by_reason.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "positions": per_pos,
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description="Replay Crash Guard (adaptive_ob) for a historical UTC window.")
    ap.add_argument("--log", required=True, help="Path to live trading log file")
    ap.add_argument("--exchange", default="bingx")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--start-utc", required=True, help="e.g. 2026-01-29T14:55:00Z")
    ap.add_argument("--end-utc", required=True, help="e.g. 2026-01-29T15:20:00Z")
    ap.add_argument("--config", default="config/config.example.yaml")
    ap.add_argument(
        "--positions",
        nargs="+",
        required=True,
        help="Position IDs to validate (e.g. pos_BTC/USDT:USDT_1769698862 ...)",
    )
    ap.add_argument("--lookback-hours", type=int, default=6)
    ap.add_argument("--output-json", default="", help="Optional path to write machine-readable replay summary JSON")
    args = ap.parse_args()

    window_start = _parse_iso_utc(args.start_utc)
    window_end = _parse_iso_utc(args.end_utc)
    if window_end <= window_start:
        raise SystemExit("end-utc must be > start-utc")

    cfg = _load_config(Path(args.config))
    strat_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {})
    crash_cfg = (strat_cfg.get("crash_guard") or {}) if isinstance(strat_cfg, dict) else {}
    cooldown_seconds = float(crash_cfg.get("cooldown_seconds", 30.0) or 30.0)

    summary = await compute_incident_metrics(
        log_path=Path(args.log),
        exchange=str(args.exchange).lower(),
        symbol=args.symbol,
        window_start=window_start,
        window_end=window_end,
        config_path=Path(args.config),
        position_ids=list(args.positions),
        lookback_hours=int(args.lookback_hours),
    )

    print("")
    print("=== Crash Guard Entry Replay (UTC) ===")
    for p in summary.get("positions", []):
        status = p.get("status")
        reason_code = p.get("reason_code")
        print(
            f"- {p.get('position_id')} @ {p.get('open_ts')} | bucket={p.get('volume_bucket')} | meta="
            f"rsi_hook={bool((p.get('reversal_meta') or {}).get('rsi_hook'))},bull={bool((p.get('reversal_meta') or {}).get('bull_candle'))},reclaim={bool((p.get('reversal_meta') or {}).get('reclaim'))} "
            f"=> status={status} reason_code={reason_code}"
        )

    print("")
    print("=== Stop->Reentry Churn Replay (UTC) ===")
    if int((summary.get("churn") or {}).get("pairs") or 0) == 0:
        print("- No stop_loss -> next entry pairs detected within the provided positions.")
    else:
        by_reason = (summary.get("churn") or {}).get("by_reason") or {}
        drop_count = int((summary.get("churn") or {}).get("drop_count") or 0)
        print(f"- churn_drop_count={drop_count} cooldown_s={cooldown_seconds:.0f} by_reason={by_reason}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
