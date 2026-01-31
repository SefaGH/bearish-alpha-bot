#!/usr/bin/env python3
"""Simulate adaptive_ob entry timing around a sharp drop.

Goal
----
Given a symbol + time window, replay historical OHLCV, run the current
`AdaptiveOversoldBounce.signal()` decision on each bar close, and verify:
  1) No entries are produced during the main down-leg (peak -> trough)
  2) Entries (if any) occur after the trough and after a bounce.

This is meant to validate the behavior you expect from the screenshot:
"do not enter while the dump is ongoing; enter after the dump ends".

Notes
-----
- This uses repo strategy logic (not the live engine) and evaluates at a fixed replay step.
- It fetches OHLCV via `CcxtClient` (network required).
- It does NOT place orders; it only prints a report and optional JSON/CSV.
- To make time-based persistency behave like live, it patches the strategy's internal `time.time()`
    during replay to the simulated timestamp.
- To exercise hybrid/persistency logic, it constructs a *forming* decision candle on `--tf`
    (default 30m) from `--form-from` sub-bars (default 5m).

Example
-------
python tools/simulate_adaptive_ob_drop_entry.py \
  --exchange bingx --symbol BTC/USDT:USDT \
  --start 2026-01-31T13:30:00Z --end 2026-01-31T15:30:00Z \
    --tf 30m --form-from 5m --eval-step 5m --lookback-bars 240
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure project root + `src/` are on sys.path when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import yaml  # repo-local minimal yaml shim

from src.core.ccxt_client import CcxtClient
from src.core.indicators import add_indicators
from src.core.strategy_coordinator import StrategyCoordinator
from src.core.volume_analyzer import VolumeAnalyzer
from src.strategies.adaptive_ob import AdaptiveOversoldBounce


@contextlib.contextmanager
def _patch_strategy_time(epoch_seconds: float):
    """Patch adaptive_ob module's time.time() to a deterministic simulated timestamp."""
    try:
        import src.strategies.adaptive_ob as adaptive_ob_mod

        orig = adaptive_ob_mod.time.time
        adaptive_ob_mod.time.time = lambda: float(epoch_seconds)
        try:
            yield
        finally:
            adaptive_ob_mod.time.time = orig
    except Exception:
        # Fall back to real time if patching fails.
        yield


def _parse_iso_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _tf_to_minutes(tf: str) -> int:
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    # conservative fallback
    return 1


def _floor_time(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    tf = str(tf).lower().strip()
    if tf.endswith("m"):
        minutes = int(tf[:-1])
        return ts.floor(f"{minutes}min")
    if tf.endswith("h"):
        hours = int(tf[:-1])
        return ts.floor(f"{hours}H")
    return ts


def _aggregate_ohlcv(df: pd.DataFrame) -> pd.Series:
    """Aggregate a slice of lower-tf OHLCV into a single candle."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    out: Dict[str, Any] = {
        "open": float(df["open"].iloc[0]),
        "high": float(df["high"].max()),
        "low": float(df["low"].min()),
        "close": float(df["close"].iloc[-1]),
    }
    if "volume" in df.columns:
        try:
            out["volume"] = float(df["volume"].sum())
        except Exception:
            out["volume"] = float(df["volume"].iloc[-1])
    return pd.Series(out)


def _build_forming_tf_from_base(
    *,
    base_df: pd.DataFrame,
    now: pd.Timestamp,
    signal_tf: str,
    include_forming: bool = True,
) -> pd.DataFrame:
    """Construct a higher-tf OHLCV series ending at `now` with a forming candle for the current bucket."""
    base_df = _ensure_utc_index(base_df)
    now = pd.Timestamp(now).tz_convert("UTC")
    bucket_open = _floor_time(now, signal_tf)

    closed_base = base_df.loc[base_df.index < bucket_open]
    if closed_base.empty:
        closed_tf = pd.DataFrame()
    else:
        rule = signal_tf.lower().replace("m", "min")
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        if "volume" in closed_base.columns:
            agg["volume"] = "sum"
        closed_tf = (
            closed_base.resample(rule, label="left", closed="left")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
        )

    if not include_forming:
        out = closed_tf.copy()
        out.attrs = dict(out.attrs or {})
        out.attrs["includes_forming"] = False
        out.attrs["fallback_reason"] = None
        return out

    forming_base = base_df.loc[(base_df.index >= bucket_open) & (base_df.index <= now)]
    if forming_base.empty:
        out = closed_tf.copy()
        out.attrs = dict(out.attrs or {})
        out.attrs["includes_forming"] = False
        out.attrs["fallback_reason"] = None
        return out

    forming_row = _aggregate_ohlcv(forming_base)
    forming_df = pd.DataFrame([forming_row], index=pd.DatetimeIndex([bucket_open], tz="UTC"))
    out = pd.concat([closed_tf, forming_df], axis=0)

    out.attrs = dict(out.attrs or {})
    out.attrs["includes_forming"] = True
    out.attrs["forming_open_time"] = int(bucket_open.timestamp() * 1000)
    out.attrs["forming_last_update_ts"] = int(now.timestamp() * 1000)
    out.attrs["forming_update_age_ms"] = 0
    out.attrs["fallback_reason"] = None
    return out


def _load_cfg(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh.read()) or {}


def _extract_adaptive_ob_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Strategy ctor expects its own block.
    return ((cfg.get("strategies") or {}).get("adaptive_ob") or {}) if isinstance(cfg, dict) else {}


class _ReplayMarketDataPipeline:
    """Minimal time-sliced OHLCV provider compatible with StrategyCoordinator + VolumeAnalyzer."""

    def __init__(self, *, dfs_by_tf: Dict[str, pd.DataFrame]):
        self._dfs_by_tf = {str(k).lower(): _ensure_utc_index(v) for k, v in (dfs_by_tf or {}).items()}
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
        df = self._dfs_by_tf.get(tf)
        if df is None or df.empty:
            return pd.DataFrame()

        # We evaluate at bar close; exclude the candle at "now".
        cutoff = pd.Timestamp(self._now).tz_convert("UTC")
        out = df.loc[df.index < cutoff].copy()
        if limit is not None:
            out = out.tail(int(limit))
        # Replay doesn't synthesize a forming candle; we mark it as closed-only.
        out.attrs = dict(out.attrs or {})
        out.attrs["includes_forming"] = False
        out.attrs["fallback_reason"] = None
        return out


def _make_coordinator(*, market_data_pipeline: Any, cfg: Dict[str, Any]) -> StrategyCoordinator:
    pm = type("PM", (), {})()
    pm.cfg = cfg or {}
    rm = type("RM", (), {})()
    return StrategyCoordinator(pm, rm, market_data_pipeline=market_data_pipeline, config=cfg)


def _should_block_by_crash_guard(
    *,
    is_panic_state: bool,
    panic_meta: Dict[str, Any],
    volume_bucket: str,
    crash_cfg: Dict[str, Any],
    signal_meta: Dict[str, Any],
) -> Tuple[bool, str]:
    """Mirror the crash-guard reversal gating used in StrategyCoordinator."""
    if not is_panic_state:
        return False, "not_panic"

    rsi_hook = bool(signal_meta.get("rsi_hook"))
    bull_candle = bool(signal_meta.get("bull_candle"))
    reclaim = bool(signal_meta.get("reclaim"))

    try:
        extreme_gap_th = float(crash_cfg.get("extreme_gap_atr_threshold", 0.0) or 0.0)
    except Exception:
        extreme_gap_th = 0.0
    ema_gap_atr = None
    try:
        ema_gap_atr = float(panic_meta.get("ema_fast_gap_atr")) if panic_meta.get("ema_fast_gap_atr") is not None else None
    except Exception:
        ema_gap_atr = None

    strict = str(volume_bucket or "").upper().strip() == "EXTREME"
    if not strict and extreme_gap_th > 0 and ema_gap_atr is not None and ema_gap_atr >= extreme_gap_th:
        strict = True

    if strict:
        reversal_ok = bool(rsi_hook and reclaim)
        reason = "panic_strict_requires_rsi_hook_and_reclaim"
    else:
        reversal_ok = bool(rsi_hook and (bull_candle or reclaim))
        reason = "panic_requires_rsi_hook_and_(bull_or_reclaim)"

    if reversal_ok:
        return False, "panic_reversal_ok"
    return True, reason


@dataclass
class SignalHit:
    ts: datetime
    entry: float
    trigger_price: Optional[float]
    side: str
    reason: str
    meta: Dict[str, Any]
    accepted: bool
    accepted_reason: str
    volume_bucket: Optional[str]
    is_panic_state: Optional[bool]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pick_entry_price(signal: Dict[str, Any]) -> Optional[float]:
    # adaptive_ob generally provides entry; fallback to close.
    for k in ("entry", "trigger_price", "price"):
        v = _safe_float(signal.get(k))
        if v and v > 0:
            return v
    meta = signal.get("meta") or {}
    v = _safe_float(meta.get("trigger_price"))
    if v and v > 0:
        return v
    return None


def _drop_window(df: pd.DataFrame, start: datetime, end: datetime) -> Tuple[datetime, datetime, float, float]:
    """Compute peak->trough window inside [start,end]."""
    w = _ensure_utc_index(df)
    w = w.loc[(w.index >= pd.Timestamp(start)) & (w.index <= pd.Timestamp(end))]
    if w.empty:
        raise ValueError("No OHLCV data in requested window")

    trough_idx = w["low"].idxmin()
    trough = float(w.loc[trough_idx]["low"])

    pre = w.loc[w.index <= trough_idx]
    peak_idx = pre["high"].idxmax()
    peak = float(pre.loc[peak_idx]["high"])

    return peak_idx.to_pydatetime(), trough_idx.to_pydatetime(), peak, trough


def _bounce_pct(entry: float, trough: float) -> Optional[float]:
    if trough <= 0 or entry <= 0:
        return None
    return (entry - trough) / trough


async def _fetch_ohlcv(client: CcxtClient, symbol: str, tf: str, since: datetime, limit: int) -> pd.DataFrame:
    df = await client.ohlcv(
        symbol,
        tf,
        limit=int(limit),
        add_indicators=False,
        since=int(since.timestamp() * 1000),
    )
    if df is None or df.empty:
        return pd.DataFrame()
    return _ensure_utc_index(df)


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="bingx")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--tf", default="30m", help="Strategy decision timeframe (live adaptive_ob uses 30m)")
    p.add_argument("--form-from", default="5m", help="Lower timeframe used to construct forming candles")
    p.add_argument("--eval-step", default="5m", help="Replay evaluation step")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--lookback-bars", type=int, default=240)
    p.add_argument("--fetch-lookback-bars", type=int, default=600)
    p.add_argument("--config", default=str(Path(_ROOT) / "config" / "config.example.yaml"))
    # Optional overrides to make offline replay deterministic / align with desired behavior.
    p.add_argument("--persistency-mode", default="")
    p.add_argument("--persistency-seconds", type=float, default=None)
    p.add_argument("--persistency-min-samples", type=int, default=None)
    p.add_argument("--persistency-min-bounce-pct", type=float, default=None)
    p.add_argument("--persistency-min-bounce-apply", default="")
    p.add_argument("--out-json", default="")
    p.add_argument("--out-csv", default="")
    p.add_argument(
        "--apply-crash-guard",
        action="store_true",
        help="Apply StrategyCoordinator crash_guard gating (panic reversal) to signals.",
    )
    args = p.parse_args()

    start = _parse_iso_utc(args.start)
    end = _parse_iso_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    cfg = _load_cfg(Path(args.config) if args.config else None)
    strat_cfg = _extract_adaptive_ob_cfg(cfg)

    # Pull crash guard + volume analyzer cfg (only used when --apply-crash-guard).
    crash_cfg = ((cfg.get("strategies") or {}).get("adaptive_ob") or {}).get("crash_guard") or {}
    vol_cfg = (cfg.get("volume_analyzer") or {}) if isinstance(cfg, dict) else {}

    # Apply optional overrides.
    if args.persistency_mode:
        strat_cfg["adaptive_ob_persistency_mode"] = str(args.persistency_mode)
    if args.persistency_seconds is not None:
        strat_cfg["adaptive_ob_persistency_seconds"] = float(args.persistency_seconds)
    if args.persistency_min_samples is not None:
        strat_cfg["adaptive_ob_persistency_min_samples"] = int(args.persistency_min_samples)
    if args.persistency_min_bounce_pct is not None:
        strat_cfg["adaptive_ob_persistency_min_bounce_pct"] = float(args.persistency_min_bounce_pct)
    if args.persistency_min_bounce_apply:
        strat_cfg["adaptive_ob_persistency_min_bounce_apply"] = str(args.persistency_min_bounce_apply)

    client = CcxtClient(str(args.exchange).lower())

    # Fetch enough history to warm indicators.
    fetch_bars = max(int(args.fetch_lookback_bars), int(args.lookback_bars) + 50)
    base_tf = str(args.form_from).lower().strip()
    since = start - pd.Timedelta(minutes=_tf_to_minutes(base_tf) * fetch_bars).to_pytimedelta()

    df_base_raw = await _fetch_ohlcv(client, args.symbol, base_tf, since=since, limit=2000)
    if df_base_raw.empty:
        raise SystemExit("No OHLCV fetched (network/symbol/timeframe issue)")

    # Restrict to a reasonable range.
    df_base_raw = df_base_raw.loc[(df_base_raw.index >= pd.Timestamp(since)) & (df_base_raw.index <= pd.Timestamp(end))]
    if df_base_raw.empty:
        raise SystemExit("Fetched OHLCV does not cover requested window")

    ind_cfg = (cfg.get("indicators") or {}) if isinstance(cfg, dict) else {}
    df_base_ind = add_indicators(df_base_raw, ind_cfg) if not df_base_raw.empty else df_base_raw
    df_base_ind = _ensure_utc_index(df_base_ind)

    # If crash guard simulation is enabled, prepare baselines + volume analyzer + coordinator.
    mdp = None
    coordinator = None
    volume_analyzer = None
    if args.apply_crash_guard:
        short_tf = str(vol_cfg.get("baseline_short_tf", "1h") or "1h")
        med_tf = str(vol_cfg.get("baseline_medium_tf", "4h") or "4h")

        # Fetch baselines for volume bucketing.
        lookback_days = int(vol_cfg.get("lookback_days", 14) or 14)
        since_vol = end - pd.Timedelta(days=lookback_days).to_pytimedelta()

        df_1h = await _fetch_ohlcv(client, args.symbol, short_tf, since=since_vol, limit=2000)
        df_4h = await _fetch_ohlcv(client, args.symbol, med_tf, since=since_vol, limit=2000)

        # Provide trade tf + panic tf for volume bucketing / panic structure.
        trade_tf = str(args.tf).lower().strip()
        panic_tf = str((crash_cfg.get("panic_tf") if isinstance(crash_cfg, dict) else None) or "5m").lower().strip()

        # Closed-only trade series (sufficient for volume bucketing).
        df_trade = _build_forming_tf_from_base(base_df=df_base_raw, now=pd.Timestamp(end), signal_tf=trade_tf, include_forming=False)
        df_trade = add_indicators(df_trade, ind_cfg) if not df_trade.empty else df_trade

        if panic_tf == base_tf:
            df_panic = df_base_ind
        else:
            df_panic_raw = await _fetch_ohlcv(client, args.symbol, panic_tf, since=since, limit=2000)
            df_panic = add_indicators(df_panic_raw, ind_cfg) if not df_panic_raw.empty else df_panic_raw
            df_panic = _ensure_utc_index(df_panic)

        dfs_by_tf = {
            trade_tf: df_trade,
            panic_tf: df_panic,
            short_tf.lower(): add_indicators(df_1h, ind_cfg) if not df_1h.empty else df_1h,
            med_tf.lower(): add_indicators(df_4h, ind_cfg) if not df_4h.empty else df_4h,
        }

        mdp = _ReplayMarketDataPipeline(dfs_by_tf=dfs_by_tf)
        volume_analyzer = VolumeAnalyzer(mdp, config=vol_cfg)
        coordinator = _make_coordinator(market_data_pipeline=mdp, cfg=cfg)

    # Instantiate strategy with current config.
    strat = AdaptiveOversoldBounce(strat_cfg, regime_analyzer=None, market_data_pipeline=None)

    # Evaluate at a fixed step cadence inside [start,end].
    hits: List[SignalHit] = []
    step_tf = str(args.eval_step).lower().strip()
    eval_base = df_base_ind.loc[(df_base_ind.index >= pd.Timestamp(start)) & (df_base_ind.index <= pd.Timestamp(end))]
    if eval_base.empty:
        raise SystemExit("No base OHLCV in requested window")

    if step_tf == base_tf:
        eval_idx = eval_base.index
    else:
        rule = step_tf.replace("m", "min")
        eval_idx = eval_base.resample(rule, label="left", closed="left").first().dropna().index

    for t in eval_idx:
        now_dt = t.to_pydatetime().replace(tzinfo=timezone.utc)

        if mdp is not None:
            mdp.set_now(now_dt)

        # Build a live-style forming decision series on args.tf using base sub-bars up to t.
        # This is critical: adaptive_ob persistency/min-bounce only applies when includes_forming=True.
        df_fast_slice = df_base_ind.loc[df_base_ind.index <= t].tail(int(args.lookback_bars * 6)).copy()
        df_sig = _build_forming_tf_from_base(base_df=df_fast_slice, now=t, signal_tf=str(args.tf), include_forming=True)
        df_sig = add_indicators(df_sig, ind_cfg) if not df_sig.empty else df_sig
        df_sig = _ensure_utc_index(df_sig)

        with _patch_strategy_time(pd.Timestamp(t).timestamp()):
            sig = strat.signal(
                df_sig,
                df_1h=None,
                regime_data=None,
                symbol=args.symbol,
                market_data={"df_5m": df_fast_slice},
            )
        if not sig:
            continue
        entry = _pick_entry_price(sig)
        if not entry:
            continue
        accepted = True
        accepted_reason = "strategy_signal_only"
        volume_bucket = None
        is_panic_state = None

        if args.apply_crash_guard and coordinator is not None and volume_analyzer is not None:
            try:
                ctx = await volume_analyzer.compute_context(args.symbol, trade_timeframe=str(args.tf))
                volume_bucket = str(getattr(ctx, "bucket", "") or "").upper().strip() or None
            except Exception:
                volume_bucket = None

            try:
                is_panic_state, panic_meta = await coordinator._compute_panic_state(
                    symbol=args.symbol,
                    volume_bucket=volume_bucket,
                    crash_cfg=crash_cfg if isinstance(crash_cfg, dict) else {},
                )
            except Exception:
                is_panic_state, panic_meta = False, {}

            block, block_reason = _should_block_by_crash_guard(
                is_panic_state=bool(is_panic_state),
                panic_meta=panic_meta or {},
                volume_bucket=str(volume_bucket or ""),
                crash_cfg=crash_cfg if isinstance(crash_cfg, dict) else {},
                signal_meta=(sig.get("meta") or {}),
            )
            if block:
                accepted = False
                accepted_reason = f"blocked:{block_reason}"
            else:
                accepted = True
                accepted_reason = "accepted:crash_guard_ok"

        hits.append(
            SignalHit(
                ts=now_dt,
                entry=float(entry),
                trigger_price=_safe_float((sig.get("meta") or {}).get("trigger_price")) or _safe_float(sig.get("trigger_price")),
                side=str(sig.get("side") or ""),
                reason=str(sig.get("reason") or ""),
                meta=dict(sig.get("meta") or {}),
                accepted=bool(accepted),
                accepted_reason=str(accepted_reason),
                volume_bucket=volume_bucket,
                is_panic_state=bool(is_panic_state) if is_panic_state is not None else None,
            )
        )

    peak_ts, trough_ts, peak, trough = _drop_window(df_base_ind, start, end)

    accepted_hits = [h for h in hits if h.accepted]
    during = [h for h in accepted_hits if h.ts <= trough_ts]
    after = [h for h in accepted_hits if h.ts > trough_ts]

    print("\n=== adaptive_ob drop-entry simulation ===")
    print(f"symbol={args.symbol} exchange={args.exchange} tf={args.tf}")
    print(f"window={start.isoformat()} -> {end.isoformat()}")
    print(f"peak={peak:.2f} @ {peak_ts.isoformat()} | trough={trough:.2f} @ {trough_ts.isoformat()}")
    print(f"drop_pct={(trough-peak)/peak*100:.2f}%")
    print("\nSignals found:")
    if args.apply_crash_guard:
        blocked = len([h for h in hits if not h.accepted])
        print(
            f"  raw_signals={len(hits)} | accepted_signals={len(accepted_hits)} | blocked_by_crash_guard={blocked} "
            f"| accepted_during_drop(<=trough)={len(during)} | accepted_after_drop(>trough)={len(after)}"
        )
    else:
        print(f"  total={len(hits)} | during_drop(<=trough)={len(during)} | after_drop(>trough)={len(after)}")

    if hits:
        print("\nFirst 10 signals:")
        for h in hits[:10]:
            b = _bounce_pct(h.entry, trough)
            b_s = f"{b*100:.3f}%" if b is not None else "n/a"
            loc = "DURING" if h.ts <= trough_ts else "AFTER"
            status = "ACCEPT" if h.accepted else "BLOCK"
            extra = ""
            if args.apply_crash_guard:
                extra = f" | bucket={h.volume_bucket} panic={h.is_panic_state} {h.accepted_reason}"
            print(
                f"  - {h.ts.isoformat()} | {loc} | {status} | entry={h.entry:.2f} | "
                f"bounce_from_trough={b_s} | side={h.side}{extra}"
            )

    verdict_ok = (len(during) == 0 and len(after) > 0) if len(accepted_hits) > 0 else None
    if verdict_ok is True:
        print("\nVERDICT: ✅ No entries during the dump; entries only after trough.")
    elif verdict_ok is False:
        print("\nVERDICT: ❌ Entries occurred during the dump window.")
    else:
        print("\nVERDICT: (no signals in window) — cannot confirm entry timing from signals.")

    payload = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "tf": args.tf,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "peak_ts": peak_ts.isoformat(),
        "peak": peak,
        "trough_ts": trough_ts.isoformat(),
        "trough": trough,
        "drop_pct": (trough - peak) / peak if peak else None,
        "raw_signals_total": len(hits),
        "accepted_signals_total": len(accepted_hits),
        "accepted_signals_during_drop": len(during),
        "accepted_signals_after_drop": len(after),
        "apply_crash_guard": bool(args.apply_crash_guard),
        "signals": [
            {
                "ts": h.ts.isoformat(),
                "entry": h.entry,
                "trigger_price": h.trigger_price,
                "side": h.side,
                "reason": h.reason,
                "bounce_from_trough": _bounce_pct(h.entry, trough),
                "accepted": bool(h.accepted),
                "accepted_reason": h.accepted_reason,
                "volume_bucket": h.volume_bucket,
                "is_panic_state": h.is_panic_state,
            }
            for h in hits
        ],
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.out_json}")

    if args.out_csv:
        rows = []
        for h in hits:
            rows.append(
                {
                    "ts": h.ts.isoformat(),
                    "entry": h.entry,
                    "trigger_price": h.trigger_price,
                    "side": h.side,
                    "bounce_from_trough": _bounce_pct(h.entry, trough),
                    "during_drop": h.ts <= trough_ts,
                    "accepted": bool(h.accepted),
                    "accepted_reason": h.accepted_reason,
                    "volume_bucket": h.volume_bucket,
                    "is_panic_state": h.is_panic_state,
                }
            )
        pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8")
        print(f"Wrote CSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
