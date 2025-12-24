import json
import os
import threading
from typing import Any, Dict, Optional, Tuple

# Import centralized config
try:
    from config.live_trading_config import get_config
except ImportError:
    # Fallback for tests or isolated runs
    get_config = None

_LOCK = threading.Lock()
_LAST_LOGGED: Dict[Tuple[str, str, str], int] = {}  # (strategy, symbol, tf) -> last_closed_ts_ms

ENV_FLAG = "STRATEGY_SHADOW_EVAL"  # "1" enables shadow eval logs


def shadow_enabled() -> bool:
    # 1. Try centralized config (App Config aware)
    if get_config:
        try:
            cfg = get_config()
            val = cfg.get("strategy_shadow_eval")
            if val is not None:
                return str(val).lower() in ("1", "true", "on")
        except Exception:
            pass  # Fallback to env var if config fails

    # 2. Fallback to legacy env var
    return os.getenv(ENV_FLAG, "0") == "1"


def extract_last_closed_ts_ms(df) -> Optional[int]:
    """
    Best-effort extraction of the latest CLOSED candle timestamp in milliseconds.
    Prefers df.attrs['last_closed_ts'] (provided by market_data_pipeline/get_latest_ohlcv).
    Falls back to 'open_time' column, then index[-1] (Timestamp).
    """
    try:
        ts = getattr(df, "attrs", {}).get("last_closed_ts")
        if ts is not None:
            return int(ts)

        if hasattr(df, "columns") and "open_time" in df.columns:
            return int(df["open_time"].iloc[-1])

        idx = df.index[-1]
        if hasattr(idx, "value"):
            return int(idx.value // 1_000_000)  # ns -> ms
        return None
    except Exception:
        return None


def extract_df_meta(df) -> Dict[str, Any]:
    """Pull ohlcv_source / retrieved_at if present (best effort)."""
    meta: Dict[str, Any] = {}
    try:
        attrs = getattr(df, "attrs", {}) or {}
        if "ohlcv_source" in attrs:
            meta["ohlcv_source"] = attrs.get("ohlcv_source")
        if "retrieved_at" in attrs:
            meta["retrieved_at"] = attrs.get("retrieved_at")
    except Exception:
        pass
    return meta


def should_log(strategy: str, symbol: str, tf: str, last_closed_ts_ms: Optional[int]) -> bool:
    if last_closed_ts_ms is None:
        return False
    key = (strategy, symbol, tf)
    with _LOCK:
        prev = _LAST_LOGGED.get(key)
        if prev == last_closed_ts_ms:
            return False
        _LAST_LOGGED[key] = last_closed_ts_ms
        return True


def emit_shadow_log(logger, payload: Dict[str, Any], strategy: str, symbol: str, tf: str, last_closed_ts_ms: Optional[int]) -> None:
    """
    Emits: logger.info("strategy_shadow_eval %s", json.dumps(payload))
    Throttled per closed candle.
    Never raises.
    """
    try:
        if not shadow_enabled():
            return
        if not should_log(strategy, symbol, tf, last_closed_ts_ms):
            return
        logger.info("strategy_shadow_eval %s", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    except Exception:
        # fail-safe: do not impact strategy
        return
