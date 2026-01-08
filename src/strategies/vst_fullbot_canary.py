from __future__ import annotations

from typing import Any, Dict, Optional


class VstFullbotCanaryStrategy:
    strategy_name = "vst_fullbot_canary"

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        side: str = "long",
        stop_pct: float = 0.01,
        target_rr: float = 5.0,
    ) -> None:
        self.cfg = cfg or {}
        self.side = str(side or "long").lower().strip()
        self.stop_pct = float(self.cfg.get("stop_pct", stop_pct) or stop_pct)
        self.target_rr = float(self.cfg.get("target_rr", target_rr) or target_rr)
        self._fired = False

    @staticmethod
    def _last_close(df: Any) -> Optional[float]:
        try:
            if df is None:
                return None
            close_series = df.get("close") if hasattr(df, "get") else None
            if close_series is None:
                close_series = df["close"]  # type: ignore[index]
            value = close_series.iloc[-1]  # type: ignore[attr-defined]
            return float(value)
        except Exception:
            return None

    def signal(
        self,
        df_30m=None,
        df_1h=None,
        regime_data=None,
        *,
        symbol: str,
        ml_context=None,
        market_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        if self._fired:
            return None

        entry_price = None
        if isinstance(market_data, dict):
            entry_price = self._last_close(market_data.get("30m")) or self._last_close(market_data.get("30m_closed"))
        if entry_price is None:
            entry_price = self._last_close(df_30m)

        if not entry_price or entry_price <= 0:
            return None

        stop_pct = float(self.stop_pct or 0.0)
        if stop_pct <= 0:
            stop_pct = 0.01

        target_rr = float(self.target_rr or 0.0)
        if target_rr <= 0:
            target_rr = 1.0

        is_short = self.side in {"short", "sell"}
        if is_short:
            stop_price = entry_price * (1 + stop_pct)
            target_price = entry_price * (1 - stop_pct * target_rr)
            side = "short"
        else:
            stop_price = entry_price * (1 - stop_pct)
            target_price = entry_price * (1 + stop_pct * target_rr)
            side = "long"

        self._fired = True

        return {
            "symbol": symbol,
            "side": side,
            "entry": float(entry_price),
            "stop": float(stop_price),
            "target": float(target_price),
            "timeframe": "30m",
            "reason": "vst_fullbot_canary_forced_entry",
            "meta": {
                "vst_fullbot_canary": True,
                "stop_pct": stop_pct,
                "target_rr": target_rr,
            },
        }
