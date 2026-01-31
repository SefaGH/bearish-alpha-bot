import logging
from datetime import datetime
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Piyasa rejimini tespit eder ve strateji filtrelemesi yapar."""

    def __init__(self, config: Dict[str, Any], market_data_pipeline: Any):
        self.config = (config or {}).get("signals", {}).get("regime_filter", {})
        self.market_data_pipeline = market_data_pipeline
        self.enabled = bool(self.config.get("enabled", False))

        self.strategy_regime_mapping = self.config.get(
            "strategy_regime_mapping",
            {
                "mean_reversion": ["range"],
                "adaptive_ob": ["range", "transitional", "crash_rebound"],
                "adaptive_str": ["trend", "transitional"],
            },
        )

        self.regime_config = {
            "trend": {"min_adx": 25},
            "range": {"max_adx": 20},
            "transitional": {"min_adx": 20, "max_adx": 25},
            "crash_rebound": {"min_price_drop_pct": 0.03, "max_recovery_pct": 0.01},
        }

    async def detect_regime(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        try:
            if not self.market_data_pipeline:
                return {
                    "label": "unknown",
                    "confidence": 0.0,
                    "reason": "no_market_data_pipeline",
                    "timeframe_used": timeframe,
                    "timestamp": datetime.utcnow(),
                }

            df = await self.market_data_pipeline.get_latest_ohlcv(symbol, timeframe=timeframe, limit=250)
            if df is None or getattr(df, "empty", True):
                return {
                    "label": "unknown",
                    "confidence": 0.0,
                    "reason": "no_ohlcv",
                    "timeframe_used": timeframe,
                    "timestamp": datetime.utcnow(),
                }

            def _last_float(col: str):
                try:
                    if col in df.columns:
                        return float(df[col].iloc[-1])
                except Exception:
                    return None
                return None

            close_px = _last_float("close")
            adx = _last_float("adx")
            atr = _last_float("atr")
            ema_mid = _last_float("ema_mid")
            if ema_mid is None:
                ema_mid = _last_float("ema50")

            atr_pct = (atr / close_px) if (atr and close_px) else 0.0
            trend_direction = None
            if close_px is not None and ema_mid is not None:
                trend_direction = "up" if close_px >= ema_mid else "down"

            regime = {
                "adx": adx,
                "atr_pct": atr_pct,
                "timeframe_used": timeframe,
                "trend_direction": trend_direction,
                "timestamp": datetime.utcnow(),
            }

            if adx is not None and adx > 25:
                regime["label"] = "trend"
                regime["confidence"] = min(adx / 50, 1.0)
            elif adx is not None and adx < 20:
                regime["label"] = "range"
                regime["confidence"] = 1.0 - (adx / 20)
            elif self._is_crash_rebound(symbol):
                regime["label"] = "crash_rebound"
                regime["confidence"] = 0.8
            else:
                regime["label"] = "transitional"
                regime["confidence"] = 0.5

            return regime

        except Exception as e:
            logger.error(f"Regime detection failed for {symbol}: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.utcnow(),
            }

    def is_strategy_allowed(self, strategy_name: str, regime: Dict[str, Any]) -> Tuple[bool, str, float]:
        if not self.enabled:
            return True, "disabled", 1.0

        regime_label = regime.get("label", "unknown")
        allowed_regimes = self.strategy_regime_mapping.get(strategy_name, [])

        if allowed_regimes and regime_label not in allowed_regimes:
            return False, f"strategy_not_allowed_in_{regime_label}", 0.0

        regime_confidence = regime.get("confidence", 0.5)
        min_confidence = self.config.get("min_regime_confidence", 0.3)

        if regime_confidence < min_confidence:
            penalty = regime_confidence / min_confidence if min_confidence else 0.0
            return True, f"low_regime_confidence_{regime_confidence:.2f}", penalty

        return True, f"allowed_in_{regime_label}", 1.0

    def _is_crash_rebound(self, symbol: str) -> bool:
        return False
