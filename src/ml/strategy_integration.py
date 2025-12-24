"""
Strategy Integration Layer for Advanced Price Prediction.

Integrates price forecasts with existing trading strategies for AI-enhanced
decision making and risk management.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING
import logging

import numpy as np
import pandas as pd

from .price_predictor import AdvancedPricePredictionEngine
from .regime_predictor import MLRegimePredictor

if TYPE_CHECKING:
    from core.market_data_pipeline import MarketDataPipeline

logger = logging.getLogger(__name__)


class AIEnhancedStrategyAdapter:
    """Adapter that enhances existing trading strategies with AI predictions."""

    def __init__(
        self,
        price_engine: AdvancedPricePredictionEngine,
        regime_predictor: MLRegimePredictor,
        config: Dict[str, Any],
    ) -> None:
        """Initialize strategy adapter.

        Args:
            price_engine: Price forecasting engine.
            regime_predictor: Market regime predictor.
            config: Configuration dictionary. Expected to include a
                "prediction" section with thresholds and scaling factors.
        """
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        self.config = config or {}

        pred_config = self.config.get("prediction", {}) or {}
        self.min_confidence: float = float(
            pred_config.get("min_confidence_threshold", 0.6)
        )
        self.min_consensus: float = float(pred_config.get("consensus_threshold", 0.7))
        self.risk_scaling_factor: float = float(
            pred_config.get("risk_scaling_factor", 1.5)
        )

        # Soft-weight thresholds from config
        regime_config = self.config.get("regime", {}) or {}
        self.regime_hard_reject: float = float(
            regime_config.get("min_confidence_hard_reject", 0.30)
        )
        self.regime_full_weight: float = float(
            regime_config.get("min_confidence_full_weight", 0.60)
        )

        logger.info("AI-Enhanced Strategy Adapter initialized")
        logger.info(
            "   - Adapter min regime confidence: %s", self.min_confidence
        )
        logger.info(
            "   - Soft-weight thresholds: hard_reject=%.2f, full_weight=%.2f",
            self.regime_hard_reject, self.regime_full_weight
        )

    def _calculate_regime_weight(self, regime_confidence: float) -> Optional[float]:
        """Calculate regime weight based on confidence with soft-weighting.
        
        Args:
            regime_confidence: Regime prediction confidence (0.0-1.0)
            
        Returns:
            regime_weight (0.0-1.0) or None if below hard reject threshold
        """
        if regime_confidence < self.regime_hard_reject:
            return None  # Hard reject
        elif regime_confidence >= self.regime_full_weight:
            return 1.0  # Full weight
        else:
            # Linear interpolation between hard_reject and full_weight
            return regime_confidence / self.regime_full_weight

    async def enhance_strategy_signal(
        self,
        symbol: str,
        base_signal: Dict[str, Any] | str,
        current_price: float,
        market_data_pipeline: "MarketDataPipeline" | None,
    ) -> Dict[str, Any]:
        """Enhance a base trading strategy signal with AI predictions.

        Args:
            symbol: Trading symbol.
            base_signal: Base strategy output (dict with 'signal' or 'side', or str).
            current_price: Latest traded price.
            market_data_pipeline: Pipeline to fetch fresh OHLCV for regime model.

        Returns:
            Enhancement dictionary including combined signal and diagnostics.
        """
        try:
            processed_signal = self._normalize_signal(base_signal)
            enhancement: Dict[str, Any] = {
                "original_signal": processed_signal["signal"],
                "original_strength": processed_signal.get("strength", 0.0),
            }

            # --- STEP 1: Get regime prediction (optional) ---
            regime_info: Optional[Dict[str, Any]] = None
            if self.regime_predictor is not None and market_data_pipeline is not None:
                try:
                    price_data = await market_data_pipeline.get_latest_ohlcv(
                        symbol, timeframe="1h", limit=2000
                    )
                    if price_data is not None and not getattr(price_data, "empty", True):
                        # NOTE: predict_regime_transition expected (symbol, price_data)
                        regime_info = await self.regime_predictor.predict_regime_transition(
                            symbol, price_data
                        )
                    else:
                        logger.warning(
                            "🧠 [ML-ADAPTER] Regime prediction skipped: No 1h data for %s.",
                            symbol,
                        )
                except Exception as e:  # noqa: BLE001 - we log full context
                    logger.error(
                        "🧠 [ML-ADAPTER] Regime prediction failed: %s", e, exc_info=True
                    )

            # --- STEP 2: Apply soft-weighting to regime by confidence ---
            if regime_info:
                regime_confidence = float(regime_info.get("confidence", 0.0))
                predicted_regime = str(regime_info.get("predicted_regime", "neutral"))
                
                # Calculate regime_weight using configured thresholds
                regime_weight = self._calculate_regime_weight(regime_confidence)
                
                if regime_weight is None:
                    # Hard reject: completely ignore low confidence predictions
                    logger.info(
                        "🧠 [ML-ADAPTER] Regime for %s ignored (Conf: %.2f < %.2f hard reject threshold)",
                        symbol,
                        regime_confidence,
                        self.regime_hard_reject,
                    )
                else:
                    # Add regime info with weight
                    enhancement["predicted_regime"] = predicted_regime
                    enhancement["regime_name"] = predicted_regime
                    enhancement["regime_confidence"] = regime_confidence
                    enhancement["regime_weight"] = float(regime_weight)
                    
                    logger.info(
                        "🧠 [ML-ADAPTER] Regime for %s is %s (Conf: %.2f, Weight: %.2f)",
                        symbol,
                        predicted_regime.upper(),
                        regime_confidence,
                        regime_weight,
                    )

            # --- STEP 3: Price forecast + AI signals ---
            price_forecast = None
            try:
                price_forecast = self.price_engine.get_price_forecast(symbol)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "🧠 [ML-ADAPTER] get_price_forecast failed for %s: %s",
                    symbol,
                    e,
                    exc_info=True,
                )

            if not price_forecast:
                enhancement["recommendations"] = [
                    "No AI price forecast available",
                ]
                logger.info(
                    "🧠 [ML-ADAPTER] No price forecast for %s. Only regime will be used.",
                    symbol,
                )
            else:
                try:
                    ai_signal = self.price_engine.generate_trading_signals(
                        symbol, current_price
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "🧠 [ML-ADAPTER] generate_trading_signals failed for %s: %s",
                        symbol,
                        e,
                        exc_info=True,
                    )
                    ai_signal = {"signal": "neutral", "strength": 0.0}

                enhancement["ai_signal"] = str(ai_signal.get("signal", "neutral"))
                enhancement["ai_strength"] = float(ai_signal.get("strength", 0.0))

                # Create comprehensive ML metadata for persistence and explainability
                ai_consensus = float(ai_signal.get("consensus", 0.0))
                ai_confidence = float(ai_signal.get("confidence", 0.0))
                ai_uncertainty = float(ai_signal.get("uncertainty", 1.0))
                ai_direction = str(ai_signal.get("signal", "neutral"))
                
                ml_metadata = {
                    "consensus": ai_consensus,
                    "price_direction": ai_direction,
                    "regime": enhancement.get("predicted_regime", "neutral"),
                    "price_confidence": ai_confidence,
                    "uncertainty": ai_uncertainty,
                    "forecast_price": ai_signal.get("forecast_price"),
                    "ml_price_score_normalized": ai_consensus * 100.0, # 0-100 scale
                    "regime_confidence": enhancement.get("regime_confidence", 0.0)
                }
                enhancement["ml_metadata"] = ml_metadata

                combined = self._combine_signals(
                    processed_signal, ai_signal, price_forecast
                )
                enhancement.update(combined)
                logger.info(
                    "🧠 [ML-ADAPTER] Price forecast enhanced signal for %s.", symbol
                )

            return enhancement

        except Exception as e:  # noqa: BLE001
            logger.error(
                "CRITICAL: Error in enhance_strategy_signal: %s", e, exc_info=True
            )
            # Best-effort original signal in error path
            original = (
                base_signal.get("signal")
                if isinstance(base_signal, dict) and "signal" in base_signal
                else base_signal.get("side")
                if isinstance(base_signal, dict)
                else str(base_signal)
            )
            return {"original_signal": (original or "unknown"), "error": str(e)}

    # ----------------------
    # Internal helpers
    # ----------------------
    def _normalize_signal(self, base_signal: Any) -> Dict[str, Any]:
        """Normalize base signal into {signal: str, strength: float}.

        Accepted inputs:
          - dict with 'signal' and optional 'strength'
          - dict with 'side' and optional 'rr_ratio'
          - str in {buy/long, sell/short, neutral}
        """
        # Case 1: Already in canonical format
        if isinstance(base_signal, dict) and "signal" in base_signal:
            signal = str(base_signal["signal"]).lower()
            try:
                strength = float(base_signal.get("strength", 0.6))
            except Exception:  # noqa: BLE001
                strength = 0.6
            return {"signal": signal, "strength": float(max(0.0, min(1.0, strength)))}

        # Case 2: Classic side + rr_ratio
        if isinstance(base_signal, dict) and "side" in base_signal:
            side = str(base_signal["side"]).lower()
            if side in ("buy", "long"):
                signal_direction = "bullish"
            elif side in ("sell", "short"):
                signal_direction = "bearish"
            else:
                signal_direction = "neutral"

            rr_ratio = base_signal.get("rr_ratio")
            if rr_ratio is not None:
                try:
                    rr_ratio_f = float(rr_ratio)
                except Exception:  # noqa: BLE001
                    rr_ratio_f = 1.0
                strength = 0.5 + (max(0.0, rr_ratio_f - 1.0) * 0.15)
                strength = max(0.5, min(0.8, strength))
            else:
                strength = 0.6

            return {"signal": signal_direction, "strength": float(strength)}

        # Case 3: Raw string
        if isinstance(base_signal, str):
            side = base_signal.lower().strip()
            if side in ("buy", "long", "bullish"):
                signal_direction = "bullish"
            elif side in ("sell", "short", "bearish"):
                signal_direction = "bearish"
            else:
                signal_direction = "neutral"
            return {"signal": signal_direction, "strength": 0.7}

        # Fallback
        logger.warning(
            "🧠 [ML-ADAPTER] Incompatible signal format: %r. Assuming neutral.",
            base_signal,
        )
        return {"signal": "neutral", "strength": 0.0}

    def _combine_signals(
        self,
        base_signal: Dict[str, Any],
        ai_signal: Dict[str, Any],
        forecast: Dict[str, Any],  # kept for future use / auditing
    ) -> Dict[str, Any]:
        """Combine base strategy signal with AI signal using weighted scheme."""
        base_strength = float(base_signal.get("strength", 0.5))
        ai_strength = float(ai_signal.get("strength", 0.0))
        ai_confidence = float(ai_signal.get("confidence", 0.0))
        consensus = float(ai_signal.get("consensus", 0.0))

        # Base/AI weights (AI scaled by its confidence)
        base_weight = 0.6
        ai_weight = 0.4 * max(0.0, min(1.0, ai_confidence))
        total_weight = base_weight + ai_weight
        if total_weight <= 0.0:
            total_weight = 1.0
        base_weight /= total_weight
        ai_weight /= total_weight

        combined_strength = base_strength * base_weight + ai_strength * ai_weight

        base_direction = self._signal_to_direction(base_signal["signal"])  # -1/0/1
        ai_direction = self._signal_to_direction(str(ai_signal.get("signal", "neutral")))

        recommendations: list[str] = []
        if base_direction == ai_direction and base_direction != 0:
            final_signal = base_signal["signal"]
            final_strength = combined_strength * 1.2
            recommendations.append("Base strategy and AI forecast agree - strong signal")
        elif abs(base_direction - ai_direction) > 1:
            final_signal = "neutral"
            final_strength = 0.0
            recommendations.append("Conflicting signals - recommend caution")
        else:
            if base_direction == 0:
                final_signal = str(ai_signal.get("signal", "neutral"))
                final_strength = ai_strength * max(0.0, min(1.0, ai_confidence))
            else:
                final_signal = base_signal["signal"]
                final_strength = base_strength * 0.8
            recommendations.append("Partial agreement - moderate confidence")

        # Timeframe consensus gate
        if consensus < self.min_consensus:
            final_strength *= 0.7
            recommendations.append(f"Low timeframe consensus ({consensus:.2f})")

        # Risk scaling: lower size when uncertainty is high
        uncertainty = float(ai_signal.get("uncertainty", 1.0))
        risk_adjustment = 1.0 / (1.0 + max(0.0, uncertainty) * self.risk_scaling_factor)
        confidence_adjustment = (
            ai_confidence
            if consensus > self.min_consensus
            else ai_confidence * 0.8
        )

        return {
            "final_signal": final_signal,
            "final_strength": float(min(max(final_strength, 0.0), 1.0)),
            "confidence_adjustment": float(max(0.0, min(1.0, confidence_adjustment))),
            "risk_adjustment": float(max(0.0, min(1.0, risk_adjustment))),
            "recommendations": recommendations,
            "forecast_price": ai_signal.get("forecast_price"),
            "uncertainty": float(uncertainty),
            "consensus": float(consensus),
        }

    @staticmethod
    def _signal_to_direction(signal: str) -> int:
        s = str(signal).lower()
        if s in {"bullish", "long", "buy"}:
            return 1
        if s in {"bearish", "short", "sell"}:
            return -1
        return 0

    def calculate_position_sizing(
        self, symbol: str, base_position: float, enhancement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Size position using confidence and risk adjustments."""
        confidence_adj = float(enhancement.get("confidence_adjustment", 1.0))
        risk_adj = float(enhancement.get("risk_adjustment", 1.0))
        adjusted_position = base_position * confidence_adj * risk_adj

        max_position = base_position * 1.5
        adjusted_position = min(adjusted_position, max_position)

        final_multiplier = (
            adjusted_position / base_position if base_position > 0 else 1.0
        )
        return {
            "base_position": float(base_position),
            "adjusted_position": float(adjusted_position),
            "confidence_multiplier": confidence_adj,
            "risk_multiplier": risk_adj,
            "final_multiplier": float(final_multiplier),
        }

    def get_risk_metrics(
        self, symbol: str, forecast: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Derive simple risk metrics from forecast aggregation."""
        if forecast is None:
            try:
                forecast = self.price_engine.get_price_forecast(symbol)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "🧠 [ML-ADAPTER] get_price_forecast failed in risk metrics for %s: %s",
                    symbol,
                    e,
                    exc_info=True,
                )
                forecast = None

        if not forecast:
            return {"risk_level": "unknown", "uncertainty": 1.0, "confidence": 0.0}

        try:
            agg = forecast["aggregated"]
            # graceful fallbacks
            uncertainty_vals = agg.get("uncertainty")
            if isinstance(uncertainty_vals, (list, tuple, np.ndarray, pd.Series)):
                uncertainty = float(np.mean(uncertainty_vals))
            else:
                uncertainty = float(uncertainty_vals)

            consensus = float(agg.get("consensus_strength", 0.0))
        except Exception:  # noqa: BLE001
            return {"risk_level": "unknown", "uncertainty": 1.0, "confidence": 0.0}

        if uncertainty < 0.02 and consensus > 0.8:
            risk_level = "low"
        elif uncertainty < 0.05 and consensus > 0.6:
            risk_level = "moderate"
        else:
            risk_level = "high"

        return {
            "risk_level": risk_level,
            "uncertainty": float(uncertainty),
            "consensus": float(consensus),
            "confidence": float(1.0 / (1.0 + max(0.0, uncertainty))),
        }


class StrategyPerformanceTracker:
    """Track and summarize strategy outcomes."""

    def __init__(self) -> None:
        self.trades: list[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_trades": 0,
            "base_strategy_wins": 0,
            "ai_enhanced_wins": 0,
            "improvement_rate": 0.0,
        }
        logger.info("Strategy Performance Tracker initialized")

    def record_trade(self, trade: Dict[str, Any]) -> None:
        trade_with_ts = {**trade, "timestamp": pd.Timestamp.now()}
        self.trades.append(trade_with_ts)
        self.metrics["total_trades"] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.trades:
            return self.metrics

        df = pd.DataFrame(self.trades)
        if (
            (df.get("strategy_type") is not None)
            and ("base" in set(df["strategy_type"]))
            and ("ai_enhanced" in set(df["strategy_type"]))
            and (df.get("pnl") is not None)
        ):
            base_pnl = df[df["strategy_type"] == "base"]["pnl"].mean()
            ai_pnl = df[df["strategy_type"] == "ai_enhanced"]["pnl"].mean()
            if pd.notna(base_pnl) and base_pnl != 0 and pd.notna(ai_pnl):
                self.metrics["improvement_rate"] = float(
                    ((ai_pnl - base_pnl) / abs(base_pnl)) * 100.0
                )
        return self.metrics

    def get_recent_performance(self, n_trades: int = 100) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades).tail(max(1, int(n_trades)))


class MLStrategyIntegrationManager:
    """Main integration manager coordinating all ML enhancements."""

    def __init__(
        self,
        price_engine: Optional[AdvancedPricePredictionEngine],
        regime_predictor: Optional[MLRegimePredictor],
        config: Dict[str, Any],
        market_data_pipeline: Optional["MarketDataPipeline"] = None,
    ) -> None:
        """Initialize the integration manager."""
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        self.market_data_pipeline = market_data_pipeline
        self.config = config or {}

        self.websocket_manager = (
            market_data_pipeline.websocket_manager if market_data_pipeline else None
        )

        self.adapter: Optional[AIEnhancedStrategyAdapter] = None
        if self.price_engine and self.regime_predictor:
            # Pass config into adapter
            self.adapter = AIEnhancedStrategyAdapter(
                self.price_engine, self.regime_predictor, self.config
            )
            logger.info("✅ AIEnhancedStrategyAdapter initialized.")
        else:
            missing = [
                name
                for name, dep in [
                    ("price_engine", self.price_engine),
                    ("regime_predictor", self.regime_predictor),
                ]
                if not dep
            ]
            logger.warning(
                "⚠️ AIEnhancedStrategyAdapter not initialized due to missing dependencies: %s.",
                ", ".join(missing) if missing else "<none>",
            )

        self.tracker = StrategyPerformanceTracker()

        logger.info("ML Strategy Integration Manager initialized")

    async def enhance_strategy_signal(
        self, symbol: str, base_signal: Dict[str, Any] | str, current_price: float
    ) -> Dict[str, Any]:
        """Facade method that calls the adapter's enhancement method."""
        if self.adapter:
            return await self.adapter.enhance_strategy_signal(
                symbol,
                base_signal,
                current_price,
                market_data_pipeline=self.market_data_pipeline,
            )
        logger.warning(
            "ML Adapter not available. Returning original signal without enhancement."
        )
        # Normalize best-effort for consistent return
        # type: ignore needed because adapter could be None, but we check it conditionally
        norm = self.adapter._normalize_signal(base_signal) if self.adapter else {  # type: ignore[attr-defined]
            "signal": (base_signal.get("signal") if isinstance(base_signal, dict) else str(base_signal)),
            "strength": 0.0,
        }
        return {"original_signal": norm.get("signal", "unknown")}

    async def get_ml_context(self, symbol: str, horizon: str = "1h") -> Dict[str, Any]:
        logger.debug("🧠 [ML-CONTEXT] Gathering ML context for %s...", symbol)
        context: Dict[str, Any] = {
            "is_healthy": False,
            "prediction": None,
            "regime": None,
            "reason": "",
        }
        price_data = None

        if not self.market_data_pipeline:
            context["reason"] = "MarketDataPipeline not available"
        else:
            try:
                price_data = await self.market_data_pipeline.get_latest_ohlcv(
                    symbol=symbol, timeframe=horizon, exchange=None, limit=2000
                )
            except Exception as e:  # noqa: BLE001
                context["reason"] = f"OHLCV data fetch error: {e}"
                price_data = None

        if self.price_engine:
            try:
                prediction = self.price_engine.get_price_forecast(symbol)
                context["prediction"] = prediction
            except Exception as e:  # noqa: BLE001
                prev = context.get("reason", "")
                context["reason"] = (prev + "; " if prev else "") + f"Price forecast error: {e}"

        if self.regime_predictor and price_data is not None:
            try:
                regime = await self.regime_predictor.predict_regime_transition(
                    symbol=symbol, price_data=price_data, horizon=horizon
                )
                context["regime"] = regime
            except Exception as e:  # noqa: BLE001
                prev = context.get("reason", "")
                context["reason"] = (prev + "; " if prev else "") + f"Regime prediction error: {e}"

        if not context.get("reason"):
            context["is_healthy"] = True
            context["reason"] = "ML context successfully gathered."
        else:
            logger.warning(
                "🧠 [ML-CONTEXT] ML context for %s is unhealthy. Reasons: %s",
                symbol,
                context["reason"],
            )

        return context

    def get_integration_status(self) -> Dict[str, Any]:
        price_engine_status: Dict[str, Any]
        if self.price_engine:
            try:
                price_engine_status = self.price_engine.get_engine_status()
            except AttributeError as e:
                price_engine_status = {
                    "status": "uninitialized",
                    "reason": f"Missing attribute: {e}",
                    "loaded_models": 0,
                }
            except Exception as e:  # noqa: BLE001
                price_engine_status = {"status": "error", "reason": str(e)}
        else:
            price_engine_status = {
                "status": "disabled",
                "reason": "Price engine not provided",
            }

        return {
            "price_engine": price_engine_status,
            "performance": self.tracker.get_performance_summary(),
            "active": bool(self.adapter is not None and self.price_engine is not None),
        }
