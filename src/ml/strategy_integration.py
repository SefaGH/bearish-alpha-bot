"""
Strategy Integration Layer for Advanced Price Prediction.

Integrates price forecasts and regime predictions with existing trading strategies 
for AI-enhanced decision making and risk management.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING
import logging

from .price_predictor import AdvancedPricePredictionEngine
from .regime_predictor import MLRegimePredictor

if TYPE_CHECKING:
    from core.production_coordinator import ProductionCoordinator
    from core.market_data_pipeline import MarketDataPipeline

logger = logging.getLogger(__name__)


class AIEnhancedStrategyAdapter:
    """
    Adapter that enhances existing trading strategies with AI predictions.
    This class is the specialized worker responsible for combining ML insights
    with a base trading signal.
    """
    
    def __init__(self, price_engine: AdvancedPricePredictionEngine,
                 regime_predictor: MLRegimePredictor, 
                 config: Dict[str, Any]):
        """ Initialize the adapter with necessary ML engines and configuration. """
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        self.config = config
        
        pred_config = self.config.get('prediction', {})
        self.min_confidence = pred_config.get('min_confidence_threshold', 0.6)
        self.min_consensus = pred_config.get('consensus_threshold', 0.7)
        self.risk_scaling_factor = pred_config.get('risk_scaling_factor', 1.5)
        
        logger.info("AI-Enhanced Strategy Adapter initialized")
        logger.info(f"   - Adapter min regime confidence threshold: {self.min_confidence}")

    async def enhance_strategy_signal(self, symbol: str, base_signal: Dict[str, Any],
                                     current_price: float, market_data_pipeline: "MarketDataPipeline") -> Dict[str, Any]:
        """
        The core method for signal enhancement. It fetches both regime and price
        predictions and combines them to produce a final, enhanced signal.
        """
        try:
            processed_signal = self._normalize_signal(base_signal)
            enhancement = {'original_signal': processed_signal['signal']}

            # --- ADIM 1: REJİM TAHMİNİNİ AL ---
            regime_info = None
            if self.regime_predictor and market_data_pipeline:
                try:
                    price_data = await market_data_pipeline.get_latest_ohlcv(symbol, timeframe='1h')
                    if price_data is not None and not price_data.empty:
                        regime_info = await self.regime_predictor.predict_regime_transition(symbol, price_data, horizon='1h')
                    else:
                        logger.warning(f"🧠 [ML-ADAPTER] Regime prediction skipped: No 1h data for {symbol}.")
                except Exception as e:
                    logger.error(f"🧠 [ML-ADAPTER] Regime prediction step failed: {e}", exc_info=True)
            
            # --- ADIM 2: REJİM BİLGİSİNİ FİLTRELE VE SONUCA EKLE ---
            if regime_info:
                regime_confidence = regime_info.get('confidence', 0.0)
                if regime_confidence >= self.min_confidence:
                    predicted_regime = regime_info.get('predicted_regime', 'neutral')
                    enhancement['predicted_regime'] = predicted_regime
                    enhancement['regime_confidence'] = regime_confidence
                    logger.info(f"🧠 [ML-ADAPTER] Regime for {symbol} is {predicted_regime.upper()} (Conf: {regime_confidence:.2f}) - PASSED")
                else:
                    logger.info(f"🧠 [ML-ADAPTER] Regime for {symbol} discarded (Conf: {regime_confidence:.2f} < {self.min_confidence})")
            
            # --- ADIM 3: FİYAT TAHMİNİNİ AL VE SİNYALİ BİRLEŞTİR ---
            price_forecast = self.price_engine.get_price_forecast(symbol)
            if price_forecast:
                ai_signal = self.price_engine.generate_trading_signals(symbol, current_price)
                combined = self._combine_signals(processed_signal, ai_signal, price_forecast)
                enhancement.update(combined)
                logger.info(f"🧠 [ML-ADAPTER] Price forecast used to enhance signal for {symbol}.")
            else:
                logger.info(f"🧠 [ML-ADAPTER] No price forecast for {symbol}. Enhancement based on regime only (if available).")

            return enhancement

        except Exception as e:
            logger.error(f"CRITICAL: Unhandled error in enhance_strategy_signal: {e}", exc_info=True)
            return {'original_signal': base_signal.get('side', 'unknown'), 'error': str(e)}

    def _normalize_signal(self, base_signal: Any) -> Dict[str, Any]:
        """ Normalizes various incoming signal formats. """
        if isinstance(base_signal, dict) and 'signal' in base_signal:
            return {'signal': str(base_signal['signal']).lower(), 'strength': float(base_signal.get('strength', 0.6))}
        if isinstance(base_signal, dict) and 'side' in base_signal:
            side = str(base_signal['side']).lower()
            signal_direction = 'bullish' if side in ('buy', 'long') else 'bearish' if side in ('sell', 'short') else 'neutral'
            if 'rr_ratio' in base_signal:
                rr_ratio = float(base_signal.get('rr_ratio', 1.0))
                strength = 0.5 + (max(0, rr_ratio - 1.0) * 0.15); strength = max(0.5, min(0.8, strength))
            else: strength = 0.6
            return {'signal': signal_direction, 'strength': strength}
        if isinstance(base_signal, str):
            side = base_signal.lower()
            signal_direction = 'bullish' if side in ('buy', 'long') else 'bearish' if side in ('sell', 'short') else 'neutral'
            return {'signal': signal_direction, 'strength': 0.7}
        logger.warning(f"🧠 [ML-ADAPTER] Unrecognized signal format: {base_signal}. Treating as neutral.")
        return {'signal': 'neutral', 'strength': 0.0}

    def _combine_signals(self, base_signal: Dict[str, Any], ai_signal: Dict[str, Any], forecast: Dict[str, Any]) -> Dict[str, Any]:
        """ Combines base strategy signal with AI price predictions. """
        base_strength=base_signal.get('strength',0.5);ai_strength=ai_signal.get('strength',0.0);ai_confidence=ai_signal.get('confidence',0.0);consensus=ai_signal.get('consensus',0.0);base_weight=0.6;ai_weight=0.4*ai_confidence;total_weight=base_weight+ai_weight
        if total_weight == 0: total_weight = 1
        base_weight/=total_weight;ai_weight/=total_weight;combined_strength=base_strength*base_weight+ai_strength*ai_weight;base_direction=self._signal_to_direction(base_signal['signal']);ai_direction=self._signal_to_direction(ai_signal['signal'])
        if base_direction==ai_direction and base_direction!=0:final_signal=base_signal['signal'];final_strength=combined_strength*1.2;recommendations=['Base strategy and AI forecast agree - strong signal']
        elif abs(base_direction-ai_direction)>1:final_signal='neutral';final_strength=0.0;recommendations=['Conflicting signals - recommend caution']
        else:
            if base_direction==0:final_signal=ai_signal['signal'];final_strength=ai_strength*ai_confidence
            else:final_signal=base_signal['signal'];final_strength=base_strength*0.8
            recommendations=['Partial agreement - moderate confidence']
        if consensus<self.min_consensus:final_strength*=0.7;recommendations.append(f'Low timeframe consensus ({consensus:.2f})')
        uncertainty=ai_signal.get('uncertainty',1.0);risk_adjustment=1.0/(1.0+uncertainty*self.risk_scaling_factor);confidence_adjustment=ai_confidence if consensus>self.min_consensus else ai_confidence*0.8
        return{'final_signal':final_signal,'final_strength':min(final_strength,1.0),'confidence_adjustment':confidence_adjustment,'risk_adjustment':risk_adjustment,'recommendations':recommendations,'forecast_price':ai_signal.get('forecast_price',None),'uncertainty':uncertainty,'consensus':consensus}

    def _signal_to_direction(self, signal: str) -> int:
        """ Converts signal string to a numeric direction. """
        if signal in ['bullish', 'long', 'buy']: return 1
        elif signal in ['bearish', 'short', 'sell']: return -1
        else: return 0


class MLStrategyIntegrationManager:
    """
    Main integration manager coordinating all ML enhancements.
    This class acts as a simplified FACADE, holding the adapter and other
    ML components, and exposing them to the rest of the system.
    """
    
    def __init__(self, 
                 price_engine: Optional[AdvancedPricePredictionEngine],
                 regime_predictor: Optional[MLRegimePredictor],
                 config: Dict[str, Any],
                 market_data_pipeline: Optional["MarketDataPipeline"] = None):
        """ Initializes the manager and its components. """
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        self.market_data_pipeline = market_data_pipeline
        self.config = config
        
        self.adapter = None
        if self.price_engine and self.regime_predictor:
            self.adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor, self.config)
            logger.info("✅ AIEnhancedStrategyAdapter created and linked within the manager.")
        else:
            missing = [name for name, dep in [("price_engine", self.price_engine), ("regime_predictor", self.regime_predictor)] if not dep]
            logger.warning(f"⚠️ AIEnhancedStrategyAdapter not created due to missing dependencies: {', '.join(missing)}.")
        
        logger.info("ML Strategy Integration Manager initialized.")

    async def enhance_strategy_signal(self, symbol: str, base_signal: Dict[str, Any],
                                     current_price: float) -> Dict[str, Any]:
        """
        Facade method that calls the adapter's enhancement method.
        This is the single, clean entry point for the StrategyCoordinator.
        """
        if self.adapter:
            return await self.adapter.enhance_strategy_signal(
                symbol,
                base_signal,
                current_price,
                market_data_pipeline=self.market_data_pipeline
            )
        
        logger.warning("ML Adapter not available. Returning original signal without enhancement.")
        return {'original_signal': base_signal.get('side', 'unknown')}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """ Provides a health check of the ML integration components. """
        price_engine_status = {'status': 'disabled'}
        if self.price_engine:
            try: price_engine_status = self.price_engine.get_engine_status()
            except Exception as e: price_engine_status = {'status': 'error', 'reason': str(e)}

        regime_predictor_status = {'status': 'disabled'}
        if self.regime_predictor and hasattr(self.regime_predictor, 'is_trained'):
            regime_predictor_status = {'status': 'enabled', 'trained': self.regime_predictor.is_trained}

        return {
            'adapter_status': 'active' if self.adapter else 'inactive',
            'price_engine': price_engine_status,
            'regime_predictor': regime_predictor_status,
        }
