"""
Strategy Integration Layer for Advanced Price Prediction.

Integrates price forecasts with existing trading strategies for AI-enhanced
decision making and risk management.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

from .price_predictor import AdvancedPricePredictionEngine
from .regime_predictor import MLRegimePredictor

# Use TYPE_CHECKING to avoid runtime import errors while preserving type hints
if TYPE_CHECKING:
    from core.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


# ... AIEnhancedStrategyAdapter ve StrategyPerformanceTracker sınıfları hiç değişmeden kalıyor ...
# ... Bu sınıflar önceki yanıtta olduğu gibi buradadır, kısalık için gizlenmiştir ...
class AIEnhancedStrategyAdapter:
    """
    Adapter that enhances existing trading strategies with AI predictions.
    
    Combines price forecasts, regime predictions, and confidence intervals
    to improve strategy signals and risk management.
    """
    
    def __init__(self, price_engine: AdvancedPricePredictionEngine,
                 regime_predictor: MLRegimePredictor):
        """
        Initialize strategy adapter.
        
        Args:
            price_engine: Advanced price prediction engine
            regime_predictor: Market regime predictor
        """
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        
        # Configuration
        self.min_confidence = 0.6
        self.min_consensus = 0.7
        self.risk_scaling_factor = 1.5
        
        logger.info("AI-Enhanced Strategy Adapter initialized")

    def _normalize_signal(self, base_signal: Any) -> Dict[str, Any]:
        """
        Normalizes various incoming signal formats into a consistent internal schema.
        Örnek: {'side': 'buy', 'rr_ratio': 2.5} -> {'signal': 'bullish', 'strength': 0.725}
        """
        # Durum 1: Sinyal zaten 'signal' anahtarı ile doğru formatta.
        if isinstance(base_signal, dict) and 'signal' in base_signal:
            return {
                'signal': str(base_signal['signal']).lower(),
                'strength': float(base_signal.get('strength', 0.6))
            }

        # Durum 2: Sinyal {'side': 'buy', ...} formatında. (En yaygın durum)
        if isinstance(base_signal, dict) and 'side' in base_signal:
            side = str(base_signal['side']).lower()
            
            if side in ('buy', 'long'):
                signal_direction = 'bullish'
            elif side in ('sell', 'short'):
                signal_direction = 'bearish'
            else:
                signal_direction = 'neutral'

            # 'strength' (güç) değerini türet. Öncelik: rr_ratio.
            if 'rr_ratio' in base_signal:
                rr_ratio = float(base_signal.get('rr_ratio', 1.0))
                # rr_ratio'yu 1.0-3.0 arasından 0.5-0.8 aralığına haritala.
                strength = 0.5 + (max(0, rr_ratio - 1.0) * 0.15)
                strength = max(0.5, min(0.8, strength)) # Güvenli aralıkta kalmasını sağla.
            else:
                strength = 0.6  # Varsayılan güç.

            return {'signal': signal_direction, 'strength': strength}

        # Durum 3: Sinyal sadece bir string ('buy', 'sell' vb.)
        if isinstance(base_signal, str):
            side = base_signal.lower()
            if side in ('buy', 'long'): signal_direction = 'bullish'
            elif side in ('sell', 'short'): signal_direction = 'bearish'
            else: signal_direction = 'neutral'
            return {'signal': signal_direction, 'strength': 0.7}

        # Durum 4: Tanınmayan format. Pass-through için 'neutral' yap.
        logger.warning(f"🧠 [ML-ADAPTER] Uyumsuz sinyal formatı: {base_signal}. Pass-through için nötr kabul ediliyor.")
        return {'signal': 'neutral', 'strength': 0.0}
    
    async def enhance_strategy_signal(self, symbol: str, base_signal: Dict[str, Any],
                                     current_price: float) -> Dict[str, Any]:
        """
        Enhance a base trading strategy signal with AI predictions.
        (GÜNCELLENDİ: Sinyal normalizasyonu ve pass-through mantığı eklendi)
        """
        try:
            # 1. Gelen sinyali standart bir formata dönüştür.
            processed_signal = self._normalize_signal(base_signal)

            # 2. Varsayılan 'enhancement' objesini oluştur. Başlangıçta final sinyal, işlenmiş sinyalin aynısıdır.
            enhancement = {
                'original_signal': processed_signal['signal'],
                'original_strength': processed_signal.get('strength', 0.6),
                'ai_signal': 'neutral',
                'ai_strength': 0.0,
                'final_signal': processed_signal['signal'], # Pass-through için başlangıç değeri
                'final_strength': processed_signal.get('strength', 0.6), # Pass-through için başlangıç değeri
                'confidence_adjustment': 1.0,
                'risk_adjustment': 1.0,
                'recommendations': []
            }

            # 3. Fiyat tahmini almayı dene.
            price_forecast = self.price_engine.get_price_forecast(symbol)

            # 4. Fiyat tahmini yoksa, sinyali olduğu gibi geri döndür (Pass-through).
            if not price_forecast:
                enhancement['recommendations'].append('No AI forecast available – pass-through')
                logger.info(f"🧠 [ML-ADAPTER] {symbol} için AI tahmini yok. Sinyal olduğu gibi geçiriliyor.")
                return enhancement

            # 5. Fiyat tahmini varsa, AI sinyali üret ve base sinyal ile birleştir.
            ai_signal = self.price_engine.generate_trading_signals(symbol, current_price)
            enhancement['ai_signal'] = ai_signal.get('signal', 'neutral')
            enhancement['ai_strength'] = float(ai_signal.get('strength', 0.0))

            combined = self._combine_signals(processed_signal, ai_signal, price_forecast)
            enhancement.update(combined)
            
            logger.info(f"🧠 [ML-ADAPTER] Sinyal zenginleştirildi: {enhancement['original_signal']} -> {enhancement['final_signal']}")
            return enhancement

        except Exception as e:
            logger.error(f"Error enhancing strategy signal: {e}", exc_info=True)
            # Hata durumunda sinyali veto ETME, olduğu gibi geçir (Pass-through).
            original_signal_val = 'unknown'
            if isinstance(base_signal, dict):
                original_signal_val = base_signal.get('side') or base_signal.get('signal', 'unknown')
            elif isinstance(base_signal, str):
                original_signal_val = base_signal

            return {
                'original_signal': original_signal_val,
                'final_signal': original_signal_val,
                'final_strength': float(base_signal.get('strength', 0.6)) if isinstance(base_signal, dict) else 0.6,
                'recommendations': [f'Enhancement failed: {e}', 'Pass-through'],
                'error': str(e)
            }
    
    def _combine_signals(self, base_signal: Dict[str, Any],
                        ai_signal: Dict[str, Any],
                        forecast: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine base strategy signal with AI predictions.
        
        Uses a weighted approach based on confidence and consensus.
        """
        base_strength = base_signal.get('strength', 0.5)
        ai_strength = ai_signal.get('strength', 0.0)
        ai_confidence = ai_signal.get('confidence', 0.0)
        consensus = ai_signal.get('consensus', 0.0)
        
        # Calculate weights
        base_weight = 0.6
        ai_weight = 0.4 * ai_confidence
        total_weight = base_weight + ai_weight
        if total_weight == 0: total_weight = 1 # Prevent division by zero

        base_weight /= total_weight
        ai_weight /= total_weight
        
        combined_strength = base_strength * base_weight + ai_strength * ai_weight
        
        base_direction = self._signal_to_direction(base_signal['signal'])
        ai_direction = self._signal_to_direction(ai_signal['signal'])
        
        if base_direction == ai_direction and base_direction != 0:
            final_signal = base_signal['signal']
            final_strength = combined_strength * 1.2
            recommendations = ['Base strategy and AI forecast agree - strong signal']
        elif abs(base_direction - ai_direction) > 1:
            final_signal = 'neutral'
            final_strength = 0.0
            recommendations = ['Conflicting signals - recommend caution']
        else:
            if base_direction == 0:
                final_signal = ai_signal['signal']
                final_strength = ai_strength * ai_confidence
            else:
                final_signal = base_signal['signal']
                final_strength = base_strength * 0.8
            recommendations = ['Partial agreement - moderate confidence']
        
        if consensus < self.min_consensus:
            final_strength *= 0.7
            recommendations.append(f'Low timeframe consensus ({consensus:.2f})')
        
        uncertainty = ai_signal.get('uncertainty', 1.0)
        risk_adjustment = 1.0 / (1.0 + uncertainty * self.risk_scaling_factor)
        confidence_adjustment = ai_confidence if consensus > self.min_consensus else ai_confidence * 0.8
        
        return {
            'final_signal': final_signal,
            'final_strength': min(final_strength, 1.0),
            'confidence_adjustment': confidence_adjustment,
            'risk_adjustment': risk_adjustment,
            'recommendations': recommendations,
            'forecast_price': ai_signal.get('forecast_price', None),
            'uncertainty': uncertainty,
            'consensus': consensus
        }
    
    def _signal_to_direction(self, signal: str) -> int:
        """Convert signal to numeric direction."""
        if signal in ['bullish', 'long', 'buy']:
            return 1
        elif signal in ['bearish', 'short', 'sell']:
            return -1
        else:
            return 0
    
    def calculate_position_sizing(self, symbol: str, base_position: float,
                                  enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate AI-adjusted position sizing.
        
        Args:
            symbol: Trading symbol
            base_position: Base position size from strategy
            enhancement: Signal enhancement data
            
        Returns:
            Adjusted position sizing with risk metrics
        """
        confidence_adj = enhancement.get('confidence_adjustment', 1.0)
        risk_adj = enhancement.get('risk_adjustment', 1.0)
        
        adjusted_position = base_position * confidence_adj * risk_adj
        
        max_position = base_position * 1.5
        adjusted_position = min(adjusted_position, max_position)
        
        return {
            'base_position': base_position,
            'adjusted_position': adjusted_position,
            'confidence_multiplier': confidence_adj,
            'risk_multiplier': risk_adj,
            'final_multiplier': adjusted_position / base_position if base_position > 0 else 1.0
        }
    
    def get_risk_metrics(self, symbol: str,
                        forecast: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate risk metrics from AI predictions.
        
        Args:
            symbol: Trading symbol
            forecast: Optional price forecast (fetched if not provided)
            
        Returns:
            Dictionary with risk metrics
        """
        if forecast is None:
            forecast = self.price_engine.get_price_forecast(symbol)
        
        if not forecast:
            return {
                'risk_level': 'unknown',
                'uncertainty': 1.0,
                'confidence': 0.0
            }
        
        agg = forecast['aggregated']
        uncertainty = float(np.mean(agg['uncertainty']))
        consensus = agg['consensus_strength']
        
        if uncertainty < 0.02 and consensus > 0.8:
            risk_level = 'low'
        elif uncertainty < 0.05 and consensus > 0.6:
            risk_level = 'moderate'
        else:
            risk_level = 'high'
        
        return {
            'risk_level': risk_level,
            'uncertainty': uncertainty,
            'consensus': consensus,
            'confidence': 1.0 / (1.0 + uncertainty)
        }


class StrategyPerformanceTracker:
    """
    Track performance of AI-enhanced vs base strategies.
    
    Monitors improvement metrics and provides feedback for continuous improvement.
    """
    
    def __init__(self):
        """Initialize performance tracker."""
        self.trades = []
        self.metrics = {
            'total_trades': 0,
            'base_strategy_wins': 0,
            'ai_enhanced_wins': 0,
            'improvement_rate': 0.0
        }
        
        logger.info("Strategy Performance Tracker initialized")
    
    def record_trade(self, trade: Dict[str, Any]):
        """
        Record a completed trade.
        
        Args:
            trade: Trade information including strategy type and outcome
        """
        self.trades.append({
            **trade,
            'timestamp': pd.Timestamp.now()
        })
        
        self.metrics['total_trades'] += 1
        
        if trade.get('strategy_type') == 'base' and trade.get('pnl', 0) > 0:
            self.metrics['base_strategy_wins'] += 1
        elif trade.get('strategy_type') == 'ai_enhanced' and trade.get('pnl', 0) > 0:
            self.metrics['ai_enhanced_wins'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return self.metrics
        
        df = pd.DataFrame(self.trades)
        
        if 'base' in df['strategy_type'].values and 'ai_enhanced' in df['strategy_type'].values:
            base_pnl = df[df['strategy_type'] == 'base']['pnl'].mean()
            ai_pnl = df[df['strategy_type'] == 'ai_enhanced']['pnl'].mean()
            
            if base_pnl != 0:
                improvement = ((ai_pnl - base_pnl) / abs(base_pnl)) * 100
                self.metrics['improvement_rate'] = improvement
        
        return self.metrics
    
    def get_recent_performance(self, n_trades: int = 100) -> pd.DataFrame:
        """
        Get recent trade performance.
        
        Args:
            n_trades: Number of recent trades to return
            
        Returns:
            DataFrame with recent trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades).tail(n_trades)

class MLStrategyIntegrationManager:
    """
    Main integration manager coordinating all ML enhancements.
    
    Provides a unified, fault-tolerant interface for strategy enhancement, 
    risk management, and performance tracking.
    """
    
    def __init__(self, 
                 price_engine: Optional[AdvancedPricePredictionEngine],
                 regime_predictor: Optional[MLRegimePredictor],
                 market_data_pipeline: Optional[Any] = None): # *** UPDATED: Use MarketDataPipeline instead of websocket_manager ***
        """
        Initialize the integration manager.
        
        Args:
            price_engine: Price prediction engine. Can be None.
            regime_predictor: Regime prediction engine. Can be None.
            market_data_pipeline: MarketDataPipeline for centralized data access. Can be None.
        """
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        self.market_data_pipeline = market_data_pipeline # *** UPDATED: Store MarketDataPipeline ***
        
        # Keep backward compatibility with websocket_manager attribute
        self.websocket_manager = market_data_pipeline.websocket_manager if market_data_pipeline else None
        
        self.adapter = None
        # Adapter'ın başlatılması için market_data_pipeline bir bağımlılık değil, bu yüzden mantık aynı kalıyor.
        if self.price_engine and self.regime_predictor:
            self.adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor)
            logger.info("✅ AIEnhancedStrategyAdapter initialized.")
        else:
            logger.warning("⚠️ AIEnhancedStrategyAdapter not initialized due to missing dependencies (price_engine or regime_predictor).")

        self.tracker = StrategyPerformanceTracker()
        
        logger.info("ML Strategy Integration Manager initialized")

    async def get_ml_context(self, symbol: str, horizon: str = '1h') -> Dict[str, Any]:
        """
        Gathers ML-driven context for a symbol, including price predictions and regime analysis.
        
        *** UPDATED: Now uses centralized MarketDataPipeline for data access ***
        This eliminates direct websocket_manager access and ensures consistent data flow.
        """
        logger.info(f"🧠 [ML-CONTEXT] Gathering ML context for {symbol}...")
        context = {
            'is_healthy': False,
            'prediction': None,
            'regime': None,
            'confidence': 0.0,
            'reason': "Initialization failed"
        }

        # Adım 1: Check if MarketDataPipeline is available
        if not self.market_data_pipeline:
            context['reason'] = "MarketDataPipeline is not available."
            logger.warning(f"🧠 [ML-CONTEXT] {context['reason']}")
            return context
        
        # Adım 2: Get OHLCV data using centralized MarketDataPipeline
        # This automatically handles WebSocket-first with REST fallback
        try:
            price_data = await self.market_data_pipeline.get_latest_ohlcv(
                symbol=symbol, 
                timeframe=horizon,
                exchange=None  # Let pipeline choose best exchange
            )
            
            if price_data is None or price_data.empty:
                context['reason'] = f"Could not retrieve OHLCV data for {symbol} from MarketDataPipeline (tried {horizon})."
                logger.warning(f"🧠 [ML-CONTEXT] {context['reason']}")
                return context
            
            logger.debug(f"🧠 [ML-CONTEXT] Retrieved {len(price_data)} candles for {symbol} via MarketDataPipeline")
            
        except Exception as e:
            context['reason'] = f"Failed to fetch OHLCV data from MarketDataPipeline: {e}"
            logger.error(f"🧠 [ML-CONTEXT] {context['reason']}", exc_info=False)
            return context

        # Adım 3: Fiyat Tahmini yap (mevcut mantık aynı kalıyor).
        if self.price_engine:
            try:
                # Not: price_engine.predict'in de 'await' gerektirdiğini varsayıyoruz.
                prediction = await self.price_engine.predict(symbol, horizon=horizon)
                if prediction and 'predicted_price' in prediction:
                    context['prediction'] = prediction
                    context['confidence'] = prediction.get('confidence', 0.0)
            except Exception as e:
                logger.error(f"🧠 [ML-CONTEXT] Price prediction crashed for {symbol}: {e}", exc_info=False)

        # Adım 4: Rejim Tahmini yap (doğru veri formatıyla).
        if self.regime_predictor:
            try:
                # Artık elimizde olan DataFrame'i veriyoruz.
                regime = await self.regime_predictor.predict_regime_transition(
                    symbol=symbol, 
                    price_data=price_data, 
                    horizon=horizon
                )
                context['regime'] = regime
                logger.info(f"🧠 [ML-CONTEXT] Market regime for {symbol}: {regime.get('predicted_regime', 'unknown')}")
            except Exception as e:
                logger.error(f"🧠 [ML-CONTEXT] Regime prediction crashed for {symbol}: {e}", exc_info=False)

        # Adım 5: Sağlık durumunu belirle.
        if context['prediction'] and context['regime']:
            context['is_healthy'] = True
            context['reason'] = "ML context successfully gathered."
        else:
            context['reason'] = "One or more ML components failed to produce output."
        
        if not context['is_healthy']:
             logger.warning(f"🧠 [ML-CONTEXT] ML context for {symbol} is unhealthy. Reason: {context['reason']}")

        return context

    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get overall integration status, now with fault tolerance.
        
        Returns:
            Status information for all components.
        """
        price_engine_status = {}
        if self.price_engine:
            try:
                price_engine_status = self.price_engine.get_engine_status()
            except AttributeError as e:
                price_engine_status = {
                    'status': 'uninitialized', 
                    'reason': f'Missing attribute, likely models not loaded: {e}',
                    'loaded_models': 0
                }
            except Exception as e:
                price_engine_status = {'status': 'error', 'reason': str(e)}
        else:
            price_engine_status = {'status': 'disabled', 'reason': 'Price engine not provided'}

        return {
            'price_engine': price_engine_status,
            'performance': self.tracker.get_performance_summary(),
            'active': self.adapter is not None and self.price_engine is not None
        }
