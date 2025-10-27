"""
Strategy Integration Layer for Advanced Price Prediction.

Integrates price forecasts with existing trading strategies for AI-enhanced
decision making and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

from .price_predictor import AdvancedPricePredictionEngine
from .regime_predictor import MLRegimePredictor
from .ml_context import MLContext

logger = logging.getLogger(__name__)


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
    
    async def enhance_strategy_signal(self, symbol: str, base_signal: Dict[str, Any],
                                     current_price: float) -> Dict[str, Any]:
        """
        Enhance a base trading strategy signal with AI predictions.
        
        Args:
            symbol: Trading symbol
            base_signal: Original strategy signal
            current_price: Current market price
            
        Returns:
            Enhanced signal with AI adjustments
        """
        try:
            logger.debug(f"🧠 [ML-ADAPTER] Enhancing signal for {symbol} at ${current_price:.2f}")
            
            # Get price forecast
            price_forecast = self.price_engine.get_price_forecast(symbol)
            
            # Get regime prediction
            regime_data = pd.DataFrame()  # Placeholder - would come from data feed
            
            # Calculate enhancement factors
            enhancement = {
                'original_signal': base_signal['signal'],
                'original_strength': base_signal.get('strength', 0.5),
                'ai_signal': 'neutral',
                'ai_strength': 0.0,
                'final_signal': base_signal['signal'],
                'final_strength': base_signal.get('strength', 0.5),
                'confidence_adjustment': 1.0,
                'risk_adjustment': 1.0,
                'recommendations': []
            }
            
            logger.debug(f"🧠 [ML-ADAPTER] Base signal: {base_signal['signal']} (strength: {base_signal.get('strength', 0.5):.2f})")
            
            # If no AI predictions available, return base signal
            if not price_forecast:
                enhancement['recommendations'].append('No AI forecast available')
                logger.debug(f"🧠 [ML-ADAPTER] No price forecast available, using base signal")
                return enhancement
            
            # Get AI signal from price forecast
            ai_signal = self.price_engine.generate_trading_signals(
                symbol, current_price
            )
            
            enhancement['ai_signal'] = ai_signal['signal']
            enhancement['ai_strength'] = ai_signal['strength']
            
            logger.debug(f"🧠 [ML-ADAPTER] AI signal: {ai_signal['signal']} (strength: {ai_signal['strength']:.2f})")
            
            # Combine signals
            combined = self._combine_signals(base_signal, ai_signal, price_forecast)
            
            enhancement.update(combined)
            
            logger.debug(f"🧠 [ML-ADAPTER] Signal enhancement: {base_signal['signal']} → {enhancement['final_signal']} (strength: {enhancement['final_strength']:.2f})")
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Error enhancing strategy signal: {e}")
            logger.debug(f"🧠 [ML-ADAPTER] Enhancement error: {e}")
            return {
                'original_signal': base_signal['signal'],
                'final_signal': base_signal['signal'],
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
        ai_strength = ai_signal['strength']
        ai_confidence = ai_signal['confidence']
        consensus = ai_signal['consensus']
        
        # Calculate weights
        base_weight = 0.6  # Base strategy gets 60% weight
        ai_weight = 0.4 * ai_confidence  # AI weight scaled by confidence
        
        total_weight = base_weight + ai_weight
        
        # Normalize weights
        base_weight /= total_weight
        ai_weight /= total_weight
        
        # Combine strengths
        combined_strength = base_strength * base_weight + ai_strength * ai_weight
        
        # Determine final signal
        base_direction = self._signal_to_direction(base_signal['signal'])
        ai_direction = self._signal_to_direction(ai_signal['signal'])
        
        # Check agreement
        if base_direction == ai_direction:
            final_signal = base_signal['signal']
            final_strength = combined_strength * 1.2  # Boost when both agree
            recommendations = ['Base strategy and AI forecast agree - strong signal']
        elif abs(base_direction - ai_direction) > 1:  # Opposite signals
            final_signal = 'neutral'
            final_strength = 0.0
            recommendations = ['Conflicting signals - recommend caution']
        else:  # One neutral
            if base_direction == 0:
                final_signal = ai_signal['signal']
                final_strength = ai_strength * ai_confidence
            else:
                final_signal = base_signal['signal']
                final_strength = base_strength * 0.8  # Reduce strength
            recommendations = ['Partial agreement - moderate confidence']
        
        # Adjust by consensus
        if consensus < self.min_consensus:
            final_strength *= 0.7
            recommendations.append(f'Low timeframe consensus ({consensus:.2f})')
        
        # Calculate risk adjustments
        uncertainty = ai_signal['uncertainty']
        risk_adjustment = 1.0 / (1.0 + uncertainty * self.risk_scaling_factor)
        
        # Confidence adjustment
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
        # Apply confidence and risk adjustments
        confidence_adj = enhancement.get('confidence_adjustment', 1.0)
        risk_adj = enhancement.get('risk_adjustment', 1.0)
        
        # Calculate adjusted position
        adjusted_position = base_position * confidence_adj * risk_adj
        
        # Cap at 1.5x base position for safety
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
        
        # Classify risk level
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
        
        # Update win counts
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
        
        # Calculate improvement
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
    
    Provides unified interface for strategy enhancement, risk management,
    and performance tracking.
    """
    
    def __init__(self, price_engine: AdvancedPricePredictionEngine,
                 regime_predictor: MLRegimePredictor):
        """
        Initialize integration manager.
        
        Args:
            price_engine: Price prediction engine
            regime_predictor: Regime prediction engine
        """
        self.adapter = AIEnhancedStrategyAdapter(price_engine, regime_predictor)
        self.tracker = StrategyPerformanceTracker()
        self.price_engine = price_engine
        self.regime_predictor = regime_predictor
        
        logger.info("ML Strategy Integration Manager initialized")
    
    async def get_ml_context(self, symbol: str, market_data: Dict[str, Any], 
                           indicator_validator=None) -> MLContext:
        """
        Generate ML context for a trading decision.
        
        This is the main entry point for getting ML predictions. It orchestrates
        all ML components and produces a unified MLContext with validation.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary with price_data, indicators, etc.
            indicator_validator: Optional IndicatorValidator for data validation
            
        Returns:
            MLContext with all ML predictions and health status
        """
        from datetime import datetime
        
        # Initialize context
        context = MLContext(
            symbol=symbol,
            timestamp=datetime.now()
        )
        
        validation_errors = []
        
        try:
            # Extract price data from market_data
            price_data = market_data.get('price_data')
            
            # Validate input data
            if price_data is None or price_data.empty:
                validation_errors.append("Missing or empty price data")
                context.is_healthy = False
                context.validation_errors = validation_errors
                logger.warning(f"🧠 [ML-CONTEXT] {symbol}: Cannot create ML context - no price data")
                return context
            
            # Use IndicatorValidator if available for thorough validation
            if indicator_validator:
                is_valid, errors = indicator_validator.validate_ml_data(price_data, symbol)
                if not is_valid:
                    validation_errors.extend(errors)
                    context.is_healthy = False
                    context.validation_errors = validation_errors
                    logger.warning(
                        f"🧠 [ML-CONTEXT] {symbol}: Data validation failed - {', '.join(errors)}"
                    )
                    return context
            else:
                # Fallback validation if no validator provided
                # Check for NaN values in critical columns
                critical_cols = ['close', 'volume', 'high', 'low', 'open']
                for col in critical_cols:
                    if col in price_data.columns and price_data[col].isna().any():
                        validation_errors.append(f"NaN values in {col}")
                
                # Check data freshness
                if len(price_data) < 50:
                    validation_errors.append(f"Insufficient data: {len(price_data)} rows (need at least 50)")
                
                # If critical validation errors, mark unhealthy and return
                if validation_errors:
                    context.is_healthy = False
                    context.validation_errors = validation_errors
                    logger.warning(
                        f"🧠 [ML-CONTEXT] {symbol}: Data validation failed - {', '.join(validation_errors)}"
                    )
                    return context
            
            # Get regime prediction
            if self.regime_predictor:
                try:
                    regime_result = await self.regime_predictor.predict_regime_transition(
                        symbol, price_data
                    )
                    
                    if regime_result:
                        context.regime_prediction = regime_result.get('predicted_regime')
                        context.regime_confidence = regime_result.get('confidence', 0.0)
                        context.regime_probabilities = regime_result.get('probabilities', {})
                        
                        logger.debug(
                            f"🧠 [ML-CONTEXT] {symbol}: Regime={context.regime_prediction} "
                            f"(conf={context.regime_confidence:.2%})"
                        )
                except Exception as e:
                    logger.warning(f"🧠 [ML-CONTEXT] {symbol}: Regime prediction failed - {e}")
                    validation_errors.append(f"Regime prediction error: {str(e)}")
            
            # Get price forecast
            if self.price_engine:
                try:
                    price_forecast = self.price_engine.get_price_forecast(symbol)
                    
                    if price_forecast:
                        context.price_forecast = price_forecast
                        
                        # Extract aggregated forecast
                        agg = price_forecast.get('aggregated', {})
                        forecast_pct = agg.get('forecast', [0])[0] if 'forecast' in agg else 0
                        
                        # Determine price direction
                        if forecast_pct > 0.02:  # 2% threshold
                            context.price_direction = 'up'
                        elif forecast_pct < -0.02:
                            context.price_direction = 'down'
                        else:
                            context.price_direction = 'neutral'
                        
                        context.price_confidence = agg.get('consensus_strength', 0.0)
                        context.uncertainty = float(agg.get('uncertainty', [1.0])[0]) if 'uncertainty' in agg else 1.0
                        
                        logger.debug(
                            f"🧠 [ML-CONTEXT] {symbol}: Price={context.price_direction} "
                            f"(conf={context.price_confidence:.2%})"
                        )
                except Exception as e:
                    logger.warning(f"🧠 [ML-CONTEXT] {symbol}: Price prediction failed - {e}")
                    validation_errors.append(f"Price prediction error: {str(e)}")
            
            # Calculate consensus score
            predictions = []
            confidences = []
            
            if context.regime_prediction:
                predictions.append(context.regime_prediction)
                confidences.append(context.regime_confidence)
            
            if context.price_direction:
                predictions.append(context.price_direction)
                confidences.append(context.price_confidence)
            
            if predictions:
                # Map to numeric for comparison
                direction_map = {
                    'bullish': 1, 'up': 1,
                    'neutral': 0,
                    'bearish': -1, 'down': -1
                }
                
                numeric_preds = [direction_map.get(p, 0) for p in predictions]
                
                # Calculate agreement (how close predictions are)
                if len(numeric_preds) > 1:
                    variance = np.var(numeric_preds)
                    context.consensus_score = 1.0 / (1.0 + variance)
                else:
                    context.consensus_score = confidences[0] if confidences else 0.0
                
                # Quality score is weighted average of confidences
                if confidences:
                    context.quality_score = sum(confidences) / len(confidences)
            
            # Mark as healthy if we have at least some predictions
            context.is_healthy = (
                (context.regime_prediction is not None or context.price_direction is not None)
                and len(validation_errors) == 0
            )
            
            if context.is_healthy:
                logger.debug(
                    f"🧠 [ML-CONTEXT] {symbol}: ✓ HEALTHY - "
                    f"Consensus={context.consensus_score:.2%} Quality={context.quality_score:.2%}"
                )
            else:
                context.validation_errors = validation_errors or ["No predictions available"]
                logger.debug(f"🧠 [ML-CONTEXT] {symbol}: ✗ UNHEALTHY - {context.validation_errors}")
            
            return context
            
        except Exception as e:
            logger.error(f"🧠 [ML-CONTEXT] {symbol}: Critical error creating ML context - {e}")
            context.is_healthy = False
            context.validation_errors = [f"Critical error: {str(e)}"]
            return context
    
    async def process_strategy_signal(self, symbol: str,
                                     base_signal: Dict[str, Any],
                                     current_price: float,
                                     base_position: float = 1.0) -> Dict[str, Any]:
        """
        Complete signal processing with AI enhancement.
        
        Args:
            symbol: Trading symbol
            base_signal: Original strategy signal
            current_price: Current market price
            base_position: Base position size
            
        Returns:
            Complete enhanced signal with position sizing and risk metrics
        """
        # Enhance signal
        enhancement = await self.adapter.enhance_strategy_signal(
            symbol, base_signal, current_price
        )
        
        # Calculate position sizing
        position_sizing = self.adapter.calculate_position_sizing(
            symbol, base_position, enhancement
        )
        
        # Get risk metrics
        risk_metrics = self.adapter.get_risk_metrics(symbol)
        
        return {
            'symbol': symbol,
            'enhancement': enhancement,
            'position_sizing': position_sizing,
            'risk_metrics': risk_metrics,
            'timestamp': pd.Timestamp.now()
        }
    
    def record_trade_outcome(self, trade: Dict[str, Any]):
        """Record completed trade for performance tracking."""
        self.tracker.record_trade(trade)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get overall integration status.
        
        Returns:
            Status information for all components
        """
        return {
            'price_engine': self.price_engine.get_engine_status(),
            'performance': self.tracker.get_performance_summary(),
            'active': True
        }
