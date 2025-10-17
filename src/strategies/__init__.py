# Trading strategies for Bearish Alpha Bot

# Default market regime data for adaptive strategies
# Used when regime data is not available
DEFAULT_MARKET_REGIME = {
    'trend': 'neutral',
    'momentum': 'sideways',
    'volatility': 'normal',
    'micro_trend_strength': 0.5,    # ✅ EKLE
    'entry_score': 0.5,              # ✅ EKLE  
    'risk_multiplier': 1.0           # ✅ EKLE
}

# ✅ STRATEGY IMPORT'LARI EKLE
from .oversold_bounce import OversoldBounce
from .short_the_rip import ShortTheRip
from .adaptive_ob import AdaptiveOversoldBounce
from .adaptive_str import AdaptiveShortTheRip

# ✅ EXPORT LİSTESİ EKLE
__all__ = [
    'OversoldBounce',
    'ShortTheRip',
    'AdaptiveOversoldBounce',
    'AdaptiveShortTheRip',
    'DEFAULT_MARKET_REGIME'
]
