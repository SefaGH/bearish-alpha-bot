# src/core/adaptive_monitor.py - Yeni dosya
import logging
from datetime import datetime, timezone
from typing import Dict, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdaptiveStrategyMonitor:
    """Monitor adaptive strategy performance and thresholds."""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'signals': 0,
            'thresholds': [],
            'regimes': defaultdict(int),
            'position_multipliers': [],
            'last_update': None
        })
        self.symbol_counter = 0
        
    def record_adaptive_signal(self, symbol: str, signal: Dict):
        """Record adaptive signal stats."""
        if not signal.get('is_adaptive'):
            return
            
        stats = self.stats[symbol]
        stats['signals'] += 1
        stats['last_update'] = datetime.now(timezone.utc)
        
        # Threshold tracking
        if 'adaptive_threshold' in signal:
            stats['thresholds'].append(signal['adaptive_threshold'])
            
        # Regime tracking
        if 'market_regime' in signal:
            regime = signal['market_regime'].get('trend', 'unknown')
            stats['regimes'][regime] += 1
            
        # Position multiplier tracking
        if 'position_multiplier' in signal:
            stats['position_multipliers'].append(signal['position_multiplier'])
            
        self.symbol_counter += 1
        
        # Log every 100 symbols
        if self.symbol_counter % 100 == 0:
            self._log_statistics()
    
    def _log_statistics(self):
        """Log aggregated statistics."""
        logger.info("\n" + "="*60)
        logger.info("📊 ADAPTIVE STRATEGY STATISTICS")
        logger.info("="*60)
        
        total_signals = sum(s['signals'] for s in self.stats.values())
        logger.info(f"Total adaptive signals: {total_signals}")
        
        # Aggregate threshold stats
        all_thresholds = []
        for stats in self.stats.values():
            all_thresholds.extend(stats['thresholds'])
            
        if all_thresholds:
            avg_threshold = sum(all_thresholds) / len(all_thresholds)
            min_threshold = min(all_thresholds)
            max_threshold = max(all_thresholds)
            
            logger.info(f"RSI Thresholds:")
            logger.info(f"  Average: {avg_threshold:.1f}")
            logger.info(f"  Range: {min_threshold:.1f} - {max_threshold:.1f}")
        
        # Regime distribution
        all_regimes = defaultdict(int)
        for stats in self.stats.values():
            for regime, count in stats['regimes'].items():
                all_regimes[regime] += count
                
        if all_regimes:
            logger.info("Regime distribution:")
            for regime, count in sorted(all_regimes.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_signals) * 100 if total_signals > 0 else 0
                logger.info(f"  {regime}: {count} ({pct:.1f}%)")
        
        # Position multiplier stats
        all_multipliers = []
        for stats in self.stats.values():
            all_multipliers.extend(stats['position_multipliers'])
            
        if all_multipliers:
            avg_mult = sum(all_multipliers) / len(all_multipliers)
            logger.info(f"Position size multiplier avg: {avg_mult:.2f}")
        
        logger.info("="*60 + "\n")
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get stats for specific symbol."""
        return dict(self.stats[symbol])
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'total_signals': sum(s['signals'] for s in self.stats.values()),
            'active_symbols': len(self.stats),
            'symbols_processed': self.symbol_counter
        }

# Global instance
adaptive_monitor = AdaptiveStrategyMonitor()
