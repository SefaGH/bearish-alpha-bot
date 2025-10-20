"""
src/core/config_validator.py - FIXED VERSION
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validate and fix configuration at startup."""
    
    # ✅ FIXED: Use ATR-based fields, not tp_pct
    REQUIRED_KEYS = {
        'signals.oversold_bounce': ['enable', 'tp_atr_mult', 'sl_atr_mult'],  # ✅ Changed
        'signals.short_the_rip': ['enable', 'tp_atr_mult', 'sl_atr_mult'],   # ✅ Changed
        'risk': ['equity_usd', 'per_trade_risk_pct'],
        'universe': ['fixed_symbols', 'auto_select']
    }
    
    # ✅ NEW: Optional keys for backward compatibility
    OPTIONAL_KEYS = {
        'signals.oversold_bounce': ['tp_pct', 'min_tp_pct', 'max_sl_pct'],
        'signals.short_the_rip': ['tp_pct', 'min_tp_pct', 'max_sl_pct']
    }
    
    ADAPTIVE_KEYS = {
        'signals.oversold_bounce': {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
            'rsi_max': 45  # Backwards compat
        },
        'signals.short_the_rip': {
            'adaptive_rsi_base': 55,
            'adaptive_rsi_range': 10,  # ✅ Changed from 15 to 10
            'rsi_min': 55  # Backwards compat
        }
    }
    
    @classmethod
    def validate_and_fix(cls, config: Dict) -> Dict:
        """Validate config and add missing keys with defaults."""
        fixed_config = config.copy()
        issues = []
        
        # Check required keys
        for path, keys in cls.REQUIRED_KEYS.items():
            parts = path.split('.')
            section = fixed_config
            for part in parts[:-1]:
                if part not in section:
                    section[part] = {}
                section = section[part]
            
            for key in keys:
                if key not in section.get(parts[-1], {}):
                    issues.append(f"Missing {path}.{key}")
        
        # Fix adaptive keys
        for path, defaults in cls.ADAPTIVE_KEYS.items():
            parts = path.split('.')
            section = fixed_config
            for part in parts[:-1]:
                section = section.setdefault(part, {})
            
            strategy_config = section.setdefault(parts[-1], {})
            for key, default_value in defaults.items():
                if key not in strategy_config:
                    strategy_config[key] = default_value
                    logger.info(f"Added missing {path}.{key} = {default_value}")
        
        if issues:
            logger.warning(f"Config validation issues: {issues}")
            # ✅ NEW: Don't crash, just warn
            logger.info("✓ Proceeding with available configuration")
        else:
            logger.info("✅ Config validation passed")
            
        return fixed_config
