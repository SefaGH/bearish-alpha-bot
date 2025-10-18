# src/core/config_validator.py - Yeni dosya
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validate and fix configuration at startup."""
    
    REQUIRED_KEYS = {
        'signals.oversold_bounce': ['enable', 'tp_pct', 'sl_atr_mult'],
        'signals.short_the_rip': ['enable', 'tp_pct', 'sl_atr_mult'],
        'risk': ['equity_usd', 'per_trade_risk_pct'],
        'universe': ['fixed_symbols', 'auto_select']
    }
    
    ADAPTIVE_KEYS = {
        'signals.oversold_bounce': {
            'adaptive_rsi_base': 45,
            'adaptive_rsi_range': 10,
            'rsi_max': 45  # Backwards compat
        },
        'signals.short_the_rip': {
            'adaptive_rsi_base': 55,
            'adaptive_rsi_range': 15,
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
        else:
            logger.info("✅ Config validation passed")
            
        return fixed_config
