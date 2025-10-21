"""
Live Trading Configuration with Environment Variable Support.
Configuration settings for production live trading.

Priority Order:
1. Environment Variables (GitHub Secrets/Variables) - HIGHEST
2. config.example.yaml - MIDDLE
3. Hardcoded Defaults - LOWEST (fallback only)

Author: SefaGH
Date: 2025-10-20
"""

import os
import re
import yaml
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LiveTradingConfiguration:
    """
    UNIFIED Production Configuration Loader.
    Supports environment variables for deployment flexibility.
    """
    
    CONFIG_FILE_PATH = 'config/config.example.yaml'
    
    # ============= HELPER METHODS =============
    
    @classmethod
    def get_env_float(cls, key: str, default: float) -> float:
        """Get float from environment variable with fallback."""
        try:
            value = os.getenv(key)
            if value is not None:
                result = float(value)
                logger.debug(f"✓ ENV: {key} = {result}")
                return result
            return default
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ Invalid float for {key}: {e}, using default: {default}")
            return default
    
    @classmethod
    def get_env_int(cls, key: str, default: int) -> int:
        """Get int from environment variable with fallback."""
        try:
            value = os.getenv(key)
            if value is not None:
                result = int(value)
                logger.debug(f"✓ ENV: {key} = {result}")
                return result
            return default
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ Invalid int for {key}: {e}, using default: {default}")
            return default
    
    @classmethod
    def get_env_bool(cls, key: str, default: bool) -> bool:
        """Get bool from environment variable with fallback."""
        value = os.getenv(key)
        if value is not None:
            result = value.lower() in ('true', '1', 'yes', 'on')
            logger.debug(f"✓ ENV: {key} = {result}")
            return result
        return default
    
    @classmethod
    def get_env_list(cls, key: str, default: List[str]) -> List[str]:
        """Get list from environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            result = [s.strip() for s in value.split(',') if s.strip()]
            
            # Validate symbols if this is TRADING_SYMBOLS
            if key == 'TRADING_SYMBOLS' and result:
                valid_symbols = cls._validate_trading_symbols(result)
                
                if not valid_symbols:
                    logger.warning(f"⚠️ All symbols in {key} are invalid, using defaults: {default}")
                    return default
                elif len(valid_symbols) < len(result):
                    invalid = set(result) - set(valid_symbols)
                    logger.warning(f"⚠️ Invalid symbols filtered from {key}: {invalid}")
                    result = valid_symbols
            
            logger.debug(f"✓ ENV: {key} = {result}")
            return result
        return default
    
    @classmethod
    def _validate_trading_symbols(cls, symbols: List[str]) -> List[str]:
        """
        Validate trading symbol format.
        
        Valid formats:
        - BTC/USDT:USDT (perpetual futures)
        - BTC/USDT (spot)
        - ETH-PERP (some exchanges)
        
        Args:
            symbols: List of symbol strings to validate
            
        Returns:
            List of valid symbols only
        """
        valid_symbols = []
        
        # Common trading pair patterns
        patterns = [
            r'^[A-Z0-9]{2,10}/[A-Z]{3,5}:[A-Z]{3,5}$',  # BTC/USDT:USDT
            r'^[A-Z0-9]{2,10}/[A-Z]{3,5}$',              # BTC/USDT
            r'^[A-Z0-9]{2,10}-PERP$',                    # BTC-PERP
        ]
        
        for symbol in symbols:
            # Check if matches any valid pattern (case-insensitive)
            if any(re.match(pattern, symbol.upper()) for pattern in patterns):
                valid_symbols.append(symbol)
            else:
                logger.warning(f"⚠️ Invalid symbol format: {symbol}")
        
        return valid_symbols
    
    # ============= CONFIG LOADERS =============
    
    @classmethod
    def load_from_env(cls) -> Dict[str, Any]:
        """
        Load configuration from environment variables (GitHub Secrets/Variables).
        Returns partial config that will override YAML and defaults.
        """
        logger.info("🔧 Loading configuration from environment variables...")
        
        env_config: Dict[str, Any] = {}
        
        # ============= SIGNALS CONFIG =============
        env_config['signals'] = {
            'duplicate_prevention': {
                'min_price_change_pct': cls.get_env_float(
                    'DUPLICATE_PREVENTION_THRESHOLD', 
                    0.05
                ),
                'cooldown_seconds': cls.get_env_int(
                    'DUPLICATE_PREVENTION_COOLDOWN', 
                    20
                ),
                'price_delta_bypass_threshold': cls.get_env_float(
                    'PRICE_DELTA_BYPASS_THRESHOLD',
                    0.0015
                ),
                'price_delta_bypass_enabled': cls.get_env_bool(
                    'PRICE_DELTA_BYPASS_ENABLED',
                    True
                ),
            },
            'oversold_bounce': {
                'enable': cls.get_env_bool('STRATEGY_OB_ENABLED', True),
                'ignore_regime': cls.get_env_bool('STRATEGY_OB_IGNORE_REGIME', True),
                'rsi_max': cls.get_env_int('RSI_MAX_OB', 45),
                'adaptive_rsi_base': cls.get_env_int('RSI_BASE_OB', 45),
                'adaptive_rsi_range': cls.get_env_int('RSI_RANGE_OB', 10),
                'adaptive_mode': os.getenv('ADAPTIVE_MODE_OB', 'dynamic'),
                'volatility_sensitivity': os.getenv('VOLATILITY_SENSITIVITY_OB', 'medium'),
                'tp_atr_mult': cls.get_env_float('TP_ATR_MULT_OB', 2.5),
                'sl_atr_mult': cls.get_env_float('SL_ATR_MULT_OB', 1.2),
                'min_tp_pct': cls.get_env_float('MIN_TP_PCT_OB', 0.008),
                'max_sl_pct': cls.get_env_float('MAX_SL_PCT_OB', 0.015),
            },
            'short_the_rip': {
                'enable': cls.get_env_bool('STRATEGY_STR_ENABLED', True),
                'ignore_regime': cls.get_env_bool('STRATEGY_STR_IGNORE_REGIME', True),
                'rsi_min': cls.get_env_int('RSI_MIN_STR', 55),
                'adaptive_rsi_base': cls.get_env_int('RSI_BASE_STR', 55),
                'adaptive_rsi_range': cls.get_env_int('RSI_RANGE_STR', 10),
                'adaptive_mode': os.getenv('ADAPTIVE_MODE_STR', 'dynamic'),
                'volatility_sensitivity': os.getenv('VOLATILITY_SENSITIVITY_STR', 'medium'),
                'tp_atr_mult': cls.get_env_float('TP_ATR_MULT_STR', 3.0),
                'sl_atr_mult': cls.get_env_float('SL_ATR_MULT_STR', 1.5),
                'min_tp_pct': cls.get_env_float('MIN_TP_PCT_STR', 0.010),
                'max_sl_pct': cls.get_env_float('MAX_SL_PCT_STR', 0.020),
                'symbols': {
                    'BTC/USDT:USDT': {
                        'rsi_threshold': cls.get_env_int('RSI_THRESHOLD_BTC', 55)
                    },
                    'ETH/USDT:USDT': {
                        'rsi_threshold': cls.get_env_int('RSI_THRESHOLD_ETH', 50)
                    },
                    'SOL/USDT:USDT': {
                        'rsi_threshold': cls.get_env_int('RSI_THRESHOLD_SOL', 50)
                    }
                }
            }
        }
        
        # ============= UNIVERSE CONFIG =============
        env_config['universe'] = {
            'fixed_symbols': cls.get_env_list(
                'TRADING_SYMBOLS',
                ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
            ),
            'auto_select': cls.get_env_bool('UNIVERSE_AUTO_SELECT', False),
            'priority_order': cls.get_env_list(
                'TRADING_SYMBOLS_PRIORITY',
                ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
            ),
        }
        
        # ============= RISK CONFIG =============
        env_config['risk'] = {
            'equity_usd': cls.get_env_float('CAPITAL_USDT', 100),
            'per_trade_risk_pct': cls.get_env_float('PER_TRADE_RISK_PCT', 0.01),
            'daily_loss_limit_pct': cls.get_env_float('DAILY_LOSS_LIMIT_PCT', 0.02),
            'risk_usd_cap': cls.get_env_float('RISK_USD_CAP', 5),
            'max_notional_per_trade': cls.get_env_float('MAX_NOTIONAL_PER_TRADE', 20),
            'min_stop_pct': cls.get_env_float('MIN_STOP_PCT', 0.003),
            'daily_max_trades': cls.get_env_int('DAILY_MAX_TRADES', 5),
        }
        
        # ============= EXECUTION CONFIG =============
        env_config['execution'] = {
            'enable_live': cls.get_env_bool('ENABLE_LIVE_TRADING', True),
            'order_type': os.getenv('ORDER_TYPE', 'market'),
            'time_in_force': os.getenv('TIME_IN_FORCE', 'IOC'),
            'fee_pct': cls.get_env_float('FEE_PCT', 0.0006),
            'max_slippage_pct': cls.get_env_float('MAX_SLIPPAGE_PCT', 0.001),
            'leverage': {
                'default': cls.get_env_int('LEVERAGE_DEFAULT', 5),
            }
        }
        
        # ============= WEBSOCKET CONFIG =============
        env_config['websocket'] = {
            'enabled': cls.get_env_bool('WEBSOCKET_ENABLED', True),
            'priority_enabled': cls.get_env_bool('WEBSOCKET_PRIORITY_ENABLED', True),
            'max_data_age': cls.get_env_int('WEBSOCKET_MAX_DATA_AGE', 60),
            'fallback_threshold': cls.get_env_int('WEBSOCKET_FALLBACK_THRESHOLD', 3),
            'max_streams_per_exchange': {
                'bingx': cls.get_env_int('WS_MAX_STREAMS_BINGX', 6),
                'binance': cls.get_env_int('WS_MAX_STREAMS_BINANCE', 20),
                'kucoinfutures': cls.get_env_int('WS_MAX_STREAMS_KUCOIN', 15),
                'default': cls.get_env_int('WS_MAX_STREAMS_DEFAULT', 10),
            },
            'stream_timeframes': cls.get_env_list('WS_STREAM_TIMEFRAMES', ['1m']),
            'reconnect_delay': cls.get_env_int('WS_RECONNECT_DELAY', 5),
            'max_reconnect_attempts': cls.get_env_int('WS_MAX_RECONNECT_ATTEMPTS', 3),
        }
        
        # ============= INDICATORS CONFIG =============
        env_config['indicators'] = {
            'rsi_period': cls.get_env_int('INDICATOR_RSI_PERIOD', 14),
            'atr_period': cls.get_env_int('INDICATOR_ATR_PERIOD', 14),
            'ema_fast': cls.get_env_int('INDICATOR_EMA_FAST', 21),
            'ema_mid': cls.get_env_int('INDICATOR_EMA_MID', 50),
            'ema_slow': cls.get_env_int('INDICATOR_EMA_SLOW', 200),
        }
        
        logger.info("✅ Environment variables loaded")
        return env_config
    
    @classmethod
    def load_from_yaml(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = cls.CONFIG_FILE_PATH
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ {config_path} not found, using defaults")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            logger.info(f"✅ YAML config loaded from {config_path}")
            return yaml_config or {}
        except Exception as e:
            logger.error(f"❌ Error loading YAML: {e}")
            return {}
    
    @classmethod
    def deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        Override values take precedence over base values.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    # ============= MAIN LOADER =============
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, log_summary: bool = True) -> Dict[str, Any]:
        """
        🎯 UNIFIED CONFIG LOADER - Use this everywhere!
        
        Priority order (highest to lowest):
        1. Environment variables (GitHub Secrets/Variables)
        2. config.example.yaml
        3. Hardcoded defaults (not used if above exist)
        
        Args:
            config_path: Optional custom YAML path
            log_summary: Whether to log configuration summary
            
        Returns:
            Complete merged configuration dictionary
        """
        logger.info("="*70)
        logger.info("🔧 LOADING CONFIGURATION")
        logger.info("="*70)
        
        # Start with YAML config
        config = cls.load_from_yaml(config_path)
        
        # Override with environment variables (highest priority)
        env_config = cls.load_from_env()
        config = cls.deep_merge(config, env_config)
        
        # Log summary if requested
        if log_summary:
            cls._log_config_summary(config)
        
        return config
    
    @classmethod
    def _log_config_summary(cls, config: Dict[str, Any]) -> None:
        """Log configuration summary."""
        logger.info("="*70)
        logger.info("📊 CONFIGURATION SUMMARY")
        logger.info("="*70)
        
        # Duplicate Prevention
        if 'signals' in config and 'duplicate_prevention' in config['signals']:
            dp = config['signals']['duplicate_prevention']
            logger.info("🚫 Duplicate Prevention:")
            logger.info(f"   Threshold: {dp.get('min_price_change_pct', 'N/A'):.2%}")
            logger.info(f"   Cooldown: {dp.get('cooldown_seconds', 'N/A')}s")
            logger.info(f"   Bypass: {dp.get('price_delta_bypass_enabled', False)}")
        
        # Trading Symbols
        if 'universe' in config:
            symbols = config['universe'].get('fixed_symbols', [])
            logger.info(f"🎯 Trading Symbols: {len(symbols)}")
            for symbol in symbols:
                logger.info(f"   - {symbol}")
        
        # Risk Parameters
        if 'risk' in config:
            risk = config['risk']
            logger.info("💰 Risk Management:")
            logger.info(f"   Capital: ${risk.get('equity_usd', 'N/A'):.2f}")
            logger.info(f"   Max Position: {risk.get('max_notional_per_trade', 'N/A'):.2f} USDT")
            logger.info(f"   Daily Max Trades: {risk.get('daily_max_trades', 'N/A')}")
        
        # Strategies
        if 'signals' in config:
            logger.info("📈 Strategies:")
            if 'oversold_bounce' in config['signals']:
                ob = config['signals']['oversold_bounce']
                logger.info(f"   OB: {'✅ Enabled' if ob.get('enable') else '❌ Disabled'}")
            if 'short_the_rip' in config['signals']:
                str_cfg = config['signals']['short_the_rip']
                logger.info(f"   STR: {'✅ Enabled' if str_cfg.get('enable') else '❌ Disabled'}")
                if 'symbols' in str_cfg:
                    logger.info("   Symbol-specific RSI:")
                    for symbol, params in str_cfg['symbols'].items():
                        logger.info(f"      {symbol}: {params.get('rsi_threshold', 'N/A')}")
        
        # WebSocket
        if 'websocket' in config:
            ws = config['websocket']
            logger.info(f"🌐 WebSocket: {'✅ Enabled' if ws.get('enabled') else '❌ Disabled'}")
            if ws.get('enabled'):
                logger.info(f"   Max streams (BingX): {ws['max_streams_per_exchange'].get('bingx', 'N/A')}")
        
        logger.info("="*70)
    
    @classmethod
    def get_all_configs(cls) -> Dict[str, Any]:
        """
        Get all configuration (alias for load() method).
        Provided for backwards compatibility with existing tests.
        
        Returns:
            Complete merged configuration dictionary
        """
        return cls.load(log_summary=False)
    
    @classmethod
    def get_config_value(cls, *keys: str, default: Any = None) -> Any:
        """
        Safely get nested config value.
        
        Example:
            threshold = LiveTradingConfiguration.get_config_value(
                'signals', 'duplicate_prevention', 'min_price_change_pct',
                default=0.05
            )
        
        Args:
            *keys: Nested dictionary keys
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        config = cls.load(log_summary=False)
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
