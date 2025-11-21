"""
Dynamic and Centralized Live Trading Configuration System.

This module provides a robust, centralized configuration loader for the Bearish Alpha Bot.
It achieves a "Single Source of Truth" by using `config.example.yaml` as the
canonical definition for all settings, including their default values, types, and
environment variable mappings.

PRIORITY ORDER (Highest to Lowest):
1. Environment Variables (from GitHub Variables/Secrets)
2. `config.example.yaml`

The loader is implemented as a Singleton, ensuring that configuration is parsed
and loaded only ONCE per application lifecycle, providing performance and consistency.

Author: SefaGH & GitHub Copilot
Date: 2025-11-03
Ref: Issue #277
"""


import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


logger = logging.getLogger(__name__)

_config_instance: Optional[Dict[str, Any]] = None
_config_signature: Optional[Tuple[str, Tuple[Tuple[str, Optional[str]]]]] = None
_config_env_keys: Tuple[str, ...] = ()
_config_path_cache: Optional[str] = None

_TRADING_SYMBOL_PATTERN = re.compile(
    r'^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}(:[A-Z0-9]{2,10})?$',
    re.IGNORECASE
)
_DERIVATIVE_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{2,10}-[A-Z0-9]{2,10}$', re.IGNORECASE)
DEFAULT_SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']


class LiveTradingConfiguration:
    """Centralized configuration loader with ENV > YAML priority."""

    CONFIG_FILE_PATH = 'config/config.example.yaml'
    ENV_OVERRIDE_PATTERN = re.compile(r'#\s*Override with:\s*(\w+)')
    _env_keys_snapshot: Tuple[str, ...] = ()

    @classmethod
    def load(
        cls,
        log_summary: bool = True,
        *,
        config_path: Optional[str] = None,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point. Loads, merges, and returns the configuration.
        Uses a signature-aware cache to ensure ENV > YAML > defaults priority.
        """
        global _config_instance, _config_signature, _config_env_keys, _config_path_cache

        resolved_path = cls._resolve_config_path(config_path)

        if force_reload:
            cls.reset_cache()

        if (
            not force_reload
            and _config_instance is not None
            and _config_signature is not None
            and _config_path_cache == resolved_path
        ):
            current_signature = cls._build_signature_from_keys(resolved_path, _config_env_keys)
            if current_signature == _config_signature:
                logger.debug("Returning cached configuration instance (env unchanged).")
                if log_summary:
                    # Maintain backward compatibility by logging summary once per caller request.
                    cls(resolved_path)._log_config_summary(_config_instance)
                return _config_instance

        logger.info("=" * 70)
        logger.info("🔧 DYNAMIC CONFIGURATION LOADER (v2.1 - Signature cache)")
        logger.info("=" * 70)

        instance = cls(resolved_path)
        try:
            config = instance._load_and_merge_configs()
            _config_instance = config
            _config_env_keys = instance._env_keys_snapshot
            _config_path_cache = resolved_path
            _config_signature = cls._build_signature_from_keys(resolved_path, _config_env_keys)

            if log_summary:
                instance._log_config_summary(config)

            return config
        except Exception as e:
            cls.reset_cache()
            logger.critical(
                f"❌ A critical error occurred during configuration loading: {e}",
                exc_info=True
            )
            raise RuntimeError("Failed to load configuration. Bot cannot start.") from e

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = self._resolve_config_path(config_path)
        self._env_keys_snapshot = ()

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> str:
        candidate = config_path or os.getenv('CONFIG_PATH') or LiveTradingConfiguration.CONFIG_FILE_PATH
        return candidate

    @classmethod
    def reset_cache(cls) -> None:
        global _config_instance, _config_signature, _config_env_keys, _config_path_cache
        _config_instance = None
        _config_signature = None
        _config_env_keys = ()
        _config_path_cache = None

    @staticmethod
    def _build_signature_from_keys(
        config_path: str,
        env_keys: Tuple[str, ...]
    ) -> Tuple[str, Tuple[Tuple[str, Optional[str]]]]:
        snapshot = tuple(sorted((key, os.getenv(key)) for key in env_keys))
        return (config_path, snapshot)

    def _load_and_merge_configs(self) -> Dict[str, Any]:
        """Orchestrates the loading and merging process."""
        # 1. Load the base YAML config and parse env var mappings from its comments
        yaml_config, env_map = self._load_yaml_and_map_env_vars()
        yaml_config = yaml_config or {}
        self._env_keys_snapshot = tuple(sorted(env_map.keys()))

        # 2. Normalize YAML values (e.g., convert trading symbol strings to lists)
        yaml_config = self._normalize_yaml_values(yaml_config)

        # 3. Get overrides from environment variables using the parsed map
        env_overrides = self._get_env_overrides(env_map, yaml_config)

        # 4. Deep merge YAML config with environment overrides
        merged = self._deep_merge(yaml_config, env_overrides)
        self._apply_universe_defaults(merged)
        self._normalize_risk_config(merged)
        return merged

    def _load_yaml_and_map_env_vars(self) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Loads the YAML file line by line to extract both config and env mappings.
        This is a robust way to link comments to their corresponding keys.
        
        Returns:
            A tuple: (loaded_yaml_dict, env_var_to_path_map)
        """
        env_map: Dict[str, List[str]] = {}
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
    
        with open(self.config_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        path_stack: List[Tuple[int, str]] = []
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue
    
            indentation = len(line) - len(line.lstrip(' '))
            
            try:
                key_part = stripped_line.split(':', 1)[0].strip()
                
                # ✅ FIX: Remove quotes from keys (for symbol names like "BTC/USDT:USDT")
                # This handles both single and double quotes
                if (key_part.startswith('"') and key_part.endswith('"')) or \
                   (key_part.startswith("'") and key_part.endswith("'")):
                    key_part = key_part[1:-1]  # Remove surrounding quotes
                    
            except IndexError:
                continue
    
            while path_stack and path_stack[-1][0] >= indentation:
                path_stack.pop()
    
            path_stack.append((indentation, key_part))
            
            match = self.ENV_OVERRIDE_PATTERN.search(line)
            if match:
                env_var = match.group(1)
                current_path = [p[1] for p in path_stack]
                env_map[env_var] = current_path
                
                # Debug log for problematic variables
                if 'RSI_THRESHOLD' in env_var or 'ML_' in env_var:
                    logger.debug(f"Mapped ENV '{env_var}' to config path: {'.'.join(current_path)}")
    
        with open(self.config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    
        logger.info(f"✅ YAML config loaded. Found {len(env_map)} environment variable mappings.")
        return yaml_config or {}, env_map

    def _normalize_yaml_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize YAML values by applying the same conversion logic as _cast_value.
        This ensures that trading symbols in YAML are converted to lists.
        """
        def normalize_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: normalize_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Use the same helper method to detect and parse trading symbols
                if self._is_trading_symbol(obj):
                    return self._parse_trading_symbols(obj)
                return obj
            else:
                return obj
        
        return normalize_recursive(config)

    def _get_env_overrides(self, env_map: Dict[str, List[str]], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a dictionary of overrides from environment variables.
        Performs automatic type casting based on the default value's type.
        """
        overrides: Dict[str, Any] = {}
        logger.info("🔧 Applying overrides from environment variables...")
        
        # RSI threshold symbol mapping (spot -> futures format)
        rsi_symbol_map = {
            'RSI_THRESHOLD_BTC': ['BTC/USDT', 'BTC/USDT:USDT'],
            'RSI_THRESHOLD_ETH': ['ETH/USDT', 'ETH/USDT:USDT'],
            'RSI_THRESHOLD_SOL': ['SOL/USDT', 'SOL/USDT:USDT'],
            'RSI_THRESHOLD_BNB': ['BNB/USDT', 'BNB/USDT:USDT'],
        }
        
        for env_var, path in env_map.items():
            env_value_str = os.getenv(env_var)
            if env_value_str is None or env_value_str == '':
                continue
    
            # Special handling for RSI_THRESHOLD variables
            if env_var in rsi_symbol_map:
                # Try to find the correct path with available symbol formats
                original_value = None
                successful_path = None
                
                for symbol_format in rsi_symbol_map[env_var]:
                    try:
                        # Create adapted path with correct symbol format
                        adapted_path = path.copy()
                        if len(adapted_path) >= 3 and 'symbols' in adapted_path:
                            # Replace the symbol key in the path
                            symbol_index = adapted_path.index('symbols') + 1
                            if symbol_index < len(adapted_path):
                                adapted_path[symbol_index] = symbol_format
                        
                        # Try to navigate with adapted path
                        test_value = base_config
                        for key in adapted_path:
                            test_value = test_value[key]
                        
                        # If successful, use this path
                        original_value = test_value
                        successful_path = adapted_path
                        logger.debug(f"✓ Found RSI config for {env_var} using format: {symbol_format}")
                        break
                        
                    except (KeyError, TypeError):
                        continue
                
                if successful_path:
                    path = successful_path
                else:
                    logger.warning(f"⚠️ Could not find valid path for {env_var}")
                    continue
    
            # Standard processing for all variables (including adapted RSI ones)
            try:
                # Navigate through config to find original value
                if 'RSI_THRESHOLD' not in env_var:  # Skip navigation for already processed RSI vars
                    original_value = base_config
                    for key in path:
                        if not isinstance(original_value, dict):
                            raise KeyError(f"Expected dict at {key}")
                        original_value = original_value[key]
                
                target_type = type(original_value) if original_value is not None else str
                converted_value = self._cast_value(env_value_str, target_type)
                
                # Build nested dictionary for override
                temp_dict = overrides
                for key in path[:-1]:
                    temp_dict = temp_dict.setdefault(key, {})
                temp_dict[path[-1]] = converted_value
                
                logger.info(f"  ✓ Applied ENV: {env_var} = {converted_value} (as {target_type.__name__})")
                
            except KeyError as e:
                logger.warning(f"  ⚠️ ENV var '{env_var}' found, but path '{'.'.join(path)}' is invalid. Error: {e}")
            except Exception as e:
                logger.error(f"  ❌ Failed to process '{env_var}': {e}")
        
        return overrides
    
    def _navigate_config_path(self, config: Dict, path: List[str]) -> Any:
        """
        Helper to navigate through nested config dictionary.
        
        Args:
            config: The configuration dictionary to navigate
            path: List of keys representing the path to navigate
            
        Returns:
            The value at the end of the path
            
        Raises:
            KeyError: If the path is invalid
        """
        result = config
        for key in path:
            if not isinstance(result, dict):
                raise KeyError(f"Expected dict, got {type(result).__name__}")
            if key not in result:
                raise KeyError(key)
            result = result[key]
        return result

    @staticmethod
    def _is_trading_symbol(value: str) -> bool:
        """
        Detect if a string represents a trading symbol or list of trading symbols.
        
        Trading symbols have the format: "BASE/QUOTE" or "BASE/QUOTE:SETTLE"
        Examples: "BTC/USDT", "ETH/USDT:USDT", "BTC/USDT,ETH/USDT"
        Returns:
            True if the string contains trading symbol(s), False otherwise.
        """
        if not isinstance(value, str) or not value.strip():
            return False

        parts = [p.strip().upper() for p in value.split(',') if p.strip()]
        if not parts:
            return False

        for part in parts:
            if _TRADING_SYMBOL_PATTERN.match(part) or _DERIVATIVE_SYMBOL_PATTERN.match(part):
                return True
        return False
    
    @staticmethod
    def _parse_trading_symbols(value_str: str) -> list:
        """
        Parse trading symbol(s) from a string into a list.
        
        Handles both single symbols and comma-separated lists.
        Examples:
            "BTC/USDT" -> ["BTC/USDT"]
            "BTC/USDT,ETH/USDT" -> ["BTC/USDT", "ETH/USDT"]
        """
        if ',' in value_str:
            # Multiple symbols
            return [s.strip() for s in value_str.split(',') if s.strip()]
        else:
            # Single symbol
            return [value_str.strip()]

    @staticmethod
    def _cast_value(value_str: str, target_type: type) -> Any:
        """Helper to convert a string value to a specific target type."""
        try:
            # Trading symbols check
            if LiveTradingConfiguration._is_trading_symbol(value_str):
                return LiveTradingConfiguration._parse_trading_symbols(value_str)
            
            # ML window lists için özel işlem
            if target_type is list:
                cleaned = value_str.strip()
                if cleaned.startswith('[') and cleaned.endswith(']'):
                    cleaned = cleaned[1:-1]
                
                parts = [s.strip() for s in cleaned.split(',') if s.strip()]
                
                # Tüm parçalar sayı mı kontrol et
                all_numeric = all(p.replace('.','').replace('-','').isdigit() for p in parts)
                
                if all_numeric:
                    # Integer list olarak döndür
                    return [int(float(p)) for p in parts]
                else:
                    # String list olarak döndür
                    return parts
            
            # Original type conversions
            if target_type is bool:
                return value_str.lower() in ('true', '1', 't', 'y', 'yes')
            if target_type is int:
                return int(value_str)
            if target_type is float:
                return float(value_str)
            if target_type is str:
                if ',' in value_str:
                    return [s.strip() for s in value_str.split(',') if s.strip()]
                return value_str
            return value_str
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not cast '{value_str}' to {target_type.__name__}: {e}")
            return value_str

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deeply merges the override dict into the base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = LiveTradingConfiguration._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def deep_merge(cls, base: Dict, override: Dict) -> Dict:
        """Public helper exposed for tests (wraps _deep_merge)."""
        return cls._deep_merge(base, override)

    def _apply_universe_defaults(self, config: Dict[str, Any]) -> None:
        universe = config.setdefault('universe', {})
        symbols = universe.get('fixed_symbols')
        cleaned = self._filter_valid_symbols(symbols)
        if not cleaned:
            cleaned = DEFAULT_SYMBOLS.copy()
        universe['fixed_symbols'] = cleaned

    def _filter_valid_symbols(self, symbols: Union[str, List[str], None]) -> List[str]:
        if symbols is None:
            return []
        if isinstance(symbols, str):
            candidates = [s.strip() for s in symbols.split(',') if s.strip()]
        elif isinstance(symbols, list):
            candidates = []
            for item in symbols:
                if isinstance(item, str):
                    if ',' in item:
                        candidates.extend([s.strip() for s in item.split(',') if s.strip()])
                    else:
                        candidates.append(item.strip())
        else:
            return []

        normalized: List[str] = []
        for candidate in candidates:
            canonical = self._normalize_symbol(candidate)
            if canonical and (_TRADING_SYMBOL_PATTERN.match(canonical) or _DERIVATIVE_SYMBOL_PATTERN.match(canonical)):
                if canonical not in normalized:
                    normalized.append(canonical)
        return normalized

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        cleaned = symbol.strip()
        return cleaned.upper() if cleaned else ''

    def _normalize_risk_config(self, config: Dict[str, Any]) -> None:
        """Ensure risk percentages stay in fractional form and derive USD helpers."""
        risk_section = config.get('risk')
        if not isinstance(risk_section, dict):
            risk_section = {}
            config['risk'] = risk_section

        percent_keys = [
            'daily_loss_limit_pct',
            'max_position_size_pct',
            'max_notional_pct_per_trade',
            'max_margin_pct_per_trade'
        ]

        # --- Critical: normalize per-trade risk with env + defaults ---
        per_trade_raw = risk_section.get('per_trade_risk_pct')
        if per_trade_raw is None:
            env_fallback = os.getenv('PER_TRADE_RISK_PCT')
            if env_fallback is not None:
                try:
                    per_trade_raw = float(env_fallback)
                except (TypeError, ValueError):
                    logger.warning(
                        f"⚠️ PER_TRADE_RISK_PCT env value '{env_fallback}' is invalid; ignoring fallback."
                    )
        if per_trade_raw is None:
            logger.warning("PER_TRADE_RISK_PCT not set, defaulting to 1% (0.01)")
            per_trade_raw = 0.01

        normalized_per_trade = self._normalize_percent_value(per_trade_raw, 'risk.per_trade_risk_pct')
        if not normalized_per_trade or normalized_per_trade <= 0 or normalized_per_trade > 1:
            logger.error(
                f"❌ per_trade_risk_pct out of bounds after normalization: {normalized_per_trade}. Resetting to 0.01 (1%)."
            )
            normalized_per_trade = 0.01
        risk_section['per_trade_risk_pct'] = normalized_per_trade

        for key in percent_keys:
            if key not in risk_section or risk_section[key] is None:
                continue
            normalized = self._normalize_percent_value(risk_section[key], f"risk.{key}")
            if normalized is not None:
                risk_section[key] = normalized

        try:
            equity = float(risk_section.get('equity_usd', 0) or 0)
        except (TypeError, ValueError):
            equity = 0

        computed_risk_usd = equity * normalized_per_trade if equity > 0 else 0.0
        risk_section['computed_max_risk_usd'] = computed_risk_usd
        logger.info(
            "✅ Risk normalization: per_trade_risk_pct=%.4f (fraction), computed_max_risk_usd=%.2f USD",
            normalized_per_trade,
            computed_risk_usd,
        )

        max_notional_pct = risk_section.get('max_notional_pct_per_trade')
        if isinstance(max_notional_pct, (int, float)) and equity > 0:
            derived_notional = equity * max_notional_pct
            risk_section['computed_max_notional_usd'] = derived_notional
            # Preserve backwards compatibility if legacy field missing or zero
            legacy_value = risk_section.get('max_notional_per_trade')
            if not legacy_value:
                risk_section['max_notional_per_trade'] = derived_notional

        config['risk'] = risk_section

    @staticmethod
    def _normalize_percent_value(value: Any, field_name: str) -> Optional[float]:
        """Convert percent inputs to safe fractional form (0-1)."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ Unable to parse {field_name}={value} as float; keeping original value.")
            return None

        if numeric <= 0:
            logger.warning(f"⚠️ {field_name}={numeric} is non-positive. Check configuration values.")
            return numeric

        if numeric >= 1.0:
            if numeric <= 100.0:
                logger.warning(
                    f"⚠️ {field_name} appears to be expressed as percent ({numeric}). Converting to fractional form."
                )
                numeric = numeric / 100.0
            else:
                logger.error(
                    f"❌ {field_name}={numeric} exceeds 100%. Clamping to 100% (1.0) to keep values in range."
                )
                numeric = 1.0
        return numeric

    @staticmethod
    def _log_config_summary(config: Dict[str, Any]) -> None:
        """Logs a summary of the final, effective configuration."""
        logger.info("="*70)
        logger.info("📊 FINAL CONFIGURATION SUMMARY (Effective Values)")
        logger.info("="*70)
        
        # Helper to safely get nested values
        def get_nested(data: Dict, path: List[str], default: Any = 'N/A') -> Any:
            for key in path:
                if not isinstance(data, dict): return default
                data = data.get(key)
            return data if data is not None else default

        # ML Settings
        if get_nested(config, ['ml', 'enabled']):
            logger.info("🧠 ML Settings:")
            logger.info(f"   Min Regime Confidence: {get_nested(config, ['ml', 'prediction', 'min_confidence_threshold'])}")
            logger.info(f"   RL Veto Threshold:     {get_nested(config, ['ml', 'reinforcement_learning', 'hold_confidence_threshold'])}")
            logger.info(f"   Legacy RL Enabled:     {get_nested(config, ['ml', 'reinforcement_learning', 'legacy_dqn_enabled'], False)}")
            ppo_enabled = bool(get_nested(config, ['ml', 'reinforcement_learning', 'ppo_enabled'], False))
            ppo_symbols = get_nested(config, ['ml', 'reinforcement_learning', 'ppo_symbols'], [])
            if isinstance(ppo_symbols, list):
                ppo_symbol_str = ', '.join(ppo_symbols)
            else:
                ppo_symbol_str = str(ppo_symbols)
            logger.info(f"   PPO Enabled:           {ppo_enabled} | Symbols: {ppo_symbol_str}")

        # Trading Universe
        symbols_val = get_nested(config, ['universe', 'fixed_symbols'], [])
        symbols = symbols_val if isinstance(symbols_val, list) else [s.strip() for s in str(symbols_val).split(',')]
        logger.info(f"🎯 Trading Universe: {len(symbols)} symbols")
        if symbols: logger.info(f"   - {', '.join(symbols)}")
        
        # Risk Management
        logger.info("💰 Risk Management:")
        capital_val = float(get_nested(config, ['risk', 'equity_usd'], 0) or 0)
        logger.info(f"   Capital: ${capital_val:.2f} USDT")

        risk_section = get_nested(config, ['risk'], {})
        if not isinstance(risk_section, dict):
            risk_section = {}
        logger.info(LiveTradingConfiguration._format_risk_summary(risk_section, capital_val))

        max_notional_usd = get_nested(config, ['risk', 'max_notional_per_trade'], 0.0)
        if (not isinstance(max_notional_usd, (int, float)) or max_notional_usd == 0.0) and isinstance(
            get_nested(config, ['risk', 'computed_max_notional_usd'], None), (int, float)
        ):
            max_notional_usd = get_nested(config, ['risk', 'computed_max_notional_usd'], 0.0)
        logger.info(f"   Max Notional Per Trade: {max_notional_usd:.2f} USDT")
        
        logger.info("="*70)

    @staticmethod
    def _format_risk_summary(risk_cfg: Dict[str, Any], capital_val: float) -> str:
        """Format risk summary line ensuring normalized percentages."""
        if not isinstance(risk_cfg, dict):
            risk_cfg = {}

        pct_value = risk_cfg.get('per_trade_risk_pct')
        usd_value = risk_cfg.get('computed_max_risk_usd')
        normalized_fraction: Optional[float] = None

        if isinstance(usd_value, (int, float)) and usd_value > 0 and capital_val > 0:
            normalized_fraction = max(usd_value / capital_val, 0.0)

        if normalized_fraction is None and isinstance(pct_value, (int, float)):
            normalized_fraction = pct_value if pct_value <= 1 else pct_value / 100.0
            if not isinstance(usd_value, (int, float)) or usd_value <= 0:
                usd_value = capital_val * normalized_fraction

        if normalized_fraction is not None and normalized_fraction > 0:
            display_pct = normalized_fraction * 100.0
            usd_value = usd_value if isinstance(usd_value, (int, float)) else capital_val * normalized_fraction
            return f"   Risk Per Trade: {display_pct:.2f}% ({usd_value:.2f} USDT max risk)"

        raw_env = os.getenv('PER_TRADE_RISK_PCT')
        if raw_env is not None:
            try:
                raw_numeric = float(raw_env)
                normalized = raw_numeric if raw_numeric <= 1 else raw_numeric / 100.0
                if normalized > 0:
                    usd_value = capital_val * normalized
                    return f"   Risk Per Trade: {normalized * 100:.2f}% ({usd_value:.2f} USDT max risk)"
                return f"   Risk Per Trade: {raw_numeric:.2f}% (raw env)"
            except (TypeError, ValueError):
                return f"   Risk Per Trade: {raw_env} (raw env)"

        return "   Risk Per Trade: N/A"

# Global accessor function for easy, consistent access from anywhere in the codebase.
def get_config() -> Dict[str, Any]:
    """
    Global accessor for the singleton configuration instance.
    This should be the ONLY way other modules get the configuration.
    
    Note: This function always logs the configuration summary on first load.
    Tests that need to suppress logging should use LiveTradingConfiguration.load(log_summary=False) directly.
    """
    return LiveTradingConfiguration.load()
