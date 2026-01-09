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


import ast
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from .mtf_policy import build_str_mtf_config

# Azure App Configuration imports (optional, graceful fallback if not available)
try:
    from azure.appconfiguration.provider import load as load_appconfig
    from azure.identity import DefaultAzureCredential
    AZURE_APPCONFIG_AVAILABLE = True
except ImportError:
    AZURE_APPCONFIG_AVAILABLE = False


logger = logging.getLogger(__name__)

_config_instance: Optional[Dict[str, Any]] = None
_config_signature: Optional[Tuple[str, Tuple[Tuple[str, Optional[str]]]]] = None
_config_env_keys: Tuple[str, ...] = ()
_config_path_cache: Optional[str] = None
_last_logged_signature: Optional[Tuple[str, Tuple[Tuple[str, Optional[str]]]]] = None
_MISSING = object()

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
    TYPE_COERCION_ALLOWLIST = (
        'ml.reinforcement_learning.training_mode',
        'signals.short_the_rip.mtf_confirmation.enabled',
        'signals.short_the_rip.mtf_confirmation.require_15m',
        'signals.short_the_rip.mtf_confirmation.require_1h',
        'pyramiding.enabled',
        'strategies.regime_routing.bullish.preferred_strategies',
        'strategies.regime_routing.bearish.preferred_strategies',
        'strategies.regime_routing.neutral.preferred_strategies',
        'strategies.regime_routing.volatile.preferred_strategies',
        'risk.min_stop_pct',
        'risk.per_trade_risk_pct',
        'risk.max_position_size_pct',
        'risk.max_notional_pct_per_trade',
        'risk.max_margin_pct_per_trade',
        'risk.daily_loss_limit_pct',
    )
    TYPE_COERCION_ALLOWLIST_PREFIXES = (
        'risk.',
    )
    TYPE_COERCION_JSON_ONLY = {
        'volume_analyzer.buckets',
    }
    TYPE_COERCION_TYPE_OVERRIDES = {
        'strategies.regime_routing.bullish.preferred_strategies': {'type': list, 'elem_type': str, 'json_only': False},
        'strategies.regime_routing.bearish.preferred_strategies': {'type': list, 'elem_type': str, 'json_only': False},
        'strategies.regime_routing.neutral.preferred_strategies': {'type': list, 'elem_type': str, 'json_only': False},
        'strategies.regime_routing.volatile.preferred_strategies': {'type': list, 'elem_type': str, 'json_only': False},
        'ml.features.volatility_windows': {'type': list, 'elem_type': int, 'json_only': False},
        'ml.features.momentum_windows': {'type': list, 'elem_type': int, 'json_only': False},
        'ml.price_prediction.timeframes': {'type': list, 'elem_type': str, 'json_only': False},
        'ml.price_prediction.models': {'type': list, 'elem_type': str, 'json_only': False},
        'ml.reinforcement_learning.ppo_lookback_windows': {'type': list, 'elem_type': int, 'json_only': False},
    }
    TYPE_VALIDATION_ALLOWLIST = TYPE_COERCION_ALLOWLIST
    TYPE_VALIDATION_ALLOWLIST_PREFIXES = TYPE_COERCION_ALLOWLIST_PREFIXES
    TYPE_VALIDATION_STRICT_ENV = 'CONFIG_STRICT_TYPE_CHECK'
    OPERATIONAL_SCHEMA = {
        'bingx_rest_debug': {
            'type': bool,
            'source': 'runtime',
            'note': 'Enable BingX REST debug logging',
        },
        'ccxt_timeout_ms': {
            'type': int,
            'source': 'runtime',
            'note': 'CCXT request timeout in milliseconds',
        },
        'debug_mode': {
            'type': bool,
            'source': 'runtime',
            'note': 'Global debug logging toggle',
        },
        'exchanges': {
            'type': list,
            'elem_type': str,
            'source': 'runtime',
            'note': 'Comma-separated exchange list',
        },
        'log_level': {
            'type': str,
            'source': 'runtime',
            'note': 'Logging level (e.g., INFO, DEBUG)',
        },
        'pythonpath': {
            'type': str,
            'source': 'runtime',
            'note': 'Python module search path',
        },
        'pythonunbuffered': {
            'type': int,
            'source': 'runtime',
            'note': 'Python unbuffered IO flag',
        },
        'strategy_shadow_eval': {
            'type': int,
            'source': 'runtime',
            'note': 'Enable shadow evaluation logging (1=on, 0=off)',
        },
        'telegram_chat_id': {
            'type': int,
            'source': 'runtime',
            'note': 'Telegram chat ID',
        },
        'ticker_cache_ttl_s': {
            'type': float,
            'source': 'runtime',
            'note': 'Ticker cache TTL in seconds',
        },
        'ticker_max_attempts': {
            'type': int,
            'source': 'runtime',
            'note': 'Ticker retry max attempts',
        },
        'ticker_retry_base_delay_s': {
            'type': float,
            'source': 'runtime',
            'note': 'Ticker retry base delay in seconds',
        },
        'trading_duration': {
            'type': int,
            'source': 'runtime',
            'note': 'Trading duration in seconds',
        },
        'trading_mode': {
            'type': str,
            'source': 'runtime',
            'note': 'Trading mode (paper/live)',
        },
    }
    DEPRECATED_LEGACY_KEYS = {
        'ml_rl_training_mode': 'ml.reinforcement_learning.training_mode',
    }

    # ------------------------------------------------------------------
    # Azure App Configuration compatibility (strict-schema safe)
    #
    # AppConfig is highest priority. When stale keys linger in the remote store,
    # strict schema mode fails early (before we can run safety validation).
    # We therefore rewrite known legacy keys into their canonical equivalents
    # and drop removed/dead blocks with an explicit warning.
    # ------------------------------------------------------------------
    APPCONFIG_DROP_PREFIXES = (
        # Removed from YAML; legacy/unused in current live launcher path.
        'adaptive_strategies',
    )
    APPCONFIG_KEY_REWRITES = {
        # Naming mismatch (RiskConfiguration ignores the *_pct legacy key)
        'risk.max_position_size_pct': 'risk.max_position_size',
        # STR MTF legacy missing-data flags -> canonical missing_*_is_fatal
        'signals.short_the_rip.mtf_confirmation.require_15m': 'signals.short_the_rip.mtf_confirmation.missing_15m_is_fatal',
        'signals.short_the_rip.mtf_confirmation.require_1h': 'signals.short_the_rip.mtf_confirmation.missing_1h_is_fatal',
    }
    APPCONFIG_PREFIX_REWRITES = {
        # Strategy rename: AdaptiveShortTheRip emits strategy_name=adaptive_str
        'strategies.adaptive_short_the_rip': 'strategies.adaptive_str',
    }

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
                global _last_logged_signature
                if _last_logged_signature != _config_signature:
                    instance._log_config_summary(config)
                    _last_logged_signature = _config_signature

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
        self._appconfig_raw_keys: Tuple[str, ...] = ()
        self._appconfig_normalized_paths: List[str] = []

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> str:
        candidate = config_path or os.getenv('CONFIG_PATH') or LiveTradingConfiguration.CONFIG_FILE_PATH
        return candidate

    @classmethod
    def reset_cache(cls) -> None:
        global _config_instance, _config_signature, _config_env_keys, _config_path_cache, _last_logged_signature
        _config_instance = None
        _config_signature = None
        _config_env_keys = ()
        _config_path_cache = None
        _last_logged_signature = None

    @staticmethod
    def _build_signature_from_keys(
        config_path: str,
        env_keys: Tuple[str, ...]
    ) -> Tuple[str, Tuple[Tuple[str, Optional[str]]]]:
        snapshot = tuple(sorted((key, os.getenv(key)) for key in env_keys))
        return (config_path, snapshot)

    def _load_and_merge_configs(self) -> Dict[str, Any]:
        """Orchestrates the loading and merging process with Azure App Configuration support.
        
        Priority Order:
        1. Azure App Configuration (cloud-based overrides)
        2. Environment Variables (legacy support)
        3. config.example.yaml (defaults)
        """
        # 1. Load the base YAML config and parse env var mappings from its comments
        yaml_config, env_map = self._load_yaml_and_map_env_vars()
        yaml_config = yaml_config or {}
        self._env_keys_snapshot = tuple(sorted(env_map.keys()))

        # 2. Normalize YAML values (e.g., convert trading symbol strings to lists)
        yaml_config = self._normalize_yaml_values(yaml_config)

        # 3. Try to load from Azure App Configuration (if available)
        appconfig_overrides = self._load_from_app_config() if AZURE_APPCONFIG_AVAILABLE else {}

        # 4. Get overrides from environment variables using the parsed map
        env_overrides = self._get_env_overrides(env_map, yaml_config)

        # 5. Deep merge with priority: App Config > ENV Vars > YAML
        merged = self._deep_merge(yaml_config, env_overrides)
        if appconfig_overrides:
            merged = self._deep_merge(merged, appconfig_overrides)

        canonical_schema = self._build_type_schema(yaml_config)
        operational_schema = self._build_operational_schema()
        combined_schema = self._merge_operational_schema(canonical_schema, operational_schema)
        self._warn_unknown_appconfig_keys(canonical_schema, operational_schema)
        self._apply_schema_type_coercion(merged, combined_schema)
        self._apply_heuristic_type_coercion(merged, combined_schema)
        self._warn_deprecated_keys(merged)
        self._validate_schema_types(merged, combined_schema)
        self._validate_security_layer(merged)
        
        self._apply_universe_defaults(merged)
        self._apply_trigger_price_defaults(merged)
        self._normalize_risk_config(merged)
        self._apply_websocket_defaults(merged)
        self._normalize_str_mtf_config(merged)
        self._normalize_mean_reversion_dynamic_controller_config(merged)
        return merged

    def _validate_security_layer(self, config: Dict[str, Any]) -> None:
        """Fail-fast validation for critical config invariants (Pydantic when available)."""
        try:
            from .schema import PYDANTIC_AVAILABLE, ConfigSafetyError, ValidationError, validate_with_schema
        except Exception as exc:  # noqa: BLE001
            # Never silently fail without visibility.
            logger.warning("?? Config security layer unavailable (schema import failed): %s", exc)
            return

        logger.info("?? [CONFIG-VALIDATION] Validating configuration safety invariants...")
        try:
            validate_with_schema(config)
            logger.info(
                "?? [CONFIG-VALIDATION] OK (%s)",
                "pydantic" if PYDANTIC_AVAILABLE else "fallback",
            )
        except (ConfigSafetyError, ValidationError, ValueError) as e:
            print("\n" + "=" * 60)
            print("?? KRITIK KONFIGURASYON HATASI (BOT BASLATILAMADI)")
            print("=" * 60)

            if isinstance(e, ValidationError) and hasattr(e, "errors"):
                try:
                    for err in e.errors():
                        loc = " -> ".join(str(l) for l in err.get("loc", []))
                        msg = err.get("msg", str(e))
                        print(f"? HATA YERI: {loc}")
                        print(f"  MESAJ: {msg}")
                except Exception:
                    print(str(e))
            else:
                print(str(e))

            print("=" * 60 + "\n")
            # Fail-fast: do not continue with a possibly unsafe config.
            sys.exit(1)

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

    @classmethod
    def _build_type_schema(cls, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        schema: Dict[str, Dict[str, Any]] = {}

        def walk(prefix: str, value: Any) -> None:
            if not isinstance(value, dict):
                return
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                meta: Dict[str, Any] = {'type': type(child), 'default': child}

                if isinstance(child, str):
                    structured = cls._parse_structured_value(child, path, warn=False)
                    if isinstance(structured, list):
                        meta['type'] = list
                        meta['default'] = structured
                        child = structured
                    elif isinstance(structured, dict):
                        meta['type'] = dict
                        meta['default'] = structured
                        child = structured

                if isinstance(child, list):
                    elem_type, json_only = cls._infer_list_meta(child)
                    meta['elem_type'] = elem_type
                    meta['json_only'] = json_only or path in cls.TYPE_COERCION_JSON_ONLY
                if path in cls.TYPE_COERCION_TYPE_OVERRIDES:
                    meta.update(cls.TYPE_COERCION_TYPE_OVERRIDES[path])
                schema[path] = meta
                walk(path, child)

        walk('', config)
        return schema

    @classmethod
    def _build_operational_schema(cls) -> Dict[str, Dict[str, Any]]:
        schema: Dict[str, Dict[str, Any]] = {}
        for key, meta in cls.OPERATIONAL_SCHEMA.items():
            entry = {
                'type': meta.get('type', str),
                'default': None,
                'source': meta.get('source', 'runtime'),
                'note': meta.get('note', ''),
            }
            if meta.get('type') is list:
                entry['elem_type'] = meta.get('elem_type', str)
                entry['json_only'] = meta.get('json_only', False)
            schema[key] = entry
        return schema

    @classmethod
    def _merge_operational_schema(
        cls,
        canonical: Dict[str, Dict[str, Any]],
        operational: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        combined = dict(canonical)
        overlaps = set(canonical.keys()) & set(operational.keys())
        if overlaps:
            message = (
                "Operational schema overlaps canonical keys: "
                + ", ".join(sorted(overlaps))
                + " (canonical wins)"
            )
            if cls._get_strict_mode():
                raise ValueError(message)
            logger.warning(message)

        for key, meta in operational.items():
            if key not in combined:
                combined[key] = meta
        return combined

    @staticmethod
    def _infer_list_meta(values: List[Any]) -> Tuple[Optional[type], bool]:
        if not values:
            return (None, False)

        if all(isinstance(item, (list, dict)) for item in values):
            return (None, True)

        if all(isinstance(item, bool) for item in values):
            return (bool, False)

        if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
            return (int, False)

        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in values):
            return (float, False)

        if all(isinstance(item, str) for item in values):
            return (str, False)

        return (None, False)

    @classmethod
    def _collect_allowlist_paths(
        cls,
        schema: Dict[str, Dict[str, Any]],
        allowlist: Iterable[str],
        prefixes: Iterable[str],
        label: str,
    ) -> List[str]:
        selected: set = set()
        missing: List[str] = []

        for path in allowlist:
            if path in schema:
                selected.add(path)
            else:
                missing.append(path)

        for prefix in prefixes:
            normalized_prefix = prefix.rstrip('.')
            matches = [
                path
                for path in schema.keys()
                if path == normalized_prefix or path.startswith(prefix)
            ]
            if not matches:
                logger.warning("%s allowlist prefix had no schema matches: %s", label, prefix)
            else:
                selected.update(matches)

        if missing:
            logger.warning(
                "%s allowlist paths missing from schema: %s",
                label,
                ", ".join(sorted(missing)),
            )

        return sorted(selected, key=lambda path: (path.count('.'), path))

    @classmethod
    def _apply_type_coercion_allowlist(
        cls,
        config: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
        paths = cls._collect_allowlist_paths(
            schema,
            cls.TYPE_COERCION_ALLOWLIST,
            cls.TYPE_COERCION_ALLOWLIST_PREFIXES,
            "Type coercion",
        )
        for path in paths:
            meta = schema.get(path)
            if not meta:
                continue

            current_value = cls._get_nested_value(config, path.split('.'))
            if current_value is _MISSING:
                continue

            coerced = cls._coerce_value(current_value, meta, path)
            if coerced is not current_value:
                cls._set_nested_value(config, path.split('.'), coerced)

    @classmethod
    def _apply_schema_type_coercion(
        cls,
        config: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
        for path in sorted(schema.keys(), key=lambda p: (p.count('.'), p)):
            current_value = cls._get_nested_value(config, path.split('.'))
            if current_value is _MISSING:
                continue

            meta = schema[path]
            coerced = cls._coerce_value(current_value, meta, path)
            if coerced is not current_value:
                cls._set_nested_value(config, path.split('.'), coerced)

    @classmethod
    def _apply_heuristic_type_coercion(
        cls,
        config: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
        schema_paths = set(schema.keys())

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    path = f"{prefix}.{key}" if prefix else str(key)
                    if isinstance(value, dict):
                        walk(value, path)
                        continue
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                walk(item, path)
                        continue
                    if path in schema_paths:
                        continue
                    if isinstance(value, str):
                        coerced = cls._heuristic_coerce_value(value, path)
                        if coerced is not value:
                            node[key] = coerced
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, dict):
                        walk(item, prefix)

        walk(config, '')

    @classmethod
    def _coerce_value(cls, value: Any, meta: Dict[str, Any], path: str) -> Any:
        if not isinstance(value, str):
            return value

        expected_type = meta.get('type')
        raw_value = value

        if expected_type is bool:
            parsed = cls._parse_bool_value(raw_value)
            if parsed is None:
                logger.warning("Type coercion failed for %s: expected bool, got %r", path, raw_value)
                return value
            return parsed

        if expected_type is int:
            try:
                return int(float(raw_value))
            except (TypeError, ValueError):
                logger.warning("Type coercion failed for %s: expected int, got %r", path, raw_value)
                return value

        if expected_type is float:
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                logger.warning("Type coercion failed for %s: expected float, got %r", path, raw_value)
                return value

        if expected_type is list:
            return cls._coerce_list_value(raw_value, meta, path)

        if expected_type is dict:
            return cls._coerce_dict_value(raw_value, path)

        return value

    @classmethod
    def _coerce_list_value(cls, raw_value: str, meta: Dict[str, Any], path: str) -> Any:
        trimmed = raw_value.strip()
        if trimmed.startswith('[') or trimmed.startswith('{'):
            parsed = cls._parse_structured_value(trimmed, path, warn=True)
            if isinstance(parsed, list):
                coerced = cls._coerce_list_elements(parsed, meta, path)
                return coerced if coerced is not None else raw_value
            if isinstance(parsed, dict):
                logger.warning(
                    "Type coercion failed for %s: expected list, got dict",
                    path,
                )
            return raw_value

        if meta.get('json_only'):
            logger.warning("Type coercion skipped for %s: expected JSON list value", path)
            return raw_value

        parts = [part.strip() for part in trimmed.split(',') if part.strip()]
        if not parts:
            return []

        coerced = cls._coerce_list_elements(parts, meta, path)
        return coerced if coerced is not None else raw_value

    @classmethod
    def _coerce_list_elements(cls, values: List[Any], meta: Dict[str, Any], path: str) -> Optional[List[Any]]:
        elem_type = meta.get('elem_type')
        if elem_type in (int, float):
            parsed_list: List[Union[int, float]] = []
            for part in values:
                try:
                    if isinstance(part, bool):
                        raise ValueError("bool is not a valid numeric")
                    numeric = float(part)
                    parsed_list.append(
                        numeric if elem_type is float else int(numeric)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Type coercion failed for %s: invalid %s element %r",
                        path,
                        elem_type.__name__,
                        part,
                    )
                    return None
            return parsed_list

        if elem_type is bool:
            parsed_list: List[bool] = []
            for part in values:
                if isinstance(part, bool):
                    parsed_list.append(part)
                    continue
                if not isinstance(part, str):
                    logger.warning(
                        "Type coercion failed for %s: invalid bool element %r",
                        path,
                        part,
                    )
                    return None
                parsed = cls._parse_bool_value(part)
                if parsed is None:
                    logger.warning(
                        "Type coercion failed for %s: invalid bool element %r",
                        path,
                        part,
                    )
                    return None
                parsed_list.append(parsed)
            return parsed_list

        if elem_type is str:
            return [str(part).strip() for part in values if str(part).strip()]

        return values

    @staticmethod
    def _coerce_dict_value(raw_value: str, path: str) -> Any:
        trimmed = raw_value.strip()
        parsed = LiveTradingConfiguration._parse_structured_value(trimmed, path, warn=True)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            logger.warning(
                "Type coercion failed for %s: expected dict, got list",
                path,
            )
            return raw_value
        if trimmed.startswith('{') or trimmed.startswith('['):
            return raw_value
        logger.warning("Type coercion failed for %s: expected dict, got %r", path, raw_value)
        return raw_value

    @staticmethod
    def _parse_structured_value(raw_value: str, path: str, *, warn: bool) -> Optional[Any]:
        trimmed = raw_value.strip()
        if not trimmed or not (trimmed.startswith('[') or trimmed.startswith('{')):
            return None
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(trimmed)
        except (ValueError, SyntaxError):
            if warn:
                logger.warning("Type coercion failed for %s: invalid JSON/literal %r", path, raw_value)
            return None
        if isinstance(parsed, (list, dict)):
            return parsed
        if warn:
            logger.warning(
                "Type coercion failed for %s: literal parsed to %s",
                path,
                type(parsed).__name__,
            )
        return None

    @classmethod
    def _heuristic_coerce_value(cls, raw_value: str, path: str) -> Any:
        trimmed = raw_value.strip()
        if not trimmed:
            return raw_value

        parsed_bool = cls._parse_bool_value(trimmed)
        if parsed_bool is not None:
            return parsed_bool

        parsed_int = cls._parse_int_value(trimmed)
        if parsed_int is not None:
            return parsed_int

        parsed_float = cls._parse_float_value(trimmed)
        if parsed_float is not None:
            return parsed_float

        structured = cls._parse_structured_value(trimmed, path, warn=False)
        if isinstance(structured, (list, dict)):
            return structured

        if ',' in trimmed:
            parts = [part.strip() for part in trimmed.split(',') if part.strip()]
            if not parts:
                return []
            numeric_list = cls._coerce_numeric_list(parts)
            if numeric_list is not None:
                return numeric_list
            return parts

        return raw_value

    @classmethod
    def _coerce_numeric_list(cls, parts: List[str]) -> Optional[List[Union[int, float]]]:
        numeric_parts: List[Union[int, float]] = []
        has_float = False
        for part in parts:
            parsed_int = cls._parse_int_value(part)
            if parsed_int is not None:
                numeric_parts.append(parsed_int)
                continue
            parsed_float = cls._parse_float_value(part)
            if parsed_float is not None:
                numeric_parts.append(parsed_float)
                has_float = True
                continue
            return None
        if has_float:
            return [float(value) for value in numeric_parts]
        return numeric_parts

    @staticmethod
    def _parse_int_value(raw_value: str) -> Optional[int]:
        if re.fullmatch(r'[+-]?\d+', raw_value):
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_float_value(raw_value: str) -> Optional[float]:
        if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?', raw_value):
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_bool_value(raw_value: str) -> Optional[bool]:
        normalized = raw_value.strip().lower()
        if normalized in ('true', '1', 'yes', 'y', 'on'):
            return True
        if normalized in ('false', '0', 'no', 'n', 'off'):
            return False
        return None

    @classmethod
    def _get_strict_mode(cls) -> bool:
        strict_env = os.getenv(cls.TYPE_VALIDATION_STRICT_ENV)
        if strict_env is None:
            return False
        parsed = cls._parse_bool_value(strict_env)
        if parsed is None:
            logger.warning(
                "Invalid %s value '%s'; using warn-only mode.",
                cls.TYPE_VALIDATION_STRICT_ENV,
                strict_env,
            )
            return False
        return parsed

    @staticmethod
    def _get_nested_value(config: Dict[str, Any], path: List[str]) -> Any:
        current = config
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return _MISSING
            current = current[key]
        return current

    @staticmethod
    def _set_nested_value(config: Dict[str, Any], path: List[str], value: Any) -> None:
        current = config
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _warn_unknown_appconfig_keys(
        self,
        canonical_schema: Dict[str, Dict[str, Any]],
        operational_schema: Dict[str, Dict[str, Any]],
    ) -> None:
        if not self._appconfig_normalized_paths:
            return

        canonical_paths = set(canonical_schema.keys())
        operational_paths = set(operational_schema.keys())
        legacy_keys = set(self.DEPRECATED_LEGACY_KEYS.keys())
        normalized = sorted(set(self._appconfig_normalized_paths))

        known_runtime = [
            path
            for path in normalized
            if path in operational_paths and path not in canonical_paths
        ]
        if known_runtime:
            logger.info(
                "AppConfig recognized runtime keys (%d): %s",
                len(known_runtime),
                ", ".join(sorted(known_runtime)),
            )

        unknown = [
            path
            for path in normalized
            if path not in canonical_paths
            and path not in operational_paths
            and path not in legacy_keys
        ]
        if not unknown:
            return

        sample_size = 20
        sample = unknown[:sample_size]
        remainder = len(unknown) - len(sample)
        suffix = f" (+{remainder} more)" if remainder > 0 else ""
        message = (
            f"AppConfig keys not in canonical or operational schema ({len(unknown)}): "
            f"{', '.join(sample)}{suffix}"
        )
        if self._get_strict_mode():
            raise ValueError(message)
        logger.warning(message)

    @classmethod
    def _warn_deprecated_keys(cls, config: Dict[str, Any]) -> None:
        for legacy_key, canonical in cls.DEPRECATED_LEGACY_KEYS.items():
            if legacy_key in config:
                logger.warning(
                    "Deprecated config key '%s' detected; use '%s' instead.",
                    legacy_key,
                    canonical,
                )

    def _normalize_str_mtf_config(self, config: Dict[str, Any]) -> None:
        signals_cfg = config.get("signals", {})
        if not isinstance(signals_cfg, dict):
            return
        str_cfg = signals_cfg.get("short_the_rip", {})
        if not isinstance(str_cfg, dict):
            return
        mtf_cfg = str_cfg.get("mtf_confirmation")
        if mtf_cfg is None:
            return
        if not isinstance(mtf_cfg, dict):
            raise ValueError("signals.short_the_rip.mtf_confirmation must be a dict.")

        mtf_policy = build_str_mtf_config(mtf_cfg, strict=self._get_strict_mode(), log=logger)
        str_cfg["mtf_confirmation_effective"] = mtf_policy
        mtf_cfg["15m_mode"] = mtf_policy.tf_15m.mode
        mtf_cfg["1h_mode"] = mtf_policy.tf_1h.mode
        mtf_cfg["missing_15m_is_fatal"] = mtf_policy.tf_15m.missing_is_fatal
        mtf_cfg["missing_1h_is_fatal"] = mtf_policy.tf_1h.missing_is_fatal
        mtf_cfg["on_missing_15m"] = mtf_policy.tf_15m.on_missing
        mtf_cfg["on_missing_1h"] = mtf_policy.tf_1h.on_missing
        mtf_cfg["mtf_policy_summary"] = mtf_policy.summary

    def _normalize_mean_reversion_dynamic_controller_config(self, config: Dict[str, Any]) -> None:
        strategies_cfg = config.get("strategies")
        if not isinstance(strategies_cfg, dict):
            return

        mr_cfg = strategies_cfg.get("mean_reversion")
        if not isinstance(mr_cfg, dict):
            return

        controller_cfg = mr_cfg.get("dynamic_controller")
        if controller_cfg is None:
            return

        if not isinstance(controller_cfg, dict):
            logger.warning(
                "strategies.mean_reversion.dynamic_controller must be a dict. Disabling controller."
            )
            mr_cfg["dynamic_controller"] = {"enabled": False}
            return

        controller_cfg["enabled"] = bool(controller_cfg.get("enabled", False))

        target = self._coerce_float(
            controller_cfg.get("target_outside_pct", 0.10),
            0.10,
            "strategies.mean_reversion.dynamic_controller.target_outside_pct",
            minimum=0.0,
        )
        if target <= 0.0 or target >= 0.5:
            logger.warning(
                "strategies.mean_reversion.dynamic_controller.target_outside_pct out of bounds. Resetting to 0.10."
            )
            target = 0.10
        controller_cfg["target_outside_pct"] = target

        abs_z_window = self._coerce_int(
            controller_cfg.get("abs_z_window", 500),
            500,
            "strategies.mean_reversion.dynamic_controller.abs_z_window",
            minimum=1,
        )
        warmup_samples = self._coerce_int(
            controller_cfg.get("warmup_samples", 50),
            50,
            "strategies.mean_reversion.dynamic_controller.warmup_samples",
            minimum=1,
        )
        if warmup_samples > abs_z_window:
            logger.warning(
                "strategies.mean_reversion.dynamic_controller.warmup_samples exceeds abs_z_window. "
                "Clamping warmup_samples to abs_z_window."
            )
            warmup_samples = abs_z_window
        controller_cfg["abs_z_window"] = abs_z_window
        controller_cfg["warmup_samples"] = warmup_samples

        controller_cfg["update_interval_sec"] = self._coerce_float(
            controller_cfg.get("update_interval_sec", 300),
            300,
            "strategies.mean_reversion.dynamic_controller.update_interval_sec",
            minimum=0.0,
        )
        controller_cfg["min_m_change"] = self._coerce_float(
            controller_cfg.get("min_m_change", 0.05),
            0.05,
            "strategies.mean_reversion.dynamic_controller.min_m_change",
            minimum=0.0,
        )
        controller_cfg["log_every_update"] = bool(controller_cfg.get("log_every_update", True))
        controller_cfg["freeze_on_trend"] = bool(controller_cfg.get("freeze_on_trend", True))
        controller_cfg["adx_freeze_threshold"] = self._coerce_float(
            controller_cfg.get("adx_freeze_threshold", mr_cfg.get("adx_threshold", 25)),
            float(mr_cfg.get("adx_threshold", 25) or 25),
            "strategies.mean_reversion.dynamic_controller.adx_freeze_threshold",
            minimum=0.0,
        )

        m_min = self._coerce_float(
            controller_cfg.get("m_min", 1.0),
            1.0,
            "strategies.mean_reversion.dynamic_controller.m_min",
            minimum=0.0,
        )
        m_max = self._coerce_float(
            controller_cfg.get("m_max", 2.5),
            2.5,
            "strategies.mean_reversion.dynamic_controller.m_max",
            minimum=0.0,
        )
        if m_max < m_min:
            logger.warning(
                "strategies.mean_reversion.dynamic_controller.m_max < m_min. Swapping values."
            )
            m_min, m_max = m_max, m_min
        controller_cfg["m_min"] = m_min
        controller_cfg["m_max"] = m_max

        lookback_base = mr_cfg.get("vwap_lookback", 1440)
        try:
            lookback_base = int(lookback_base)
        except Exception:
            lookback_base = 1440

        dyn_lb_cfg = controller_cfg.get("dynamic_lookback", {})
        if dyn_lb_cfg is None:
            dyn_lb_cfg = {}
        if not isinstance(dyn_lb_cfg, dict):
            logger.warning(
                "strategies.mean_reversion.dynamic_controller.dynamic_lookback must be a dict. Disabling lookback control."
            )
            dyn_lb_cfg = {"enabled": False}

        dyn_lb_cfg["enabled"] = bool(dyn_lb_cfg.get("enabled", False))
        dyn_lb_cfg["lookback_min"] = self._coerce_int(
            dyn_lb_cfg.get("lookback_min", 120),
            120,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.lookback_min",
            minimum=1,
        )
        dyn_lb_cfg["lookback_max"] = self._coerce_int(
            dyn_lb_cfg.get("lookback_max", lookback_base),
            lookback_base,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.lookback_max",
            minimum=int(dyn_lb_cfg["lookback_min"]),
        )
        dyn_lb_cfg["lookback_static"] = self._coerce_int(
            dyn_lb_cfg.get("lookback_static", lookback_base),
            lookback_base,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.lookback_static",
            minimum=int(dyn_lb_cfg["lookback_min"]),
        )
        dyn_lb_cfg["atr_squeeze_pct"] = self._coerce_float(
            dyn_lb_cfg.get("atr_squeeze_pct", 0.0015),
            0.0015,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.atr_squeeze_pct",
            minimum=0.0,
        )
        dyn_lb_cfg["atr_expand_pct"] = self._coerce_float(
            dyn_lb_cfg.get("atr_expand_pct", 0.0040),
            0.0040,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.atr_expand_pct",
            minimum=0.0,
        )
        dyn_lb_cfg["atr_hysteresis_pct"] = self._coerce_float(
            dyn_lb_cfg.get("atr_hysteresis_pct", 0.0002),
            0.0002,
            "strategies.mean_reversion.dynamic_controller.dynamic_lookback.atr_hysteresis_pct",
            minimum=0.0,
        )

        controller_cfg["dynamic_lookback"] = dyn_lb_cfg
        mr_cfg["dynamic_controller"] = controller_cfg

    @classmethod
    def _validate_schema_types(
        cls,
        config: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> None:
        strict = cls._get_strict_mode()

        for path in sorted(schema.keys(), key=lambda p: (p.count('.'), p)):
            meta = schema.get(path)
            if not meta:
                continue

            value = cls._get_nested_value(config, path.split('.'))
            if value is _MISSING:
                continue

            expected = meta.get('type')
            if expected is bool:
                ok = isinstance(value, bool)
            elif expected is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif expected is float:
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected is list:
                ok = isinstance(value, list)
            elif expected is dict:
                ok = isinstance(value, dict)
            elif expected is None:
                ok = True
            else:
                ok = isinstance(value, expected)

            if ok:
                continue

            expected_name = expected.__name__ if hasattr(expected, '__name__') else str(expected)
            message = (
                f"Type validation failed for {path}: expected {expected_name}, "
                f"got {type(value).__name__} ({value!r})"
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

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

    def _apply_trigger_price_defaults(self, config: Dict[str, Any]) -> None:
        """Ensure trigger price defaults are set even when omitted from YAML/env."""
        trigger_cfg = config.setdefault('trigger_price', {})
        trigger_cfg.setdefault('source', 'mid')
        trigger_cfg.setdefault('diag_interval_sec', 60)

        strategies_cfg = config.setdefault('strategies', {})
        adaptive_cfg = strategies_cfg.setdefault('adaptive_ob', {})
        adaptive_cfg.setdefault('adaptive_ob_trigger_price_source', trigger_cfg.get('source', 'mid'))

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
            # Canonical key (RiskConfiguration expects `max_position_size`).
            'max_position_size',
            # Legacy key (ignored by RiskConfiguration; validated/blocked by schema layer).
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
            logger.warning("PER_TRADE_RISK_PCT not set, defaulting to 0.3% (0.003)")
            per_trade_raw = 0.003

        normalized_per_trade = self._normalize_percent_value(per_trade_raw, 'risk.per_trade_risk_pct')
        if not normalized_per_trade or normalized_per_trade <= 0 or normalized_per_trade > 1:
            logger.error(
                f"❌ per_trade_risk_pct out of bounds after normalization: {normalized_per_trade}. Resetting to 0.003 (0.3%)."
            )
            normalized_per_trade = 0.003
        risk_section['per_trade_risk_pct'] = normalized_per_trade

        # --- Normalize max_portfolio_risk (Portfolio Heat) ---
        portfolio_risk_raw = risk_section.get('max_portfolio_risk') or risk_section.get('max_portfolio_risk_pct')
        if portfolio_risk_raw is None:
            # Default to 6% (Balanced preset)
            portfolio_risk_raw = 0.06
        
        normalized_portfolio_risk = self._normalize_percent_value(portfolio_risk_raw, 'risk.max_portfolio_risk')
        if not normalized_portfolio_risk or normalized_portfolio_risk <= 0:
            logger.warning(f"⚠️ Invalid max_portfolio_risk: {normalized_portfolio_risk}. Resetting to 0.06 (6%).")
            normalized_portfolio_risk = 0.06
        
        risk_section['max_portfolio_risk'] = normalized_portfolio_risk
        # Ensure consistency
        risk_section['max_portfolio_risk_pct'] = normalized_portfolio_risk

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

        # Normalize advanced sub-sections
        queue_cfg = risk_section.get('queue') or {}
        risk_section['queue'] = self._normalize_queue_config(queue_cfg)

        concurrent_cfg = risk_section.get('concurrent_limits') or {}
        risk_section['concurrent_limits'] = self._normalize_concurrent_limits(concurrent_cfg)

        volatility_cfg = risk_section.get('volatility_sizing') or {}
        risk_section['volatility_sizing'] = self._normalize_volatility_sizing(volatility_cfg)

        config['risk'] = risk_section

    def _apply_websocket_defaults(self, config: Dict[str, Any]) -> None:
        """Ensure websocket diagnostic defaults exist for trigger price resolution."""
        ws_cfg = config.setdefault('websocket', {})
        ws_cfg.setdefault('ticker_stale_ms', 5000)
        ws_cfg.setdefault('trigger_diag_interval_sec', 60)

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

    def _normalize_queue_config(self, queue_cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            'ttl_seconds': 60,
            'max_queue_depth': 50,
            'batch_dequeue': 3,
            'max_pending_per_symbol': 1,
            'max_pending_scale_in_per_symbol': 0,
        }
        normalized = {}
        for key, default in defaults.items():
            minimum = 1 if key != 'max_pending_scale_in_per_symbol' else 0
            normalized[key] = self._coerce_int(queue_cfg.get(key, default), default, f"risk.queue.{key}", minimum=minimum)

        weight_defaults = {
            'explicit_priority': 0.4,
            'risk_reward': 0.3,
            'ml_confidence': 0.2,
            'urgency': 0.1,
            'regime_alignment': 0.05,
            'strategy_urgency': 0.05,
        }
        weights = {}
        provided_weights = queue_cfg.get('priority_weights') or {}
        total = 0.0
        for key, default in weight_defaults.items():
            value = self._coerce_float(provided_weights.get(key, default), default, f"risk.queue.priority_weights.{key}", minimum=0)
            weights[key] = value
            total += value

        # Surface any user-provided keys we don't yet honor to reduce confusion.
        unknown_keys = set(provided_weights.keys()) - set(weight_defaults.keys())
        if unknown_keys:
            logger.warning(
                "⚠️ Ignoring unsupported priority weight keys: %s",
                ', '.join(sorted(unknown_keys))
            )

        if total <= 0:
            logger.warning("risk.queue.priority_weights sum to <= 0. Reverting to defaults.")
            weights = weight_defaults.copy()
            total = sum(weights.values())

        normalized['priority_weights'] = {k: v / total for k, v in weights.items()}
        return normalized

    def _normalize_concurrent_limits(self, limits_cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            'max_open_positions': 3,
            'max_positions_per_symbol': 1,
            'max_total_risk_pct': 0.06,
            'correlation_bucket_threshold': 0.8,
        }
        normalized = {
            'max_open_positions': self._coerce_int(limits_cfg.get('max_open_positions', defaults['max_open_positions']),
                                                   defaults['max_open_positions'], 'risk.concurrent_limits.max_open_positions', minimum=0),
            'max_positions_per_symbol': self._coerce_int(
                limits_cfg.get('max_positions_per_symbol', defaults['max_positions_per_symbol']),
                defaults['max_positions_per_symbol'],
                'risk.concurrent_limits.max_positions_per_symbol',
                minimum=0,
            ),
        }

        max_total_risk = limits_cfg.get('max_total_risk_pct', defaults['max_total_risk_pct'])
        normalized['max_total_risk_pct'] = self._normalize_percent_value(max_total_risk, 'risk.concurrent_limits.max_total_risk_pct') or defaults['max_total_risk_pct']

        corr_threshold = limits_cfg.get('correlation_bucket_threshold', defaults['correlation_bucket_threshold'])
        normalized['correlation_bucket_threshold'] = self._normalize_percent_value(
            corr_threshold,
            'risk.concurrent_limits.correlation_bucket_threshold',
        ) or defaults['correlation_bucket_threshold']

        return normalized

    def _normalize_volatility_sizing(self, vol_cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            'enabled': True,
            'atr_window': 14,
            'atr_floor_pct': 0.005,
            'atr_ceiling_pct': 0.02,
            'low_vol_multiplier': 1.2,
            'baseline_multiplier': 1.0,
            'high_vol_multiplier': 0.6,
            'min_position_size_pct': 0.01,
        }

        normalized = {
            'enabled': bool(vol_cfg.get('enabled', defaults['enabled'])),
            'atr_window': self._coerce_int(vol_cfg.get('atr_window', defaults['atr_window']), defaults['atr_window'], 'risk.volatility_sizing.atr_window', minimum=1),
            'low_vol_multiplier': self._coerce_float(vol_cfg.get('low_vol_multiplier', defaults['low_vol_multiplier']), defaults['low_vol_multiplier'], 'risk.volatility_sizing.low_vol_multiplier', minimum=0.1),
            'baseline_multiplier': self._coerce_float(vol_cfg.get('baseline_multiplier', defaults['baseline_multiplier']), defaults['baseline_multiplier'], 'risk.volatility_sizing.baseline_multiplier', minimum=0.1),
            'high_vol_multiplier': self._coerce_float(vol_cfg.get('high_vol_multiplier', defaults['high_vol_multiplier']), defaults['high_vol_multiplier'], 'risk.volatility_sizing.high_vol_multiplier', minimum=0.05),
        }

        normalized['atr_floor_pct'] = self._normalize_percent_value(
            vol_cfg.get('atr_floor_pct', defaults['atr_floor_pct']), 'risk.volatility_sizing.atr_floor_pct'
        ) or defaults['atr_floor_pct']
        normalized['atr_ceiling_pct'] = self._normalize_percent_value(
            vol_cfg.get('atr_ceiling_pct', defaults['atr_ceiling_pct']), 'risk.volatility_sizing.atr_ceiling_pct'
        ) or defaults['atr_ceiling_pct']
        normalized['min_position_size_pct'] = self._normalize_percent_value(
            vol_cfg.get('min_position_size_pct', defaults['min_position_size_pct']), 'risk.volatility_sizing.min_position_size_pct'
        ) or defaults['min_position_size_pct']

        if normalized['atr_floor_pct'] >= normalized['atr_ceiling_pct']:
            logger.warning("risk.volatility_sizing atr_floor_pct >= atr_ceiling_pct; adjusting ceiling to maintain spread")
            normalized['atr_ceiling_pct'] = normalized['atr_floor_pct'] * 1.5

        return normalized

    def _load_from_app_config(self) -> Dict[str, Any]:
        """
        Load configuration from Azure App Configuration using REST API.
        
        Uses REST API directly instead of SDK to work around IMDS API version issues
        with Managed Identity on some Azure VMs.
        
        Returns:
            Dict with configuration overrides from App Configuration.
            Returns empty dict if not available or on any error (graceful fallback).
        """
        if not AZURE_APPCONFIG_AVAILABLE:
            return {}
        
        endpoint = os.getenv('AZURE_APPCONFIG_ENDPOINT')
        if not endpoint:
            logger.debug("AZURE_APPCONFIG_ENDPOINT not set, skipping App Configuration load")
            return {}
        
        try:
            import json
            import subprocess
            from urllib.parse import quote
            
            label = os.getenv('AZURE_APPCONFIG_LABEL', 'production')
            
            logger.info(f"📡 Loading configuration from Azure App Configuration (via REST API)...")
            logger.info(f"   Endpoint: {endpoint}")
            logger.info(f"   Label: {label}")
            
            # Get token from IMDS using correct API version (2017-12-01)
            try:
                token_cmd = [
                    'curl', '-s', '-H', 'Metadata:true',
                    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2017-12-01&resource=https://appconfig.azure.com'
                ]
                try:
                    token_response = subprocess.check_output(token_cmd, text=True, stderr=subprocess.STDOUT)
                    token_data = json.loads(token_response)
                    access_token = token_data.get('access_token')
                    
                    if not access_token:
                        logger.warning("⚠️ Failed to acquire token from IMDS (no access_token in response)")
                        logger.debug(f"   IMDS Response: {token_response[:200]}")
                        return {}
                except FileNotFoundError:
                    logger.error("❌ curl command not found in container. Install curl in Dockerfile.")
                    return {}
                    
                # Query App Configuration REST API
                url = f"{endpoint}/kv?key=BearishAlphaBot/*&label={quote(label)}&api-version=1.0"
                
                curl_cmd = ['curl', '-s', '-H', f'Authorization: Bearer {access_token}', url]
                try:
                    response = subprocess.check_output(curl_cmd, text=True, stderr=subprocess.STDOUT)
                    response_data = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"❌ Failed to parse App Configuration response as JSON")
                    logger.debug(f"   Response: {response[:500]}")
                    return {}
                
                # Extract items from response
                items = response_data.get('items', [])
                app_config_dict = {}
                
                for item in items:
                    key = item.get('key', '')
                    # Remove prefix 'BearishAlphaBot/'
                    if key.startswith('BearishAlphaBot/'):
                        key = key[len('BearishAlphaBot/'):]
                    app_config_dict[key] = item.get('value', '')
                
                if app_config_dict:
                    logger.info(f"✅ Loaded {len(app_config_dict)} settings from App Configuration")
                    app_config_dict = self._sanitize_appconfig_flat_dict(app_config_dict)
                    self._appconfig_raw_keys = tuple(sorted(app_config_dict.keys()))
                    self._appconfig_normalized_paths = self._normalize_appconfig_keys(self._appconfig_raw_keys)
                    nested_config = self._flatten_to_nested(app_config_dict)
                    logger.info(f"   Converted to nested structure")
                    return nested_config
                else:
                    logger.info("⊘ No settings found in App Configuration")
                    return {}
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ curl command failed with exit code {e.returncode}: {e}")
                return {}
                
        except Exception as e:
            logger.error(
                f"❌ Failed to load from Azure App Configuration: {e}",
                exc_info=True
            )
            return {}
    
    def _sanitize_appconfig_flat_dict(self, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite/drop stale AppConfig keys so strict schema mode can proceed safely."""
        if not isinstance(flat_dict, dict) or not flat_dict:
            return {}

        def _has_prefix(key: str, prefix: str) -> bool:
            return key == prefix or key.startswith(prefix + '.')

        sanitized: Dict[str, Any] = {}
        dropped: List[str] = []
        rewritten: List[str] = []
        seen_lower: set = set()

        for raw_key, value in flat_dict.items():
            key_str = str(raw_key).strip().strip('"').strip("'")
            if not key_str:
                continue

            key_lower = key_str.lower()

            # Drop known removed blocks (prevents strict-mode startup failure).
            if any(_has_prefix(key_lower, prefix) for prefix in self.APPCONFIG_DROP_PREFIXES):
                dropped.append(key_str)
                continue

            # Exact key rewrites.
            target = self.APPCONFIG_KEY_REWRITES.get(key_lower)
            if target:
                target_lower = target.lower()
                if target_lower in seen_lower:
                    # Prefer the already-present canonical key over legacy alias.
                    dropped.append(key_str)
                    continue
                sanitized[target] = value
                seen_lower.add(target_lower)
                rewritten.append(f"{key_str} -> {target}")
                continue

            # Prefix rewrites (strategy renames, etc.).
            prefix_rewritten = False
            for old_prefix, new_prefix in self.APPCONFIG_PREFIX_REWRITES.items():
                old_lower = old_prefix.lower()
                if not _has_prefix(key_lower, old_lower):
                    continue

                suffix = key_lower[len(old_lower):]
                target_key = new_prefix + suffix
                target_lower = target_key.lower()
                if target_lower in seen_lower:
                    dropped.append(key_str)
                    prefix_rewritten = True
                    break

                sanitized[target_key] = value
                seen_lower.add(target_lower)
                rewritten.append(f"{key_str} -> {target_key}")
                prefix_rewritten = True
                break

            if prefix_rewritten:
                continue

            # Default: keep key as-is (first occurrence wins).
            if key_lower in seen_lower:
                dropped.append(key_str)
                continue

            sanitized[key_str] = value
            seen_lower.add(key_lower)

        if rewritten:
            sample = sorted(rewritten)[:20]
            suffix = " (+more)" if len(rewritten) > 20 else ""
            logger.warning(
                "?? [APPCONFIG] Rewrote deprecated keys (%d): %s%s",
                len(rewritten),
                ", ".join(sample),
                suffix,
            )
        if dropped:
            sample = sorted(dropped)[:20]
            suffix = " (+more)" if len(dropped) > 20 else ""
            logger.warning(
                "?? [APPCONFIG] Ignored deprecated/duplicate keys (%d): %s%s",
                len(dropped),
                ", ".join(sample),
                suffix,
            )

        return sanitized

    @classmethod
    def _flatten_to_nested(cls, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat dictionary keys to nested structure.
        
        Example:
            {'TRADING_MODE': 'paper', 'CAPITAL_USDT': '1000'}
            ->
            {'trading_mode': 'paper', 'capital_usdt': '1000'}
            (keys lowercase to match config.example.yaml structure, symbol segments preserved)
        """
        nested: Dict[str, Any] = {}

        for key, value in flat_dict.items():
            parts = cls._normalize_appconfig_key(key)
            if not parts:
                continue

            cursor = nested
            for segment in parts[:-1]:
                if segment not in cursor or not isinstance(cursor.get(segment), dict):
                    cursor[segment] = {}
                cursor = cursor[segment]
            cursor[parts[-1]] = value

        return nested

    @classmethod
    def _normalize_appconfig_keys(cls, keys: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        for key in keys:
            parts = cls._normalize_appconfig_key(key)
            if parts:
                normalized.append('.'.join(parts))
        return normalized

    @classmethod
    def _normalize_appconfig_key(cls, key: str) -> List[str]:
        parts = key.split('.') if '.' in key else [key]
        normalized: List[str] = []
        for segment in parts:
            cleaned = segment.strip()
            if not cleaned:
                continue
            normalized.append(cls._normalize_appconfig_segment(cleaned))
        return normalized

    @classmethod
    def _normalize_appconfig_segment(cls, segment: str) -> str:
        stripped = segment.strip().strip('"').strip("'")
        if cls._is_symbol_segment(stripped):
            return cls._normalize_symbol(stripped)
        return stripped.lower()

    @classmethod
    def _is_symbol_segment(cls, segment: str) -> bool:
        if not segment:
            return False
        if _TRADING_SYMBOL_PATTERN.match(segment):
            return True
        if _DERIVATIVE_SYMBOL_PATTERN.match(segment):
            return True
        return False

    @staticmethod
    def _coerce_int(value: Any, default: int, field_name: str, minimum: Optional[int] = None) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ {field_name} value '{value}' invalid. Using default {default}.")
            numeric = default

        if minimum is not None and numeric < minimum:
            logger.warning(f"⚠️ {field_name} value {numeric} below minimum {minimum}. Raising to minimum.")
            numeric = minimum
        return numeric

    @staticmethod
    def _coerce_float(value: Any, default: float, field_name: str, minimum: Optional[float] = None) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ {field_name} value '{value}' invalid. Using default {default}.")
            numeric = default

        if minimum is not None and numeric < minimum:
            logger.warning(f"⚠️ {field_name} value {numeric} below minimum {minimum}. Raising to minimum.")
            numeric = minimum
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

        mtf_summary = get_nested(
            config,
            ["signals", "short_the_rip", "mtf_confirmation", "mtf_policy_summary"],
            None,
        )
        if mtf_summary:
            logger.info(f"STR MTF policy: {mtf_summary}")

        # Pyramiding summary
        logger.info("Pyramiding Settings:")
        pyramiding_enabled = bool(get_nested(config, ['pyramiding', 'enabled'], False))
        logger.info(f"   Enabled: {pyramiding_enabled}")
        logger.info(f"   Max layers per symbol: {get_nested(config, ['pyramiding', 'max_layers_per_symbol'], 'N/A')}")
        logger.info(
            f"   Min scale-in quality: {get_nested(config, ['pyramiding', 'min_scale_in_quality'], 'N/A')} | "
            f"Min scale-in PnL pct: {get_nested(config, ['pyramiding', 'min_scale_in_unrealized_pnl_pct'], 'N/A')} | "
            f"Min scale-in distance pct: {get_nested(config, ['pyramiding', 'min_scale_in_distance_pct'], 'N/A')}"
        )
        logger.info(
            f"   Queue max pending scale_in per symbol: "
            f"{get_nested(config, ['risk', 'queue', 'max_pending_scale_in_per_symbol'], 'N/A')}"
        )

        # Position management / exits
        logger.info("Position Management:")
        exit_enabled = get_nested(config, ['position_management', 'exit_monitoring', 'enabled'], True)
        exit_frequency = get_nested(config, ['position_management', 'exit_monitoring', 'check_frequency'], 'N/A')
        logger.info(f"   Exit monitoring enabled: {exit_enabled} | check_frequency: {exit_frequency}")

        engine_loop_interval = get_nested(
            config,
            ['position_management', 'position_monitoring_loop_interval_s'],
            10,
        )
        logger.info(f"   Position monitoring loop interval (engine): {engine_loop_interval}s")
        logger.info(
            "   Exit guardrails eps: "
            f"{get_nested(config, ['position_management', 'exit_guardrails', 'eps'], 0.0)}"
        )

        # NOTE: Trailing stop is resolved per-position:
        #   Signal overrides > Strategy execution_profile > Global defaults.
        # The global position_management.trailing_stop block is intended to contain
        # only safety floors (min step / min update interval) that cannot be overridden.
        trailing_block = get_nested(config, ['position_management', 'trailing_stop'], {})
        if not isinstance(trailing_block, dict):
            trailing_block = {}

        legacy_ts_keys = ('trailing_stop_enabled', 'trailing_stop_distance', 'activation_threshold')
        legacy_ts_present = any(key in trailing_block for key in legacy_ts_keys)

        min_step_bps = get_nested(config, ['position_management', 'trailing_stop', 'min_trail_step_bps'], 0)
        min_update_s = get_nested(config, ['position_management', 'trailing_stop', 'min_trail_update_interval_s'], 0)

        if legacy_ts_present:
            logger.info(
                "   Trailing stop (legacy global defaults): "
                f"enabled={get_nested(config, ['position_management', 'trailing_stop', 'trailing_stop_enabled'], False)} "
                f"distance={get_nested(config, ['position_management', 'trailing_stop', 'trailing_stop_distance'], 'N/A')} "
                f"activation_threshold={get_nested(config, ['position_management', 'trailing_stop', 'activation_threshold'], 'N/A')} "
                f"| safety_floors: min_trail_step_bps={min_step_bps} min_trail_update_interval_s={min_update_s}"
            )
        else:
            profiles = get_nested(config, ['execution_profiles'], {})
            if not isinstance(profiles, dict):
                profiles = {}

            strategies = get_nested(config, ['strategies'], {})
            if not isinstance(strategies, dict):
                strategies = {}

            used_profiles: List[str] = []
            for strat_cfg in strategies.values():
                if not isinstance(strat_cfg, dict):
                    continue
                name = (strat_cfg.get('execution_profile') or '').strip()
                if name:
                    used_profiles.append(name)

            # Preserve order but de-duplicate.
            used_profiles = list(dict.fromkeys(used_profiles))
            if not used_profiles and profiles:
                used_profiles = sorted(profiles.keys())

            profile_summaries: List[str] = []
            for profile_name in used_profiles:
                profile_cfg = profiles.get(profile_name)
                if not isinstance(profile_cfg, dict):
                    continue
                ts_cfg = profile_cfg.get('trailing_stop')
                if not isinstance(ts_cfg, dict) or not ts_cfg:
                    continue

                enabled = ts_cfg.get('enabled', 'N/A')
                delta = ts_cfg.get('delta_pct', 'N/A')
                activation = ts_cfg.get('activation_threshold_pct', 'N/A')
                profile_summaries.append(
                    f"{profile_name}(enabled={enabled} delta_pct={delta} activation_threshold_pct={activation})"
                )

            if profile_summaries:
                logger.info(
                    "   Trailing stop (resolved via execution_profiles; signal overrides can override): "
                    + "; ".join(profile_summaries)
                    + f" | safety_floors: min_trail_step_bps={min_step_bps} min_trail_update_interval_s={min_update_s}"
                )
            else:
                logger.info(
                    "   Trailing stop: not configured in global defaults or execution_profiles "
                    f"(safety_floors: min_trail_step_bps={min_step_bps} min_trail_update_interval_s={min_update_s})"
                )
        logger.info(
            "   Trigger price: "
            f"source={get_nested(config, ['trigger_price', 'source'], 'mid')} "
            f"diag_interval_sec={get_nested(config, ['trigger_price', 'diag_interval_sec'], 'N/A')} "
            f"ticker_stale_ms={get_nested(config, ['websocket', 'ticker_stale_ms'], 'N/A')} "
            f"trigger_diag_interval_sec={get_nested(config, ['websocket', 'trigger_diag_interval_sec'], 'N/A')}"
        )
        
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
