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

import os
import re
import yaml
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)

# Singleton instance storage to ensure config is loaded only once.
_config_instance: Optional[Dict[str, Any]] = None

class LiveTradingConfiguration:
    """
    A dynamic, singleton configuration loader.

    It reads `config.example.yaml`, parses special `# Override with: ENV_VAR`
    comments, and intelligently merges environment variables with automatic
    type casting. The result is cached and served on all subsequent calls.
    """
    CONFIG_FILE_PATH = 'config/config.example.yaml'
    ENV_OVERRIDE_PATTERN = re.compile(r'#\s*Override with:\s*(\w+)')

    @classmethod
    def load(cls) -> Dict[str, Any]:
        """
        Main entry point. Loads, merges, and returns the configuration.
        Uses a singleton pattern to load only once.
        """
        global _config_instance
        if _config_instance is not None:
            logger.debug("Returning cached configuration instance.")
            return _config_instance

        logger.info("=" * 70)
        logger.info("🔧 DYNAMIC CONFIGURATION LOADER (v2.0 - Singleton)")
        logger.info("=" * 70)
        
        instance = cls()
        try:
            config = instance._load_and_merge_configs()
            _config_instance = config
            instance._log_config_summary(config)
            return _config_instance
        except Exception as e:
            logger.critical(f"❌ A critical error occurred during configuration loading: {e}", exc_info=True)
            raise RuntimeError("Failed to load configuration. Bot cannot start.") from e

    def _load_and_merge_configs(self) -> Dict[str, Any]:
        """Orchestrates the loading and merging process."""
        # 1. Load the base YAML config and parse env var mappings from its comments
        yaml_config, env_map = self._load_yaml_and_map_env_vars()
        if not yaml_config:
            raise ValueError("Base configuration from YAML is empty or could not be loaded.")

        # 2. Normalize YAML values (e.g., convert trading symbol strings to lists)
        yaml_config = self._normalize_yaml_values(yaml_config)

        # 3. Get overrides from environment variables using the parsed map
        env_overrides = self._get_env_overrides(env_map, yaml_config)

        # 4. Deep merge YAML config with environment overrides
        return self._deep_merge(yaml_config, env_overrides)

    def _load_yaml_and_map_env_vars(self) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Loads the YAML file line by line to extract both config and env mappings.
        This is a robust way to link comments to their corresponding keys.
        
        Returns:
            A tuple: (loaded_yaml_dict, env_var_to_path_map)
        """
        env_map: Dict[str, List[str]] = {}
        
        if not os.path.exists(self.CONFIG_FILE_PATH):
            raise FileNotFoundError(f"Configuration file not found at: {self.CONFIG_FILE_PATH}")

        with open(self.CONFIG_FILE_PATH, 'r') as f:
            lines = f.readlines()
            
        path_stack: List[Tuple[int, str]] = []
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                continue

            indentation = len(line) - len(line.lstrip(' '))
            
            try:
                key_part = stripped_line.split(':', 1)[0].strip()
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
                logger.debug(f"Mapped ENV '{env_var}' to config path: {current_path}")

        with open(self.CONFIG_FILE_PATH, 'r') as f:
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
                # Apply the same logic as _cast_value for string values
                if '/' in obj:  # Trading symbol format
                    if ',' in obj:
                        return [s.strip() for s in obj.split(',') if s.strip()]
                    else:
                        return [obj.strip()]
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

        for env_var, path in env_map.items():
            env_value_str = os.getenv(env_var)
            if env_value_str is None or env_value_str == '':
                continue

            try:
                original_value = base_config
                for key in path:
                    original_value = original_value[key]
                
                target_type = type(original_value)
                converted_value = self._cast_value(env_value_str, target_type)
                
                # Build the nested dictionary for the override
                temp_dict = overrides
                for key in path[:-1]:
                    temp_dict = temp_dict.setdefault(key, {})
                temp_dict[path[-1]] = converted_value

                logger.info(f"  ✓ Applied ENV: {env_var} = {converted_value} (as {target_type.__name__})")

            except KeyError:
                logger.warning(f"  ⚠️ ENV var '{env_var}' found, but path '{'.'.join(path)}' is invalid in YAML. Ignoring.")
            except Exception:
                logger.error(f"  ❌ Failed to process ENV var '{env_var}' with value '{env_value_str}'", exc_info=True)
        
        return overrides

    @staticmethod
    def _cast_value(value_str: str, target_type: type) -> Any:
        """Helper to convert a string value to a specific target type."""
        try:
            # === CRITICAL FIX: Handle trading symbols specifically ===
            # Trading symbols have the format "BTC/USDT" or "BTC/USDT:USDT"
            # They must be converted to a list, even if there's only one symbol
            if '/' in value_str:  # Detect trading pair format
                if ',' in value_str:
                    # Multiple symbols: "BTC/USDT,ETH/USDT"
                    return [s.strip() for s in value_str.split(',') if s.strip()]
                else:
                    # Single symbol: "BTC/USDT" -> ["BTC/USDT"]
                    return [value_str.strip()]
            # === END CRITICAL FIX ===

            # Original type-based casting for non-trading-symbol values
            if target_type is bool:
                return value_str.lower() in ('true', '1', 't', 'y', 'yes')
            if target_type is int:
                return int(value_str)
            if target_type is float:
                return float(value_str)
            if target_type is list:
                return [s.strip() for s in value_str.split(',') if s.strip()]
            if target_type is str:
                # For strings that look like lists (comma-separated)
                if ',' in value_str:
                    return [s.strip() for s in value_str.split(',') if s.strip()]
                return value_str
            return value_str  # Assume string
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

        # Trading Universe
        symbols_val = get_nested(config, ['universe', 'fixed_symbols'], [])
        symbols = symbols_val if isinstance(symbols_val, list) else [s.strip() for s in str(symbols_val).split(',')]
        logger.info(f"🎯 Trading Universe: {len(symbols)} symbols")
        if symbols: logger.info(f"   - {', '.join(symbols)}")
        
        # Risk Management
        logger.info("💰 Risk Management:")
        logger.info(f"   Capital: ${get_nested(config, ['risk', 'equity_usd'], 0):.2f} USDT")
        logger.info(f"   Max Notional Per Trade: {get_nested(config, ['risk', 'max_notional_per_trade'], 0):.2f} USDT")
        
        logger.info("="*70)

# Global accessor function for easy, consistent access from anywhere in the codebase.
def get_config() -> Dict[str, Any]:
    """
    Global accessor for the singleton configuration instance.
    This should be the ONLY way other modules get the configuration.
    """
    return LiveTradingConfiguration.load()
