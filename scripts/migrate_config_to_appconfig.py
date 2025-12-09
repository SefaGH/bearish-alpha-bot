#!/usr/bin/env python3
"""
Azure App Configuration Migration Script

Migrates bearish-bot.env settings to Azure App Configuration and Azure Key Vault.

Sensitive data (credentials) -> Key Vault secrets
Non-sensitive data (config) -> App Configuration direct values

Usage:
    python scripts/migrate_config_to_appconfig.py \
        --env-file /path/to/bearish-bot.env \
        --app-config-name appcs-bearish-bot \
        --app-config-rg TradeBot \
        --keyvault-name bearish-kv \
        --keyvault-rg tradebot-ops \
        --prefix BearishAlphaBot/ \
        --label production \
        --phase 1 \
        [--dry-run]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from azure.identity import DefaultAzureCredential
from azure.appconfiguration import AzureAppConfigurationClient, ConfigurationSetting
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceExistsError, ResourceNotoundError


class SettingType(Enum):
    """Setting classification"""
    SENSITIVE = "sensitive"  # Goes to Key Vault
    CONIG = "config"  # Direct value in App Config


@dataclass
class Setting:
    """Represents a single configuration setting"""
    key: str
    value: str
    type: SettingType
    label: str = "production"
    content_type: str = "text/plain"
    
    def __str__(self):
        return f"{self.key}={self.value[:20]}..." if len(self.value) > 20 else f"{self.key}={self.value}"


@dataclass
class MigrationResult:
    """Results of migration process"""
    success: bool = True
    created_secrets: List[str] = field(default_factory=list)
    created_configs: List[str] = field(default_factory=list)
    failed_secrets: List[Tuple[str, str]] = field(default_factory=list)  # (key, error)
    failed_configs: List[Tuple[str, str]] = field(default_factory=list)  # (key, error)
    skipped_configs: List[str] = field(default_factory=list)
    
    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        
        if self.created_secrets:
            print(f"\n[SUCCESS] Created {len(self.created_secrets)} Key Vault secrets:")
            for secret in self.created_secrets:
                print(f"   - {secret}")
        
        if self.created_configs:
            print(f"\n[SUCCESS] Created {len(self.created_configs)} App Configuration settings:")
            for config in self.created_configs:
                print(f"   - {config}")
        
        if self.skipped_configs:
            print(f"\n[SKIP] Skipped {len(self.skipped_configs)} existing configs (already exist):")
            for config in self.skipped_configs:
                print(f"   - {config}")
        
        if self.failed_secrets or self.failed_configs:
            print(f"\n AILURES:")
            for key, error in self.failed_secrets:
                print(f"   - Secret {key}: {error}")
            for key, error in self.failed_configs:
                print(f"   - Config {key}: {error}")
            self.success = alse
        
        if self.success and (self.created_secrets or self.created_configs):
            print("\n[SUCCESS] Migration completed successfully!")
        elif not self.success:
            print("\n Migration completed with errors!")
        else:
            print("\n[SKIP] No changes made (all settings already exist)")
        
        print("=" * 60 + "\n")


class ConfigMigrator:
    """Handles migration of configuration settings"""
    
    # Settings that are sensitive and should go to Key Vault
    SENSITIVE_KEYS = {
        'BINGX_KEY',
        'BINGX_SECRET',
        'TELEGRAM_BOT_TOKEN',
    }
    
    # Strategy variables that can be added for dynamic configuration
    # Phase 1: Critical feature flags (recommended to add immediately)
    PHASE1_STRATEGY_VARS = {
        'STRATEGY_OB_ENABLED': ('true', 'Enable/disable Oversold Bounce strategy'),
        'STRATEGY_STR_ENABLED': ('true', 'Enable/disable Short the Rip strategy'),
        'SIGNAL_BYPASS_ENABLED': ('true', 'Enable signal bypass for extreme RSI conditions'),
        'SIGNAL_BYPASS_RSI_OVERSOLD': ('12', 'RSI threshold for oversold bypass'),
        'SIGNAL_BYPASS_RSI_OVERBOUGHT': ('88', 'RSI threshold for overbought bypass'),
        'ML_REGIME_MIN_CONIDENCE': ('0.6', 'Minimum confidence for regime prediction'),
        'ML_RL_PPO_ENABLED': ('true', 'Enable PPO reinforcement learning agent'),
        'ML_RL_TRAINING_MODE': ('false', 'Enable training mode for RL agent'),
    }
    
    # Phase 2: Operational parameters (recommended for next phase)
    PHASE2_STRATEGY_VARS = {
        'ADAPTIVE_STRATEGIES_ENABLED': ('true', 'Enable adaptive strategy adjustments'),
        'ADAPTIVE_MONITORING_ENABLED': ('true', 'Enable adaptive performance monitoring'),
        'ADAPTIVE_MIN_VOLATILITY': ('0.02', 'Minimum volatility for adjustments'),
        'ADAPTIVE_MAX_POS_MULT': ('2.0', 'Maximum position multiplier'),
        'ADAPTIVE_MIN_POS_MULT': ('0.5', 'Minimum position multiplier'),
        'ML_LSTM_HIDDEN': ('64', '️  CRITICAL: LSTM hidden size (must match trained models)'),
        'ML_LSTM_LAYERS': ('2', '️  CRITICAL: LSTM number of layers (must match trained models)'),
        'ML_LSTM_DROPOUT': ('0.6', '️  CRITICAL: LSTM dropout rate (must match trained models)'),
        'ML_RL_HOLD_CONIDENCE_THRESHOLD': ('0.60', 'RL confidence threshold for holding positions'),
        'OB_RSI_MAX': ('45', 'Oversold Bounce RSI entry threshold'),
        'STR_RSI_MIN': ('55', 'Short the Rip RSI entry threshold'),
        'OB_MIN_RR_RATIO': ('1.5', 'Oversold Bounce minimum risk/reward ratio'),
        'STR_MIN_RR_RATIO': ('1.5', 'Short the Rip minimum risk/reward ratio'),
    }
    
    # Phase 3: Technical parameters (for advanced users)
    PHASE3_STRATEGY_VARS = {
        'RSI_BASE_OB': ('32', 'Adaptive RSI base for Oversold Bounce'),
        'RSI_RANGE_OB': ('8', 'Adaptive RSI range for Oversold Bounce'),
        'RSI_BASE_STR': ('68', 'Adaptive RSI base for Short the Rip'),
        'RSI_RANGE_STR': ('8', 'Adaptive RSI range for Short the Rip'),
        'ML_EATURE_SIZE': ('42', '️  CRITICAL: eature dimension (must match models)'),
        'ML_EAT_RSI_PERIOD': ('14', 'RSI indicator period'),
        'ML_EAT_ATR_PERIOD': ('14', 'ATR indicator period'),
        'ML_EAT_MACD_AST': ('12', 'MACD fast period'),
        'ML_EAT_MACD_SLOW': ('26', 'MACD slow period'),
        'ML_EAT_BB_PERIOD': ('20', 'Bollinger Bands period'),
        'ML_RL_EPSILON_INERENCE': ('0.01', 'Exploration rate during inference'),
        'ML_RL_LEARNING_RATE': ('0.00003', 'RL training learning rate'),
        'ML_RL_GAMMA': ('0.95', 'RL discount factor'),
        'OB_TP_ATR_MULT': ('2.5', 'Oversold Bounce take profit ATR multiplier'),
        'OB_SL_ATR_MULT': ('1.2', 'Oversold Bounce stop loss ATR multiplier'),
        'STR_TP_ATR_MULT': ('3.0', 'Short the Rip take profit ATR multiplier'),
        'STR_SL_ATR_MULT': ('1.5', 'Short the Rip stop loss ATR multiplier'),
    }
    
    def __init__(
        self,
        env_file: str,
        app_config_name: str,
        app_config_rg: str,
        keyvault_name: str,
        keyvault_rg: str,
        prefix: str = "BearishAlphaBot/",
        label: str = "production",
        dry_run: bool = alse,
        config_yaml: Optional[str] = None,
        phase: int = 0,
    ):
        self.env_file = env_file
        self.app_config_name = app_config_name
        self.keyvault_name = keyvault_name
        self.prefix = prefix
        self.label = label
        self.dry_run = dry_run
        self.config_yaml = config_yaml
        self.phase = phase  # 0=none, 1=phase1, 2=phase1+2, 3=all phases
        
        self.credential = DefaultAzureCredential()
        
        # Initialize Azure clients
        self.appconfig_client = self._init_appconfig_client(app_config_name, app_config_rg)
        self.keyvault_client = self._init_keyvault_client(keyvault_name, keyvault_rg)
        
        self.settings: Dict[str, Setting] = {}
    
    def _init_appconfig_client(self, name: str, rg: str) -> AzureAppConfigurationClient:
        """Initialize App Configuration client"""
        # Get endpoint from Azure
        from azure.mgmt.appconfiguration import AppConfigurationManagementClient
        
        mgmt_client = AppConfigurationManagementClient(
            self.credential,
            self._get_subscription_id()
        )
        config_store = mgmt_client.configuration_stores.get(rg, name)
        endpoint = config_store.endpoint
        
        print(f"[OK] Connected to App Configuration: {endpoint}")
        return AzureAppConfigurationClient(base_url=endpoint, credential=self.credential)
    
    def _init_keyvault_client(self, name: str, rg: str) -> SecretClient:
        """Initialize Key Vault client"""
        vault_url = f"https://{name}.vault.azure.net/"
        print(f"[OK] Connected to Key Vault: {vault_url}")
        return SecretClient(vault_url=vault_url, credential=self.credential)
    
    def _get_subscription_id(self) -> str:
        """Get current subscription ID"""
        from azure.mgmt.subscription import SubscriptionClient
        
        client = SubscriptionClient(self.credential)
        subscription = next(client.subscriptions.list())
        return subscription.subscription_id
    
    def load_env_file(self) -> bool:
        """Load and parse .env file"""
        if not os.path.exists(self.env_file):
            print(f" ile not found: {self.env_file}")
            return alse
        
        print(f"\n Loading {self.env_file}...")
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' not in line:
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    # Determine setting type
                    setting_type = SettingType.SENSITIVE if key in self.SENSITIVE_KEYS else SettingType.CONIG
                    
                    self.settings[key] = Setting(
                        key=key,
                        value=value,
                        type=setting_type,
                        label=self.label,
                    )
            
            print(f"[OK] Loaded {len(self.settings)} settings:")
            print(f"  - Sensitive: {sum(1 for s in self.settings.values() if s.type == SettingType.SENSITIVE)}")
            print(f"  - Config: {sum(1 for s in self.settings.values() if s.type == SettingType.CONIG)}")
            return True
        
        except Exception as e:
            print(f" Error reading file: {e}")
            return alse
    
    def load_strategy_variables(self) -> bool:
        """Load strategy variables based on phase selection"""
        if self.phase == 0:
            print("\n(No strategy variables to load - phase=0)")
            return True
        
        print(f"\n Loading strategy variables (Phase {self.phase})...")
        
        # Select variables based on phase
        strategy_vars = {}
        if self.phase >= 1:
            strategy_vars.update(self.PHASE1_STRATEGY_VARS)
        if self.phase >= 2:
            strategy_vars.update(self.PHASE2_STRATEGY_VARS)
        if self.phase >= 3:
            strategy_vars.update(self.PHASE3_STRATEGY_VARS)
        
        added_count = 0
        for key, (default_value, description) in strategy_vars.items():
            if key not in self.settings:  # Don't override if already in env file
                self.settings[key] = Setting(
                    key=key,
                    value=default_value,
                    type=SettingType.CONIG,
                    label=self.label,
                )
                added_count += 1
                
                # Show warning for critical parameters
                if '️' in description:
                    print(f"  ️  {key:40} = {default_value:15} ({description.replace('️  ', '')})")
                else:
                    print(f"  [OK] {key:40} = {default_value:15}")
        
        print(f"[OK] Loaded {added_count} strategy variables")
        return True
    
    def load_from_yaml(self) -> bool:
        """Load strategy variables from config.example.yaml"""
        if not self.config_yaml or not os.path.exists(self.config_yaml):
            if self.config_yaml:
                print(f"\n️  YAML file not found: {self.config_yaml}")
            return True
        
        print(f"\n Extracting strategy variables from {self.config_yaml}...")
        
        try:
            import yaml
            
            with open(self.config_yaml, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                print("(YAML file is empty)")
                return True
            
            # Extract specific sections
            extracted_count = 0
            
            # Adaptive strategies
            if 'adaptive_strategies' in config and isinstance(config['adaptive_strategies'], dict):
                adaptive = config['adaptive_strategies']
                if 'enable' in adaptive:
                    self.settings['ADAPTIVE_STRATEGIES_ENABLED'] = Setting(
                        key='ADAPTIVE_STRATEGIES_ENABLED',
                        value=str(adaptive['enable']).lower(),
                        type=SettingType.CONIG,
                        label=self.label,
                    )
                    extracted_count += 1
            
            # Signals section
            if 'signals' in config and isinstance(config['signals'], dict):
                signals = config['signals']
                
                # Extreme condition bypass
                if 'extreme_condition_bypass' in signals and isinstance(signals['extreme_condition_bypass'], dict):
                    bypass = signals['extreme_condition_bypass']
                    if 'enabled' in bypass:
                        self.settings['SIGNAL_BYPASS_ENABLED'] = Setting(
                            key='SIGNAL_BYPASS_ENABLED',
                            value=str(bypass['enabled']).lower(),
                            type=SettingType.CONIG,
                            label=self.label,
                        )
                        extracted_count += 1
                
                # Oversold bounce strategy
                if 'oversold_bounce' in signals and isinstance(signals['oversold_bounce'], dict):
                    ob = signals['oversold_bounce']
                    if 'enabled' in ob:
                        self.settings['STRATEGY_OB_ENABLED'] = Setting(
                            key='STRATEGY_OB_ENABLED',
                            value=str(ob['enabled']).lower(),
                            type=SettingType.CONIG,
                            label=self.label,
                        )
                        extracted_count += 1
                
                # Short the rip strategy
                if 'short_the_rip' in signals and isinstance(signals['short_the_rip'], dict):
                    str_strat = signals['short_the_rip']
                    if 'enabled' in str_strat:
                        self.settings['STRATEGY_STR_ENABLED'] = Setting(
                            key='STRATEGY_STR_ENABLED',
                            value=str(str_strat['enabled']).lower(),
                            type=SettingType.CONIG,
                            label=self.label,
                        )
                        extracted_count += 1
            
            # ML section
            if 'ml' in config and isinstance(config['ml'], dict):
                ml = config['ml']
                
                if 'enabled' in ml:
                    self.settings['ML_ENABLED'] = Setting(
                        key='ML_ENABLED',
                        value=str(ml['enabled']).lower(),
                        type=SettingType.CONIG,
                        label=self.label,
                    )
                    extracted_count += 1
                
                # Regime prediction
                if 'regime_prediction' in ml and isinstance(ml['regime_prediction'], dict):
                    regime = ml['regime_prediction']
                    if 'enabled' in regime:
                        self.settings['ML_REGIME_ENABLED'] = Setting(
                            key='ML_REGIME_ENABLED',
                            value=str(regime['enabled']).lower(),
                            type=SettingType.CONIG,
                            label=self.label,
                        )
                        extracted_count += 1
                
                # Reinforcement learning
                if 'reinforcement_learning' in ml and isinstance(ml['reinforcement_learning'], dict):
                    rl = ml['reinforcement_learning']
                    if 'ppo' in rl and isinstance(rl['ppo'], dict):
                        ppo = rl['ppo']
                        if 'enabled' in ppo:
                            self.settings['ML_RL_PPO_ENABLED'] = Setting(
                                key='ML_RL_PPO_ENABLED',
                                value=str(ppo['enabled']).lower(),
                                type=SettingType.CONIG,
                                label=self.label,
                            )
                            extracted_count += 1
            
            print(f"[OK] Extracted {extracted_count} variables from YAML")
            return True
        
        except ImportError:
            print("️  PyYAML not installed - skipping YAML extraction")
            return True
        except Exception as e:
            print(f" Error reading YAML: {e}")
            return alse
    
    def migrate(self) -> MigrationResult:
        """Execute migration"""
        result = MigrationResult()
        
        if self.dry_run:
            print("\n DRY RUN MODE - No changes will be made")
            self._dry_run_preview(result)
        else:
            print("\n[CONIG]  MIGRATION IN PROGRESS...")
            print(f"   Prefix: {self.prefix}")
            print(f"   Label: {self.label}")
            self._migrate_secrets(result)
            self._migrate_configs(result)
        
        result.print_summary()
        return result
    
    def _dry_run_preview(self, result: MigrationResult):
        """Preview what would be migrated"""
        print("\n--- SENSITIVE SETTINGS (-> Key Vault) ---")
        for setting in self.settings.values():
            if setting.type == SettingType.SENSITIVE:
                secret_name = self._normalize_secret_name(setting.key)
                print(f"  {secret_name:40} = {setting.value[:30]}...")
                result.created_secrets.append(secret_name)
        
        print("\n--- CONIG SETTINGS (-> App Configuration) ---")
        for setting in self.settings.values():
            if setting.type == SettingType.CONIG:
                app_config_key = f"{self.prefix}{setting.key}"
                print(f"  {app_config_key:50} = {setting.value[:40]}")
                result.created_configs.append(app_config_key)
    
    def _migrate_secrets(self, result: MigrationResult):
        """Migrate sensitive values to Key Vault"""
        sensitive_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.SENSITIVE}
        
        if not sensitive_settings:
            print("\n(No sensitive settings to migrate)")
            return
        
        print(f"\n Migrating {len(sensitive_settings)} sensitive settings to Key Vault...")
        
        for key, setting in sensitive_settings.items():
            secret_name = self._normalize_secret_name(key)
            
            try:
                # Check if secret already exists
                try:
                    self.keyvault_client.get_secret(secret_name)
                    print(f"  [SKIP] {secret_name:35} (already exists, skipping)")
                    result.skipped_configs.append(secret_name)
                    continue
                except ResourceNotoundError:
                    pass
                
                # Create secret
                self.keyvault_client.set_secret(secret_name, setting.value)
                print(f"  [OK] {secret_name:35} (created)")
                result.created_secrets.append(secret_name)
            
            except Exception as e:
                error_msg = str(e)
                print(f"   {secret_name:35} (error: {error_msg})")
                result.failed_secrets.append((secret_name, error_msg))
    
    def _migrate_configs(self, result: MigrationResult):
        """Migrate configuration values to App Configuration"""
        config_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.CONIG}
        
        if not config_settings:
            print("\n(No config settings to migrate)")
            return
        
        print(f"\n[CONIG]  Migrating {len(config_settings)} config settings to App Configuration...")
        
        for key, setting in config_settings.items():
            app_config_key = f"{self.prefix}{key}"
            
            try:
                # Check if setting already exists
                try:
                    self.appconfig_client.get_configuration_setting(key=app_config_key, label=self.label)
                    print(f"  [SKIP] {app_config_key:50} (already exists, skipping)")
                    result.skipped_configs.append(app_config_key)
                    continue
                except ResourceNotoundError:
                    pass
                
                # Create setting
                config_setting = ConfigurationSetting(
                    key=app_config_key,
                    value=setting.value,
                    label=self.label,
                    content_type=setting.content_type,
                    tags={
                        'service': 'bearish-bot',
                        'migrated': 'true',
                        'environment': self.label,
                    }
                )
                self.appconfig_client.set_configuration_setting(config_setting)
                print(f"  [OK] {app_config_key:50}")
                result.created_configs.append(app_config_key)
            
            except Exception as e:
                error_msg = str(e)
                print(f"   {app_config_key:50} (error: {error_msg})")
                result.failed_configs.append((app_config_key, error_msg))
    
    def create_keyvault_references(self, result: MigrationResult):
        """Create App Configuration entries with Key Vault references"""
        if not result.created_secrets:
            print("\n(No Key Vault secrets to reference)")
            return
        
        print(f"\n Creating Key Vault references in App Configuration...")
        
        sensitive_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.SENSITIVE}
        
        for key, setting in sensitive_settings.items():
            secret_name = self._normalize_secret_name(key)
            app_config_key = f"{self.prefix}{key}"
            
            try:
                # Get secret version to create exact reference
                secret = self.keyvault_client.get_secret(secret_name)
                secret_uri = secret.id
                
                # Create reference value
                reference_value = f"@Microsoft.KeyVault(SecretUri={secret_uri})"
                
                # Check if reference already exists
                try:
                    self.appconfig_client.get_configuration_setting(key=app_config_key, label=self.label)
                    print(f"  [SKIP] {app_config_key:50} (reference already exists, skipping)")
                    continue
                except ResourceNotoundError:
                    pass
                
                # Create reference setting
                ref_setting = ConfigurationSetting(
                    key=app_config_key,
                    value=reference_value,
                    label=self.label,
                    content_type='application/vnd.microsoft.appconfig.keyvaultref+json;charset=utf-8',
                    tags={
                        'service': 'bearish-bot',
                        'secret-reference': 'true',
                        'environment': self.label,
                    }
                )
                self.appconfig_client.set_configuration_setting(ref_setting)
                print(f"  [OK] {app_config_key:50} -> {secret_name}")
            
            except Exception as e:
                print(f"   {app_config_key:50} (error: {str(e)})")
    
    @staticmethod
    def _normalize_secret_name(key: str) -> str:
        """Convert setting key to valid Key Vault secret name (lowercase, hyphens)"""
        # Key Vault secret names: alphanumeric + hyphen, must start with letter/digit
        return key.lower().replace('_', '-')


def main():
    parser = argparse.ArgumentParser(
        description='Migrate bearish-bot.env to Azure App Configuration and Key Vault'
    )
    parser.add_argument('--env-file', required=True, help='Path to bearish-bot.env file')
    parser.add_argument('--app-config-name', required=True, help='App Configuration store name')
    parser.add_argument('--app-config-rg', required=True, help='App Configuration resource group')
    parser.add_argument('--keyvault-name', required=True, help='Key Vault name')
    parser.add_argument('--keyvault-rg', required=True, help='Key Vault resource group')
    parser.add_argument('--prefix', default='BearishAlphaBot/', help='Key prefix in App Configuration')
    parser.add_argument('--label', default='production', help='Label for settings')
    parser.add_argument('--dry-run', action='store_true', help='Preview without making changes')
    parser.add_argument('--config-yaml', default=None, help='Path to config.example.yaml for strategy variables')
    parser.add_argument('--phase', type=int, default=0, choices=[0,1,2,3], 
                       help='Strategy variables phase: 0=none, 1=critical flags, 2=operational, 3=all')
    
    args = parser.parse_args()
    
    print(" Azure App Configuration Migration Script")
    print("=" * 60)
    
    migrator = ConfigMigrator(
        env_file=args.env_file,
        app_config_name=args.app_config_name,
        app_config_rg=args.app_config_rg,
        keyvault_name=args.keyvault_name,
        keyvault_rg=args.keyvault_rg,
        prefix=args.prefix,
        label=args.label,
        dry_run=args.dry_run,
        config_yaml=args.config_yaml,
        phase=args.phase,
    )
    
    # Load settings
    if not migrator.load_env_file():
        return 1
    
    # Load strategy variables (either from defaults or from YAML)
    if args.config_yaml:
        if not migrator.load_from_yaml():
            return 1
    else:
        if not migrator.load_strategy_variables():
            return 1
    
    # Migrate
    result = migrator.migrate()
    
    # Create Key Vault references (only in sensitive settings)
    if not args.dry_run and result.created_secrets:
        migrator.create_keyvault_references(result)
    
    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
