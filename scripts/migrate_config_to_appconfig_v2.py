#!/usr/bin/env python3
"""
Azure App Configuration Migration Script

Migrates bearish-bot.env settings to Azure App Configuration and Azure Key Vault.
Supports adding strategy variables from config.example.yaml.

Sensitive data (credentials) -> Key Vault secrets
Non-sensitive data (config) -> App Configuration direct values

Usage:
    python scripts/migrate_config_to_appconfig_v2.py \
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
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError


class SettingType(Enum):
    """Setting classification"""
    SENSITIVE = "sensitive"
    CONFIG = "config"


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
    failed_secrets: List[Tuple[str, str]] = field(default_factory=list)
    failed_configs: List[Tuple[str, str]] = field(default_factory=list)
    skipped_configs: List[str] = field(default_factory=list)
    
    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 70)
        print("MIGRATION SUMMARY")
        print("=" * 70)
        
        if self.created_secrets:
            print(f"\n[SUCCESS] Created {len(self.created_secrets)} Key Vault secrets:")
            for secret in self.created_secrets:
                print(f"   - {secret}")
        
        if self.created_configs:
            print(f"\n[SUCCESS] Created {len(self.created_configs)} App Configuration settings:")
            for config in self.created_configs:
                print(f"   - {config}")
        
        if self.skipped_configs:
            print(f"\n[SKIP] Skipped {len(self.skipped_configs)} existing configs:")
            for config in self.skipped_configs:
                print(f"   - {config}")
        
        if self.failed_secrets or self.failed_configs:
            print(f"\n[ERROR] FAILURES:")
            for key, error in self.failed_secrets:
                print(f"   - Secret {key}: {error}")
            for key, error in self.failed_configs:
                print(f"   - Config {key}: {error}")
            self.success = False
        
        if self.success and (self.created_secrets or self.created_configs):
            print("\n[SUCCESS] Migration completed successfully!")
        elif not self.success:
            print("\n[ERROR] Migration completed with errors!")
        else:
            print("\n[SKIP] No changes made (all settings already exist)")
        
        print("=" * 70 + "\n")


class ConfigMigrator:
    """Handles migration of configuration settings"""
    
    SENSITIVE_KEYS = {
        'BINGX_KEY',
        'BINGX_SECRET',
        'TELEGRAM_BOT_TOKEN',
    }
    
    # Phase 1: Critical feature flags
    PHASE1_STRATEGY_VARS = {
        'STRATEGY_OB_ENABLED': ('true', 'Enable/disable Oversold Bounce strategy'),
        'STRATEGY_STR_ENABLED': ('true', 'Enable/disable Short the Rip strategy'),
        'SIGNAL_BYPASS_ENABLED': ('true', 'Enable signal bypass for extreme RSI'),
        'SIGNAL_BYPASS_RSI_OVERSOLD': ('12', 'RSI threshold for oversold bypass'),
        'SIGNAL_BYPASS_RSI_OVERBOUGHT': ('88', 'RSI threshold for overbought bypass'),
        'ML_REGIME_MIN_CONFIDENCE': ('0.6', 'Minimum confidence for regime prediction'),
        'ML_RL_PPO_ENABLED': ('true', 'Enable PPO reinforcement learning agent'),
        'ML_RL_TRAINING_MODE': ('false', 'Enable training mode for RL agent'),
    }
    
    # Phase 2: Operational parameters
    PHASE2_STRATEGY_VARS = {
        'ADAPTIVE_STRATEGIES_ENABLED': ('true', 'Enable adaptive strategy adjustments'),
        'ADAPTIVE_MONITORING_ENABLED': ('true', 'Enable adaptive performance monitoring'),
        'ADAPTIVE_MIN_VOLATILITY': ('0.02', 'Minimum volatility for adjustments'),
        'ADAPTIVE_MAX_POS_MULT': ('2.0', 'Maximum position multiplier'),
        'ADAPTIVE_MIN_POS_MULT': ('0.5', 'Minimum position multiplier'),
        'ML_LSTM_HIDDEN': ('64', '[CRITICAL] LSTM hidden size - must match models'),
        'ML_LSTM_LAYERS': ('2', '[CRITICAL] LSTM layers - must match models'),
        'ML_LSTM_DROPOUT': ('0.6', '[CRITICAL] LSTM dropout - must match models'),
        'ML_RL_HOLD_CONFIDENCE_THRESHOLD': ('0.60', 'RL confidence threshold'),
        'OB_RSI_MAX': ('45', 'Oversold Bounce RSI entry threshold'),
        'STR_RSI_MIN': ('55', 'Short the Rip RSI entry threshold'),
        'OB_MIN_RR_RATIO': ('1.5', 'Oversold Bounce risk/reward ratio'),
        'STR_MIN_RR_RATIO': ('1.5', 'Short the Rip risk/reward ratio'),
    }
    
    # Phase 3: Technical parameters
    PHASE3_STRATEGY_VARS = {
        'RSI_BASE_OB': ('32', 'Adaptive RSI base for Oversold Bounce'),
        'RSI_RANGE_OB': ('8', 'Adaptive RSI range for Oversold Bounce'),
        'RSI_BASE_STR': ('68', 'Adaptive RSI base for Short the Rip'),
        'RSI_RANGE_STR': ('8', 'Adaptive RSI range for Short the Rip'),
        'ML_FEATURE_SIZE': ('42', '[CRITICAL] Feature dimension'),
        'ML_FEAT_RSI_PERIOD': ('14', 'RSI indicator period'),
        'ML_FEAT_ATR_PERIOD': ('14', 'ATR indicator period'),
        'ML_FEAT_MACD_FAST': ('12', 'MACD fast period'),
        'ML_FEAT_MACD_SLOW': ('26', 'MACD slow period'),
        'ML_FEAT_BB_PERIOD': ('20', 'Bollinger Bands period'),
        'ML_RL_EPSILON_INFERENCE': ('0.01', 'Exploration rate during inference'),
        'ML_RL_LEARNING_RATE': ('0.00003', 'RL training learning rate'),
        'ML_RL_GAMMA': ('0.95', 'RL discount factor'),
        'OB_TP_ATR_MULT': ('2.5', 'Oversold Bounce take profit multiplier'),
        'OB_SL_ATR_MULT': ('1.2', 'Oversold Bounce stop loss multiplier'),
        'STR_TP_ATR_MULT': ('3.0', 'Short the Rip take profit multiplier'),
        'STR_SL_ATR_MULT': ('1.5', 'Short the Rip stop loss multiplier'),
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
        dry_run: bool = False,
        phase: int = 0,
    ):
        self.env_file = env_file
        self.app_config_name = app_config_name
        self.keyvault_name = keyvault_name
        self.prefix = prefix
        self.label = label
        self.dry_run = dry_run
        self.phase = phase
        
        self.credential = DefaultAzureCredential()
        self.appconfig_client = self._init_appconfig_client(app_config_name, app_config_rg)
        self.keyvault_client = self._init_keyvault_client(keyvault_name, keyvault_rg)
        self.settings: Dict[str, Setting] = {}
    
    def _init_appconfig_client(self, name: str, rg: str) -> AzureAppConfigurationClient:
        """Initialize App Configuration client"""
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
        subscription = next(iter(client.subscriptions.list()))
        return subscription.subscription_id
    
    def load_env_file(self) -> bool:
        """Load and parse .env file"""
        if not os.path.exists(self.env_file):
            print(f"[ERROR] File not found: {self.env_file}")
            return False
        
        print(f"\n[LOAD] Loading {self.env_file}...")
        
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
                    
                    setting_type = SettingType.SENSITIVE if key in self.SENSITIVE_KEYS else SettingType.CONFIG
                    
                    self.settings[key] = Setting(
                        key=key,
                        value=value,
                        type=setting_type,
                        label=self.label,
                    )
            
            print(f"[OK] Loaded {len(self.settings)} settings:")
            print(f"  - Sensitive: {sum(1 for s in self.settings.values() if s.type == SettingType.SENSITIVE)}")
            print(f"  - Config: {sum(1 for s in self.settings.values() if s.type == SettingType.CONFIG)}")
            return True
        
        except Exception as e:
            print(f"[ERROR] Error reading file: {e}")
            return False
    
    def load_strategy_variables(self) -> bool:
        """Load strategy variables based on phase selection"""
        if self.phase == 0:
            print("\n(No strategy variables to load - phase=0)")
            return True
        
        print(f"\n[CHART] Loading strategy variables (Phase {self.phase})...")
        
        strategy_vars = {}
        if self.phase >= 1:
            strategy_vars.update(self.PHASE1_STRATEGY_VARS)
        if self.phase >= 2:
            strategy_vars.update(self.PHASE2_STRATEGY_VARS)
        if self.phase >= 3:
            strategy_vars.update(self.PHASE3_STRATEGY_VARS)
        
        added_count = 0
        for key, (default_value, description) in strategy_vars.items():
            if key not in self.settings:
                self.settings[key] = Setting(
                    key=key,
                    value=default_value,
                    type=SettingType.CONFIG,
                    label=self.label,
                )
                added_count += 1
                
                if '[CRITICAL]' in description:
                    print(f"  [WARN] {key:40} = {default_value:15}")
                else:
                    print(f"  [OK]  {key:40} = {default_value:15}")
        
        print(f"[OK] Loaded {added_count} strategy variables")
        return True
    
    def migrate(self) -> MigrationResult:
        """Execute migration"""
        result = MigrationResult()
        
        if self.dry_run:
            print("\n[DRY] DRY RUN MODE - No changes will be made")
            self._dry_run_preview(result)
        else:
            print("\n[CONFIG] MIGRATION IN PROGRESS...")
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
        
        print("\n--- CONFIG SETTINGS (-> App Configuration) ---")
        for setting in self.settings.values():
            if setting.type == SettingType.CONFIG:
                app_config_key = f"{self.prefix}{setting.key}"
                print(f"  {app_config_key:50} = {setting.value[:40]}")
                result.created_configs.append(app_config_key)
    
    def _migrate_secrets(self, result: MigrationResult):
        """Migrate sensitive values to Key Vault"""
        sensitive_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.SENSITIVE}
        
        if not sensitive_settings:
            print("\n(No sensitive settings to migrate)")
            return
        
        print(f"\n[SECRET] Migrating {len(sensitive_settings)} sensitive settings to Key Vault...")
        
        for key, setting in sensitive_settings.items():
            secret_name = self._normalize_secret_name(key)
            
            try:
                try:
                    self.keyvault_client.get_secret(secret_name)
                    print(f"  [SKIP] {secret_name:35} (already exists)")
                    result.skipped_configs.append(secret_name)
                    continue
                except ResourceNotFoundError:
                    pass
                
                self.keyvault_client.set_secret(secret_name, setting.value)
                print(f"  [OK]  {secret_name:35} (created)")
                result.created_secrets.append(secret_name)
            
            except Exception as e:
                error_msg = str(e)
                print(f"  [ERROR] {secret_name:35} ({error_msg})")
                result.failed_secrets.append((secret_name, error_msg))
    
    def _migrate_configs(self, result: MigrationResult):
        """Migrate configuration values to App Configuration"""
        config_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.CONFIG}
        
        if not config_settings:
            print("\n(No config settings to migrate)")
            return
        
        print(f"\n[CONFIG] Migrating {len(config_settings)} config settings to App Configuration...")
        
        for key, setting in config_settings.items():
            app_config_key = f"{self.prefix}{key}"
            
            try:
                try:
                    self.appconfig_client.get_configuration_setting(key=app_config_key, label=self.label)
                    print(f"  [SKIP] {app_config_key:50} (already exists)")
                    result.skipped_configs.append(app_config_key)
                    continue
                except ResourceNotFoundError:
                    pass
                
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
                print(f"  [OK]  {app_config_key:50}")
                result.created_configs.append(app_config_key)
            
            except Exception as e:
                error_msg = str(e)
                print(f"  [ERROR] {app_config_key:50} ({error_msg})")
                result.failed_configs.append((app_config_key, error_msg))
    
    def create_keyvault_references(self, result: MigrationResult):
        """Create App Configuration entries with Key Vault references"""
        if not result.created_secrets:
            print("\n(No Key Vault secrets to reference)")
            return
        
        print(f"\n[LINK] Creating Key Vault references in App Configuration...")
        
        sensitive_settings = {k: v for k, v in self.settings.items() if v.type == SettingType.SENSITIVE}
        
        for key, setting in sensitive_settings.items():
            secret_name = self._normalize_secret_name(key)
            app_config_key = f"{self.prefix}{key}"
            
            try:
                secret = self.keyvault_client.get_secret(secret_name)
                secret_uri = secret.id
                reference_value = f"@Microsoft.KeyVault(SecretUri={secret_uri})"
                
                try:
                    self.appconfig_client.get_configuration_setting(key=app_config_key, label=self.label)
                    print(f"  [SKIP] {app_config_key:50} (reference already exists)")
                    continue
                except ResourceNotFoundError:
                    pass
                
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
                print(f"  [OK]  {app_config_key:50} -> {secret_name}")
            
            except Exception as e:
                print(f"  [ERROR] {app_config_key:50} ({str(e)})")
    
    @staticmethod
    def _normalize_secret_name(key: str) -> str:
        """Convert setting key to valid Key Vault secret name"""
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
    parser.add_argument('--phase', type=int, default=0, choices=[0,1,2,3], 
                       help='Strategy phase: 0=none, 1=critical, 2=operational, 3=all')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Azure App Configuration Migration Script (with Strategy Variables)")
    print("=" * 70)
    
    migrator = ConfigMigrator(
        env_file=args.env_file,
        app_config_name=args.app_config_name,
        app_config_rg=args.app_config_rg,
        keyvault_name=args.keyvault_name,
        keyvault_rg=args.keyvault_rg,
        prefix=args.prefix,
        label=args.label,
        dry_run=args.dry_run,
        phase=args.phase,
    )
    
    if not migrator.load_env_file():
        return 1
    
    if not migrator.load_strategy_variables():
        return 1
    
    result = migrator.migrate()
    
    if not args.dry_run and result.created_secrets:
        migrator.create_keyvault_references(result)
    
    return 0 if result.success else 1


if __name__ == '__main__':
    sys.exit(main())
