import os
import sys
import subprocess
import time
import logging
from pathlib import Path
import json

# Azure SDK (Hata almamak için try-except bloğu)
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    from azure.core.exceptions import AzureError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False

try:
    from keep_alive import start_health_server
    HEALTH_SERVER_AVAILABLE = True
except ImportError:
    HEALTH_SERVER_AVAILABLE = False

# Loglama Ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("azure_boot")

def setup_environment():
    """Set up the Python environment and paths for Azure."""
    current_dir = Path(__file__).parent.absolute()
    
    # Add project directories to Python path
    python_paths = [
        str(current_dir),
        str(current_dir / 'src'),
        str(current_dir / 'scripts')
    ]
    
    for path in python_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set PYTHONPATH environment variable for subprocesses
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = ':'.join(python_paths)
    if current_pythonpath:
        new_pythonpath = f"{new_pythonpath}:{current_pythonpath}"
    
    os.environ['PYTHONPATH'] = new_pythonpath
    log.info(f"✅ PYTHONPATH configured: {new_pythonpath}")

def ensure_directories():
    """Create required directories if they don't exist."""
    directories = [
        'logs',
        'data', 
        'artifacts/gemma/final',
        'artifacts/ppo',
        'features/gemma/selected',
        'data/models/final',
        'data/cache/gemma'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    placeholder_files = [
        'data/state.json',
        'data/day_stats.json', 
        'logs/.placeholder'
    ]
    
    for file_path in placeholder_files:
        path = Path(file_path)
        if not path.exists():
            if file_path.endswith('.json'):
                path.write_text('{}')
            else:
                path.touch()
    
    log.info("✅ Required directories and placeholder files created")

def setup_default_manifest():
    """Set up proper GEMMA-2.0.0 manifest for Azure deployment."""
    manifest_path = Path('artifacts/gemma/final/manifest.json')
    
    if not manifest_path.exists():
        manifest_content = {
            "version": "GEMMA-2.0.0",
            "feature_count": 82,
            "model_type": "gemma",
            "description": "GEMMA-2.0.0 manifest for Azure deployment with 82 features",
            "feature_names_ordered": [
                "open", "high", "low", "close", "volume", "rsi", "rsi_oversold", "rsi_overbought", 
                "macd", "macd_signal", "macd_histogram", "macd_cross", "ema_12", "ema_26", "ema_50", 
                "ema_cross", "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position", "atr", 
                "volatility_parkinson", "volatility_realized", "close_to_high", "close_to_low", 
                "price_change_pct", "volume_sma", "volume_ratio", "obv", "stoch_k", "stoch_d", 
                "williams_r", "cci", "momentum", "roc", "adx", "plus_di", "minus_di", "aroon_up", 
                "aroon_down", "tsi", "ultimate_oscillator", "stochrsi_k", "stochrsi_d", "kama", 
                "tema", "wma", "hma", "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", 
                "ichimoku_senkou_b", "vwap", "pivot_point", "support1", "resistance1", "support2", 
                "resistance2", "fibonacci_23_6", "fibonacci_38_2", "fibonacci_50_0", "fibonacci_61_8", 
                "fibonacci_78_6", "doji", "hammer", "shooting_star", "engulfing", "morning_star", 
                "evening_star", "three_white_soldiers", "three_black_crows", "inside_bar", 
                "outside_bar", "price_ma_20", "price_ma_50", "price_ma_200", "volume_ma_10", 
                "high_low_ratio", "open_close_ratio", "body_size", "upper_shadow", "lower_shadow", 
                "total_shadow", "body_shadow_ratio", "gap_up", "gap_down"
            ],
            "regime_features": [
                "rsi", "macd", "ema_50", "bb_position", "atr", "volatility_realized", "adx", 
                "stoch_k", "williams_r", "cci", "momentum", "roc", "aroon_up", "tsi", 
                "ultimate_oscillator", "stochrsi_k", "kama", "vwap", "price_ma_20", "volume_ratio"
            ],
            "regime_feature_count": 20
        }
        
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_content, indent=2))
        log.info("✅ GEMMA-2.0.0 manifest created for Azure")

def setup_ml_environment():
    """Set up ML environment variables and run setup scripts for Azure."""
    log.info("🧠 Setting up ML environment for Azure...")
    
    # Set GEMMA environment variables
    os.environ['GEMMA_ENABLED'] = 'true'
    os.environ['ML_ACTIVE_BUNDLE'] = 'artifacts/gemma/final'
    log.info("✅ GEMMA environment variables set")
    
    # Run setup scripts if they exist
    setup_scripts = [
        'scripts/setup_gemma_artifacts.sh',
        'scripts/setup_ml_model_links.sh'
    ]
    
    for script in setup_scripts:
        if Path(script).exists():
            try:
                log.info(f"🔧 Running {script}...")
                result = subprocess.run(['bash', script], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    log.info(f"✅ {script} completed successfully")
                else:
                    log.warning(f"⚠️ {script} failed: {result.stderr}")
            except Exception as e:
                log.warning(f"⚠️ Failed to run {script}: {e}")
        else:
            log.warning(f"⚠️ Setup script not found: {script}")

    # Verify GEMMA manifest exists (mirror of CI pre-flight check)
    gemma_manifest = Path('artifacts/gemma/final/manifest.json')
    if not gemma_manifest.exists():
        log.error("❌ GEMMA manifest not found at %s", gemma_manifest)
    else:
        log.info("✅ GEMMA manifest found at %s", gemma_manifest)

    # Verify PPO model artifact exists (mirror of CI pre-flight check)
    ppo_path = Path('artifacts/ppo/ppo_trading_agent.zip')
    if not ppo_path.exists():
        log.error("❌ PPO model not found at %s", ppo_path)
    else:
        log.info("✅ PPO model found at %s", ppo_path)

# 1. Sağlık Sunucusunu Başlat
if HEALTH_SERVER_AVAILABLE:
    log.info("🟢 Azure Health Check Sunucusu Başlatılıyor...")
    start_health_server()
else:
    log.warning("⚠️ Health server not available, continuing without health endpoint")

# 2. Key Vault Entegrasyonu
def load_secrets_from_keyvault(vault_name, secret_names):
    if not AZURE_SDK_AVAILABLE:
        log.warning("Azure SDK yüklü değil, Key Vault atlanıyor.")
        return

    if not vault_name:
        log.info("ℹ️ KEYVAULT_NAME tanımlı değil; .env veya App Settings kullanılacak.")
        return

    kv_uri = f"https://{vault_name}.vault.azure.net"
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=kv_uri, credential=credential)
        log.info(f"🔐 Key Vault Bağlantısı Başarılı: {vault_name}")
    except Exception as e:
        log.error(f"❌ Key Vault bağlantı hatası: {e}")
        return

    for s in secret_names:
        if os.getenv(s) is None:
            try:
                secret = client.get_secret(s)
                os.environ[s] = secret.value
                log.info(f"✅ Secret başarıyla yüklendi: {s}")
            except AzureError as ae:
                log.warning(f"⚠️ Secret okunamadı {s}: {ae}")

KV_NAME = os.getenv("KEYVAULT_NAME")
SECRETS_TO_LOAD = os.getenv("KV_SECRETS", "KUCOIN_API_KEY,KUCOIN_API_SECRET,KUCOIN_API_PASSPHRASE").split(",")

if AZURE_SDK_AVAILABLE:
    load_secrets_from_keyvault(KV_NAME, SECRETS_TO_LOAD)

# 3. Eski Azure App Service retry döngüsü devre dışı
# (VM senaryosunda yeni main() ve vm_boot.py akışı kullanılacak.)

def determine_trading_mode():
    """Determine the trading mode based on environment variables."""
    trading_mode = os.environ.get('TRADING_MODE', 'paper').lower()
    debug_mode = os.environ.get('DEBUG_MODE', 'false').lower() == 'true'
    duration = os.environ.get('TRADING_DURATION')
    
    args = []
    
    # Always default to paper mode in Azure for safety
    if trading_mode != 'live':
        args.append('--paper')
        log.info("📝 Running in paper trading mode (Azure default)")
    else:
        log.info("💰 Running in LIVE trading mode")
    
    if debug_mode:
        args.append('--debug')
        log.info("🔍 Debug mode enabled")
    
    if duration:
        args.extend(['--duration', duration])
        log.info(f"⏱️ Trading duration set to {duration} seconds")
    
    return args

def main():
    """Enhanced main function with proper Azure setup."""
    log.info("========================================")
    log.info("🤖 Bearish Alpha Bot - Azure Boot Enhanced")
    log.info("========================================")
    log.info(f"Python version: {sys.version}")
    log.info(f"Working directory: {os.getcwd()}")
    
    try:
        # Set up environment
        setup_environment()
        ensure_directories()
        setup_default_manifest()
        setup_ml_environment()
        
        # Load secrets from Key Vault if available
        vault_name = os.environ.get("KEYVAULT_NAME")
        if vault_name:
            load_secrets_from_keyvault(vault_name, ["BINGX-KEY", "BINGX-SECRET", "TELEGRAM-BOT-TOKEN"])
        
        # Determine trading arguments
        mode_args = determine_trading_mode()
        
        # Build command
        cmd = [sys.executable, 'scripts/live_trading_launcher.py'] + mode_args
        
        log.info("========================================")
        log.info("🚀 Starting Bearish Alpha Bot")
        log.info("========================================")
        log.info(f"Command: {' '.join(cmd)}")
        log.info("Environment Variables:")
        log.info(f"  TRADING_MODE: {os.environ.get('TRADING_MODE', 'paper')}")
        log.info(f"  DEBUG_MODE: {os.environ.get('DEBUG_MODE', 'false')}")
        log.info(f"  ML_ENABLED: {os.environ.get('ML_ENABLED', 'true')}")
        log.info(f"  EXCHANGES: {os.environ.get('EXCHANGES', 'bingx')}")
        log.info("========================================")
        
        # Execute the trading bot
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
        
    except KeyboardInterrupt:
        log.info("🛑 Received interrupt signal, shutting down gracefully...")
        return 0
    except Exception as e:
        log.error(f"❌ Unexpected error in main: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())