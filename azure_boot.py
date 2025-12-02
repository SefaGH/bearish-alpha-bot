"""Azure Boot Utilities

Utility functions for Azure environment setup.
Used by vm_boot.py for container initialization.

Functions:
    - setup_environment(): Configure PYTHONPATH for project
    - ensure_directories(): Create required directories
    - setup_default_manifest(): Create GEMMA-2.0.0 manifest (CRITICAL for ML)
    - setup_ml_environment(): Configure ML environment variables
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
import json
from typing import Optional

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

def setup_default_manifest(manifest_path: Optional[str] = None):
    """Set up proper GEMMA-2.0.0 manifest for Azure deployment.
    
    Args:
        manifest_path: Optional custom path for manifest. Defaults to artifacts/gemma/final/manifest.json
        
    Note:
        This manifest is CRITICAL - ML system (Gemma adapter) reads this to determine feature count.
        Missing or invalid manifest will cause ML predictions to fail.
    """
    if manifest_path is None:
        manifest_path = os.getenv('GEMMA_MANIFEST_PATH', 'artifacts/gemma/final/manifest.json')
    
    manifest_path = Path(manifest_path)
    
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
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    log.info(f"✅ {script} completed successfully")
                else:
                    log.warning(f"⚠️ {script} failed (non-critical): {result.stderr}")
            except subprocess.TimeoutExpired:
                log.error(f"❌ {script} timed out after 120 seconds")
                raise RuntimeError(f"Critical setup script timed out: {script}")
            except Exception as e:
                log.warning(f"⚠️ Failed to run {script}: {e}")
        else:
            log.warning(f"⚠️ Setup script not found: {script}")

    # Verify GEMMA manifest exists (mirror of CI pre-flight check)
    manifest_path_str = os.getenv('GEMMA_MANIFEST_PATH', 'artifacts/gemma/final/manifest.json')
    gemma_manifest = Path(manifest_path_str)
    if not gemma_manifest.exists():
        log.error("❌ GEMMA manifest not found at %s", gemma_manifest)
        raise FileNotFoundError(f"Critical: GEMMA manifest missing at {gemma_manifest}")
    else:
        log.info("✅ GEMMA manifest found at %s", gemma_manifest)

    # Verify PPO model artifact exists (mirror of CI pre-flight check)
    ppo_path = Path('artifacts/ppo/ppo_trading_agent.zip')
    if not ppo_path.exists():
        log.warning("⚠️ PPO model not found at %s (optional)", ppo_path)
    else:
        log.info("✅ PPO model found at %s", ppo_path)