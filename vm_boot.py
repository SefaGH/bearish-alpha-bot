import os
import sys
import subprocess
import logging
from pathlib import Path

try:
    from azure_boot import setup_environment, ensure_directories, setup_default_manifest, setup_ml_environment
except ImportError as e:
    logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("vm_boot")
    log.error("Failed to import azure_boot module: %s", e)
    log.error("This container may be missing required Azure setup scripts.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vm_boot")


def build_mode_args() -> list[str]:
    """Build CLI args for live_trading_launcher based on environment variables.

    Supported envs:
      - TRADING_MODE: 'paper' (default) or 'live'
      - DEBUG_MODE: 'true' / 'false'
      - TRADING_DURATION: seconds, if set
    """

    mode_args: list[str] = []

    trading_mode = os.environ.get("TRADING_MODE", "paper").lower()
    debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"
    duration = os.environ.get("TRADING_DURATION")

    if trading_mode != "live":
        mode_args.append("--paper")
        log.info("Running in PAPER mode (default for VM)")
    else:
        # live_trading_launcher.py defaults to paper unless --live is explicitly passed.
        mode_args.append("--live")
        log.info("Running in LIVE mode (TRADING_MODE=live)")

    if debug_mode:
        mode_args.append("--debug")
        log.info("Debug mode enabled (DEBUG_MODE=true)")

    if duration:
        try:
            # Validate duration is a valid integer
            duration_int = int(duration)
            if duration_int <= 0:
                log.warning("TRADING_DURATION must be positive, got %s. Using default.", duration)
            else:
                mode_args.extend(["--duration", str(duration_int)])
                log.info("Trading duration set via env TRADING_DURATION=%s seconds", duration_int)
        except ValueError:
            log.warning("Invalid TRADING_DURATION value '%s' (must be integer). Ignoring.", duration)
    else:
        log.info("Trading duration not set (TRADING_DURATION unset) - launcher controls loop length")

    return mode_args


def main() -> int:
    log.info("========================================")
    log.info("Bearish Alpha Bot - VM Boot")
    log.info("========================================")
    log.info("Python version: %s", sys.version.replace("\n", " "))
    log.info("Working directory: %s", os.getcwd())

    # Verify critical environment variables are loaded
    required_env_vars = [
        'BINGX_KEY',
        'BINGX_SECRET',
        'CAPITAL_USDT',
        'TRADING_MODE',
        'EXCHANGES',
    ]
    
    optional_env_vars = [
        'AZURE_APPCONFIG_ENDPOINT',
        'AZURE_APPCONFIG_LABEL',
        'TELEGRAM_BOT_TOKEN',
    ]
    
    missing_required = [var for var in required_env_vars if not os.getenv(var)]
    if missing_required:
        log.error("❌ CRITICAL: Missing required environment variables: %s", ', '.join(missing_required))
        log.error("   Ensure --env-file is passed to docker run or env vars are set manually")
        return 1
    
    log.info("✅ All required environment variables present")
    
    # Check optional App Configuration env vars
    appconfig_vars = [var for var in optional_env_vars if var.startswith('AZURE_APPCONFIG')]
    if all(os.getenv(var) for var in appconfig_vars):
        log.info("✅ Azure App Configuration environment variables configured")
    else:
        log.warning("⚠️ App Configuration env vars missing (will fallback to YAML config)")

    # 1) Ortam ve dizinleri hazırla (mevcut azure_boot yardımcılarını tekrar kullanıyoruz)
    setup_environment()
    ensure_directories()
    setup_default_manifest()
    setup_ml_environment()

    # 2) Çalıştırma argümanlarını env'den üret
    mode_args = build_mode_args()

    cmd = [sys.executable, "scripts/live_trading_launcher.py", *mode_args]

    log.info("========================================")
    log.info("Starting Bearish Alpha Bot (VM)")
    log.info("Command: %s", " ".join(cmd))
    log.info("TRADING_MODE=%s", os.environ.get("TRADING_MODE", "paper"))
    log.info("DEBUG_MODE=%s", os.environ.get("DEBUG_MODE", "false"))
    log.info("TRADING_DURATION=%s", os.environ.get("TRADING_DURATION", "<unset>"))
    log.info("EXCHANGES=%s", os.environ.get("EXCHANGES", "bingx"))
    log.info("========================================")

    try:
        exit_code = subprocess.call(cmd)
    except FileNotFoundError as e:
        log.error("Failed to execute launcher: %s", e)
        log.error("Launcher script not found: %s", cmd[1] if len(cmd) > 1 else "unknown")
        return 127  # Command not found
    except Exception as e:
        log.error("Unexpected error during launcher execution: %s", e)
        log.error("Command was: %s", " ".join(cmd))
        return 1  # General error

    log.info("========================================")
    if exit_code == 0:
        log.info("✅ Bearish Alpha Bot completed successfully (exit code 0)")
    else:
        log.warning("⚠️ Bearish Alpha Bot exited with non-zero code: %s", exit_code)
    log.info("(Analysis is based on logs; use diagnostics/log_analyzer_auto_plus.py or helper scripts after stop, regardless of reason)")
    log.info("========================================")

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
