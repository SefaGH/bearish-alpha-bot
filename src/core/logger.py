import io
import logging
import logging.handlers
import os
import queue
import sys
from datetime import datetime

# Global değişkenler, listener'ın sadece bir kez başlatılmasını sağlar.
_listener = None
_log_queue = queue.Queue(-1)
CURRENT_LOG_FILE = None  # Exposed for other modules to know the active log file

def setup_logger(name: str = "bearish_alpha_bot",
                 debug_mode: bool = False,
                 log_to_file: bool = True,
                 log_filename: str = None,
                 level: int = None) -> logging.Logger:
    """
    Sets up a centralized, queue-based logger that is safe for concurrent asyncio tasks.
    It configures the root logger, so it only needs to be called once.
    
    (GÜNCELLENDİ: 'level' ve 'debug_mode' parametrelerini birlikte kabul eder hale getirildi.)

    Args:
        name: Name of the logger to return (typically __name__).
        debug_mode: If True, sets log level to DEBUG.
        log_to_file: If True, logs to a file.
        log_filename: Specific filename for the log.
        level: (YENİ) Explicit log level (e.g., logging.DEBUG). Overrides debug_mode if set.
    """
    global _listener

    # --- YENİ MANTIK: 'level' ve 'debug_mode' uyumluluğu ---
    # Eğer 'level' parametresi verilmişse, onu öncelikli olarak kullan.
    if level is not None:
        log_level = level
    # Eğer 'level' verilmemişse, 'debug_mode' bayrağına göre karar ver.
    else:
        log_level_str = 'DEBUG' if debug_mode else os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    # --- YENİ MANTIK SONU ---

    root_logger = logging.getLogger()

    if _listener is None:
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)
        
        # Suppress noisy third-party library logs
        logging.getLogger("websockets.client").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("ccxt.base.exchange").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        formatter = logging.Formatter(
            f'%(asctime)s - {"DEBUG " if log_level == logging.DEBUG else ""}[%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        handlers_to_listen = []
        
        stream = sys.stdout
        if hasattr(stream, "buffer"):
            stream = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        handlers_to_listen.append(console_handler)

        if log_to_file:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            if log_filename:
                log_file_path = os.path.join(log_dir, log_filename)
                file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                basename = os.getenv('LOG_FILE_BASENAME', 'live_trading')
                log_file_path = os.path.join(log_dir, f'{basename}_{timestamp}.log')
                file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
                _create_symlinks(log_dir, log_file_path)

            global CURRENT_LOG_FILE
            CURRENT_LOG_FILE = log_file_path

            file_handler.setFormatter(formatter)
            handlers_to_listen.append(file_handler)
            
            print(f"File logging enabled: {log_file_path}")

        # --- AZURE APPLICATION INSIGHTS SUPPORT ---
        app_insights_key = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        if app_insights_key:
            try:
                from opencensus.ext.azure.log_exporter import AzureLogHandler
                
                # Extract run_id from filename if available
                run_id = "unknown"
                if CURRENT_LOG_FILE:
                    filename = os.path.basename(CURRENT_LOG_FILE)
                    if filename.startswith("live_trading_") and filename.endswith(".log"):
                        run_id = filename.replace("live_trading_", "").replace(".log", "")
                
                # Callback to add run_id to every log
                def callback_add_run_id(envelope):
                    envelope.data.baseData.properties['run_id'] = run_id
                    return True

                azure_handler = AzureLogHandler(connection_string=app_insights_key)
                azure_handler.add_telemetry_processor(callback_add_run_id)
                azure_handler.setFormatter(formatter)
                handlers_to_listen.append(azure_handler)
                print(f"Azure Application Insights logging enabled (run_id: {run_id})")
            except ImportError:
                print("opencensus-ext-azure not installed. Azure logging disabled.")
            except Exception as e:
                print(f"Failed to setup Azure logging: {e}")
        # ------------------------------------------

        # DIRECT LOGGING (Fix for missing logs in container)
        # We bypass QueueListener to ensure logs are written immediately and not lost on exit.
        for handler in handlers_to_listen:
            root_logger.addHandler(handler)

        if log_level == logging.DEBUG:
            root_logger.info("DEBUG MODE: Enhanced logging enabled.")

    return logging.getLogger(name)

def _create_symlinks(log_dir: str, log_file: str):
    """Creates symlinks for 'latest' and legacy log file patterns."""
    try:
        abs_log_file = os.path.abspath(log_file)
        latest_name = os.path.join(log_dir, 'live_trading_latest.log')
        
        if os.path.lexists(latest_name):
            os.remove(latest_name)
        os.symlink(abs_log_file, latest_name)

        ts_part = os.path.basename(abs_log_file).split('_', 1)[-1]
        legacy_name = os.path.join(log_dir, f'bearish_alpha_bot_{ts_part}')
        if os.path.lexists(legacy_name):
            os.remove(legacy_name)
        os.symlink(abs_log_file, legacy_name)

    except (OSError, AttributeError, ImportError) as e:
        logging.getLogger(__name__).debug(f"Could not create symlinks: {e}")
