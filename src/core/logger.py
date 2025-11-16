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

            file_handler.setFormatter(formatter)
            handlers_to_listen.append(file_handler)
            
            print(f"File logging enabled: {log_file_path}")

        _listener = logging.handlers.QueueListener(_log_queue, *handlers_to_listen, respect_handler_level=True)
        _listener.start()
        
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

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
