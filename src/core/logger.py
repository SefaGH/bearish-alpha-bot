"""
Centralized, Asynchronous, and Unified Logging Configuration for the Bot.
This version is asyncio-safe, merges debug logic, and preserves all original features.
"""
import logging
import logging.handlers
import sys
import os
import queue
from datetime import datetime

# Global değişkenler, listener'ın sadece bir kez başlatılmasını sağlar.
_listener = None
_log_queue = queue.Queue(-1)

def setup_logger(name: str = "bearish_alpha_bot", debug_mode: bool = False, log_to_file: bool = True) -> logging.Logger:
    """
    Sets up a centralized, queue-based logger that is safe for concurrent asyncio tasks.
    It configures the root logger, so it only needs to be called once.
    
    Args:
        name: Name of the logger to return (typically __name__).
        debug_mode: If True, sets log level to DEBUG and adds debug formatting.
        log_to_file: If True, logs to a timestamped file with symlinks.
    """
    global _listener

    log_level_str = 'DEBUG' if debug_mode else os.getenv('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Root logger'ı yapılandır, tüm loglar buradan geçecek.
    root_logger = logging.getLogger()

    # Listener sadece bir kez, ilk setup_logger çağrısında kurulmalı.
    if _listener is None:
        # Mevcut tüm handler'ları temizle, sıfırdan yapılandırıyoruz.
        root_logger.handlers.clear()
        root_logger.setLevel(log_level)
        
        # Gürültücü kütüphaneleri sustur (Hedef 1: Gürültüyü Engelleme)
        logging.getLogger("websockets.client").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

        formatter = logging.Formatter(
            f'%(asctime)s - {"🔍 " if debug_mode else ""}[%(name)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler listesini oluştur
        handlers_to_listen = []
        
        # Konsol Handler'ı
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers_to_listen.append(console_handler)

        # Dosya Handler'ı (Mevcut kodunuzdaki tüm mantık korunarak)
        if log_to_file:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            basename = os.getenv('LOG_FILE_BASENAME', 'live_trading')
            log_file = os.path.join(log_dir, f'{basename}_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            handlers_to_listen.append(file_handler)
            
            # Sembolik linkleri oluştur (Önemli özellik korundu)
            _create_symlinks(log_dir, log_file)
            
            # İlk log mesajını doğrudan yaz (listener başlamadan önce görünsün)
            print(f"File logging enabled: {log_file}")

        # Kuyruk Listener'ını başlat (Hedef 2: Bölünmüş Logları Düzeltme)
        _listener = logging.handlers.QueueListener(_log_queue, *handlers_to_listen, respect_handler_level=True)
        _listener.start()
        
        # Tüm logları kuyruğa yönlendiren ana handler'ı root'a ekle.
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

        if debug_mode:
            root_logger.info("🔍 DEBUG MODE: Enhanced logging enabled.")

    # İstenen isimde bir logger döndür. Bu logger, root'a bağlı olduğu için
    # tüm logları otomatik olarak kuyruğa gönderecektir.
    return logging.getLogger(name)

def _create_symlinks(log_dir: str, log_file: str):
    """Creates symlinks for 'latest' and legacy log file patterns."""
    try:
        abs_log_file = os.path.abspath(log_file)
        latest_name = os.path.join(log_dir, 'live_trading_latest.log')
        
        # 'latest' symlink'ini güncelle
        if os.path.lexists(latest_name): # islink() yerine lexists() daha güvenli
            os.remove(latest_name)
        os.symlink(abs_log_file, latest_name)

        # Geriye uyumlu symlink (opsiyonel ama iyi bir özellik)
        ts_part = os.path.basename(abs_log_file).split('_', 1)[-1]
        legacy_name = os.path.join(log_dir, f'bearish_alpha_bot_{ts_part}')
        if os.path.lexists(legacy_name):
            os.remove(legacy_name)
        os.symlink(abs_log_file, legacy_name)

    except (OSError, AttributeError, ImportError) as e:
        # Symlink oluşturma başarısız olursa (örn: Windows'ta yetki yok), sessizce devam et.
        logging.getLogger(__name__).debug(f"Could not create symlinks: {e}")
