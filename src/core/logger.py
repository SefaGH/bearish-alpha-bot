import logging
import logging.handlers
import sys
import os
import queue
from datetime import datetime

# Global değişkenler, listener'ın sadece bir kez başlatılmasını sağlar.
_listener = None
_log_queue = queue.Queue(-1)

def setup_logger(name: str = "bearish_alpha_bot", debug_mode: bool = False, log_to_file: bool = True, log_filename: str = None) -> logging.Logger:
    """
    Sets up a centralized, queue-based logger that is safe for concurrent asyncio tasks.
    It configures the root logger, so it only needs to be called once.
    
    (GÜNCELLENDİ: 'log_filename' parametresini kabul eder hale getirildi.)

    Args:
        name: Name of the logger to return (typically __name__).
        debug_mode: If True, sets log level to DEBUG and adds debug formatting.
        log_to_file: If True, logs to a file.
        log_filename: (YENİ) Specific filename for the log. If None, a timestamped name is generated.
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
        
        # Gürültücü kütüphaneleri sustur
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

        # Dosya Handler'ı
        if log_to_file:
            log_dir = 'logs'
            os.makedirs(log_dir, exist_ok=True)
            
            # --- YENİ MANTIK BAŞLANGICI ---
            # Eğer bir log_filename belirtilmişse onu kullan, belirtilmemişse eskisi gibi zaman damgalı oluştur.
            if log_filename:
                # eğitim betiği gibi özel durumlar için sabit bir isim kullan
                log_file_path = os.path.join(log_dir, log_filename)
                # Bu durumda sembolik link oluşturmaya gerek yok, çünkü dosya adı sabit.
                file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' (append) modu ile üzerine yazmayı engelle
            else:
                # Canlı bot için zaman damgalı dosya adı oluşturma mantığı korunuyor.
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                basename = os.getenv('LOG_FILE_BASENAME', 'live_trading')
                log_file_path = os.path.join(log_dir, f'{basename}_{timestamp}.log')
                file_handler = logging.FileHandler(log_file_path, mode='w')
                
                # Sembolik linkleri sadece zaman damgalı loglar için oluştur.
                _create_symlinks(log_dir, log_file_path)
            # --- YENİ MANTIK SONU ---

            file_handler.setFormatter(formatter)
            handlers_to_listen.append(file_handler)
            
            print(f"File logging enabled: {log_file_path}")

        # Kuyruk Listener'ını başlat
        _listener = logging.handlers.QueueListener(_log_queue, *handlers_to_listen, respect_handler_level=True)
        _listener.start()
        
        # Tüm logları kuyruğa yönlendiren ana handler'ı root'a ekle.
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        root_logger.addHandler(queue_handler)

        if debug_mode:
            root_logger.info("🔍 DEBUG MODE: Enhanced logging enabled.")

    # İstenen isimde bir logger döndür.
    return logging.getLogger(name)

def _create_symlinks(log_dir: str, log_file: str):
    """Creates symlinks for 'latest' and legacy log file patterns."""
    try:
        abs_log_file = os.path.abspath(log_file)
        latest_name = os.path.join(log_dir, 'live_trading_latest.log')
        
        if os.path.lexists(latest_name):
            os.remove(latest_name)
        os.symlink(abs_log_file, latest_name)

        # Geriye uyumlu symlink
        ts_part = os.path.basename(abs_log_file).split('_', 1)[-1]
        legacy_name = os.path.join(log_dir, f'bearish_alpha_bot_{ts_part}')
        if os.path.lexists(legacy_name):
            os.remove(legacy_name)
        os.symlink(abs_log_file, legacy_name)

    except (OSError, AttributeError, ImportError) as e:
        logging.getLogger(__name__).debug(f"Could not create symlinks: {e}")
