"""
Unified ML Model Training Script

This script trains all enabled ML models (e.g., GEMMA, LSTM) based on the
central project configuration (`config.example.yaml`).

It performs the following steps:
1. Loads the prepared training data from `data/cache`.
2. For each enabled model in the config:
   a. Initializes the appropriate model trainer (e.g., RegimeModelTrainer).
   b. Passes the model-specific configuration to the trainer.
   c. Trains the model using the prepared data.
   d. Saves the trained model and its performance metrics.
"""
import sys
from pathlib import Path
import logging
import numpy as np
import yaml
import json
from datetime import datetime

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.model_trainer import RegimeModelTrainer

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Loads the main YAML configuration file."""
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.example.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Kritik Hata: Konfigürasyon dosyası bulunamadı: {config_path}")
        sys.exit(1)
    return None

def load_training_data(config: dict) -> tuple:
    """Loads the prepared training data from the cache."""
    data_path_str = config.get('ml', {}).get('feature_selection', {}).get('data_path', 'data/cache/BTC-USDT_training_data.npz')
    data_path = Path(data_path_str)
    
    if not data_path.exists():
        logger.error(f"❌ Kritik Hata: Hazırlanmış eğitim verisi bulunamadı: {data_path}")
        logger.error("Lütfen önce 'prepare_training_data.py' script'ini veya ilgili workflow adımını çalıştırın.")
        sys.exit(1)
    
    try:
        logger.info(f"Eğitim verisi yükleniyor: {data_path}")
        data = np.load(data_path)
        X, y = data['X'], data['y']
        logger.info(f"✅ Veri yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik.")
        return X, y
    except Exception as e:
        logger.error(f"Eğitim verisi yüklenirken hata oluştu: {e}", exc_info=True)
        sys.exit(1)

def train_model(model_name: str, model_config: dict, X_train: np.ndarray, y_train: np.ndarray):
    """
    Initializes and trains a single model based on its configuration.
    """
    logger.info("\n" + "="*70)
    logger.info(f"🚀 {model_name.upper()} MODELİ EĞİTİMİ BAŞLATILIYOR 🚀")
    logger.info("="*70)

    # Şu an için tüm modeller RegimeModelTrainer'ı kullanıyor.
    # Gelecekte farklı trainer'lar gerekirse, burada bir koşul eklenebilir.
    # Örnek: if model_config.get('type') == 'price_prediction': trainer = PriceModelTrainer(...)
    trainer = RegimeModelTrainer(config=model_config)
    
    # Modeli eğit
    results = trainer.train_and_evaluate(X_train, y_train, model_type=model_name)
    
    if not results or results.get('status') != 'completed':
        logger.error(f"❌ {model_name.upper()} modeli eğitimi başarısız oldu veya tamamlanamadı.")
        return

    # Başarı metriklerini kaydet
    log_dir = Path(f'logs/final_training/{model_name}')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = log_dir / f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ {model_name.upper()} modelinin metrikleri kaydedildi: {metrics_file}")
    except Exception as e:
        logger.error(f"Metrikler kaydedilirken hata: {e}")

def main():
    """
    Main function to run the entire training pipeline for all enabled models.
    """
    logger.info("="*80)
    logger.info("🤖 BİRLEŞİK MODEL EĞİTİM PİPELINE'I BAŞLATILIYOR 🤖")
    logger.info("="*80)

    # 1. Konfigürasyonu Yükle
    config = load_config()
    ml_config = config.get('ml', {})
    
    # 2. Hazırlanmış Eğitim Verisini Yükle
    X, y = load_training_data(config)
    
    # 3. Aktif Olan Her Bir Model İçin Eğitim Döngüsünü Çalıştır
    models_to_train = {name: conf for name, conf in ml_config.items() if isinstance(conf, dict) and conf.get('enabled', False)}
    
    if not models_to_train:
        logger.warning("⚠️ Konfigürasyonda eğitilecek aktif bir model bulunamadı. İşlem sonlandırılıyor.")
        return

    logger.info(f"Eğitilecek aktif modeller: {', '.join(models_to_train.keys())}")

    for model_name, model_config in models_to_train.items():
        train_model(model_name, model_config, X, y)

    logger.info("\n" + "="*80)
    logger.info("🎉 TÜM AKTİF MODEL EĞİTİMLERİ TAMAMLANDI 🎉")
    logger.info("="*80)

if __name__ == "__main__":
    main()
