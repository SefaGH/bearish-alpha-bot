"""
Prepare and cache real training data for hyperparameter tuning.
This script is a core part of the data pipeline and mirrors the logic
in train_all_models.py for data preparation.

Usage:
    python scripts/prepare_training_data.py --symbol BTC/USDT
"""
import asyncio
import argparse
import sys
import os
import numpy as np
import yaml  # EKLENDİ: Konfigürasyon okumak için
from pathlib import Path
import logging

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ccxt_client import CcxtClient
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.label_generator import generate_regime_labels

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- YENİ: Konfigürasyon Yükleme Fonksiyonu ---
def load_config():
    """Loads the main YAML configuration file."""
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.example.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Kritik Hata: Konfigürasyon dosyası bulunamadı: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Konfigürasyon dosyası okunurken hata: {e}")
        sys.exit(1)

# Ana konfigürasyonu yükle
config = load_config()
ml_config = config.get('ml', {})

# Konfigürasyondan değerleri al
CANDLE_LIMIT = ml_config.get('data_source', {}).get('candle_limit', 1440)
REGIME_TRAINING_TIMEFRAMES = ml_config.get('data_source', {}).get('timeframes', ['15m', '30m', '1h', '4h', '1d'])

async def fetch_and_process_data(symbol='BTC/USDT',
                                 timeframes=None,
                                 use_feature_selection=True):
    """
    Fetch real market data, apply feature engineering based on config,
    and prepare it for training.
    """
    if timeframes is None:
        timeframes = REGIME_TRAINING_TIMEFRAMES
    
    logger.info("="*70)
    logger.info(f"📊 GERÇEK PİYASA VERİSİ HAZIRLANIYOR: {symbol}")
    logger.info("="*70)
    
    logger.info("Initializing BingX exchange client...")
    exchange_client = CcxtClient('bingx')
    
    # --- DÜZELTİLDİ: FeatureEngineeringPipeline artık konfigürasyon ile çağrılıyor ---
    logger.info("Initializing feature engineering pipeline with config...")
    feature_engine = FeatureEngineeringPipeline(config=config)
    
    all_features, all_labels = [], []
    
    for tf in timeframes:
        logger.info(f"\n--- {tf} verisi işleniyor ---")
        try:
            logger.info(f"  {CANDLE_LIMIT} adet mum verisi çekiliyor...")
            ohlcv_df = await exchange_client.ohlcv(symbol, timeframe=tf, limit=CANDLE_LIMIT)
            
            if ohlcv_df is None or len(ohlcv_df) < 200:
                logger.warning(f"  ⚠️ Yetersiz veri ({len(ohlcv_df)} mum), atlanıyor.")
                continue
            
            logger.info(f"  ✅ {len(ohlcv_df)} adet mum verisi çekildi.")
            
            logger.info("  Özellikler çıkarılıyor...")
            features_df = feature_engine.extract_features(ohlcv_df)
            logger.info(f"  ✅ {features_df.shape[1]} adet özellik çıkarıldı.")
            
            logger.info("  Rejim etiketleri oluşturuluyor...")
            regime_labels = generate_regime_labels(ohlcv_df, **ml_config.get('regime_detection', {}))
            logger.info(f"  ✅ {len(regime_labels)} adet etiket oluşturuldu.")

            logger.info("  Veri temizleniyor ve hizalanıyor...")
            X_prepared, y_prepared = feature_engine.prepare_for_training(
                features_df,
                regime_labels,
                feature_selection_mode='auto'
            )
            
            if len(X_prepared) > 0:
                all_features.append(X_prepared)
                all_labels.append(y_prepared)
                logger.info(f"  ✅ {len(X_prepared)} adet örnek eklendi.")
            else:
                logger.warning("  ⚠️ Hazırlık sonrası hiç örnek kalmadı.")
            
        except Exception as e:
            logger.error(f"  ❌ {tf} işlenirken hata oluştu: {e}", exc_info=True)
            continue
            
    if not all_features:
        raise RuntimeError("Hiçbir zaman diliminden başarılı bir şekilde veri çekilemedi.")
    
    logger.info("\n" + "="*70)
    logger.info("📊 TÜM ZAMAN DİLİMLERİNDEN GELEN VERİLER BİRLEŞTİRİLİYOR")
    logger.info("="*70)
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    logger.info(f"✅ Toplam örnek sayısı: {len(X)}")
    logger.info(f"✅ Orijinal özellik sayısı: {X.shape[1]}")
    
    if use_feature_selection:
        feature_mask_path = Path('data/cache/feature_selection_mask.npy')
        if feature_mask_path.exists():
            try:
                feature_mask = np.load(feature_mask_path)
                if len(feature_mask) != X.shape[1]:
                    logger.warning(f"⚠️ Özellik maskesi boyutu uyuşmuyor! Maske: {len(feature_mask)}, Özellikler: {X.shape[1]}. Seçim atlanıyor.")
                else:
                    removed_count = (~feature_mask).sum()
                    X = X[:, feature_mask]
                    logger.info(f"✅ Özellik seçimi uygulandı. Yeni özellik sayısı: {X.shape[1]} (kaldırılan: {removed_count})")
            except Exception as e:
                logger.warning(f"⚠️ Özellik maskesi yüklenemedi: {e}. Seçim yapılmadan devam ediliyor.")
        else:
            logger.warning(f"⚠️ Özellik seçim maskesi bulunamadı: {feature_mask_path}. Tüm özelliklerle devam ediliyor.")
    else:
        logger.info("⚠️ Özellik seçimi atlandı (--no-feature-selection).")
        
    logger.info(f"✅ Nihai özellik sayısı: {X.shape[1]}")
    logger.info("✅ Etiket dağılımı:")
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: 'Bullish', 1: 'Neutral', 2: 'Bearish'}
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        logger.info(f"     {label_names.get(int(label), f'Class_{label}')}: {count} ({percentage:.1f}%)")
    
    cache_dir = Path('data/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'{symbol.replace("/", "-")}_training_data.npz'
    
    logger.info(f"\n💾 Önbelleğe kaydediliyor: {cache_file}")
    np.savez_compressed(cache_file, X=X, y=y)
    
    logger.info("="*70)
    logger.info("✅ EĞİTİM VERİSİ HAZIR")
    logger.info("="*70)
    
    return X, y

async def async_main():
    """Asenkron ana giriş noktası."""
    parser = argparse.ArgumentParser(description='Makine öğrenmesi modelleri için eğitim verisi hazırlar.')
    parser.add_argument('--symbol', default=config.get('trading', {}).get('symbol', 'BTC/USDT'), help='İşlem yapılacak sembol')
    parser.add_argument('--timeframes', nargs='+', default=REGIME_TRAINING_TIMEFRAMES, help='Veri çekilecek zaman dilimleri')
    parser.add_argument('--no-feature-selection', action='store_true', help='Özellik seçimini devre dışı bırakır')
    
    args = parser.parse_args()
    X, y = await fetch_and_process_data(
        args.symbol,
        args.timeframes,
        use_feature_selection=not args.no_feature_selection
    )
    
    logger.info(f"\n✅ İŞLEM TAMAMLANDI: {len(X)} adet örnek, {X.shape[1]} özellik ile hazırlandı.")

def main():
    """Senkron ana giriş noktası."""
    asyncio.run(async_main())

if __name__ == '__main__':
    main()
