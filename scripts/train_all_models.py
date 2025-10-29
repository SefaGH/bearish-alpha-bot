import asyncio
import os
import sys
import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import logging

# --- YOL AYARLAMASI (IMPORT HATALARINI ÖNLER) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- YOL AYARLAMASI SONU ---

# Gerekli modüllerin import edilmesi
from src.core.ccxt_client import CcxtClient
from src.core.logger import setup_logger
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.model_trainer import RegimeModelTrainer
from src.ml.price_predictor import (
    AdvancedPricePredictionEngine,
    MultiTimeframePricePredictor,
    EnsemblePricePredictor,
    LSTMPricePredictor,
    TransformerPricePredictor
)
from src.ml.label_generator import generate_regime_labels

# Logger kurulumu
logger = setup_logger("model-trainer", level=logging.DEBUG, log_to_file=True, log_filename="training.log")

# --- EĞİTİM PARAMETRELERİ ---
SYMBOLS_TO_TRAIN = ['BTC/USDT']
TIMEFRAMES_TO_TRAIN = ['1h', '4h']
CANDLE_LIMIT = 1440

async def main():
    logger.info("="*60)
    logger.info("🤖 BAŞLIYOR: BİRLEŞİK ML MODEL EĞİTİM BETİĞİ 🤖")
    logger.info("="*60)

    exchange_client = CcxtClient('bingx')
    feature_engine = FeatureEngineeringPipeline()
    logger.info("✅ Borsa istemcisi ve özellik motoru başlatıldı.")

    training_data = {symbol: {} for symbol in SYMBOLS_TO_TRAIN}
    for symbol in SYMBOLS_TO_TRAIN:
        for timeframe in TIMEFRAMES_TO_TRAIN:
            logger.info(f"\n--- Veri Çekiliyor: {symbol} [{timeframe}] ---")
            try:
                # *** DEĞİŞİKLİK: add_indicators=False olarak ayarlandı ***
                # Bu, eğitim ortamının canlı ortamı birebir taklit etmesini sağlar.
                # Özellik mühendisliği boru hattı (pipeline) zaten tüm indikatörleri hesaplayacaktır.
                ohlcv_df = await exchange_client.ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT, add_indicators=False)
                
                if ohlcv_df is None or ohlcv_df.empty:
                    logger.warning(f"Veri çekilemedi. Atlanıyor.")
                    continue
                logger.info(f"✅ {len(ohlcv_df)} adet HAM mum verisi çekildi.")
                training_data[symbol][timeframe] = ohlcv_df
            except Exception as e:
                logger.error(f"❌ Veri çekme hatası: {e}", exc_info=True)
    
    # 1. REJİM MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("🧠 ADIM 1: PİYASA REJİMİ MODELLERİ EĞİTİLİYOR 🧠")
    logger.info("="*60)
    
    regime_training_tf = TIMEFRAMES_TO_TRAIN[-1]
    if SYMBOLS_TO_TRAIN[0] in training_data and regime_training_tf in training_data[SYMBOLS_TO_TRAIN[0]]:
        regime_training_data = training_data[SYMBOLS_TO_TRAIN[0]].get(regime_training_tf)

        if regime_training_data is not None and not regime_training_data.empty:
            # === KÖK NEDEN ÇÖZÜMÜ BURADA ===
            # 1. Önce sadece ham OHLCV verisinden özellikleri çıkar.
            #    .copy() kullanarak orijinal verinin bozulmasını önle.
            features_df = feature_engine.extract_features(regime_training_data.copy())
            logger.info(f"Rejim modeli için {features_df.shape[1]} özellik çıkarıldı.")
            
            # 2. Ayrı bir şekilde, yine ham OHLCV verisinden etiketleri oluştur.
            regime_labels = generate_regime_labels(regime_training_data.copy())
            
            # 3. Özellikleri ve etiketleri, `prepare_for_training` ile hizala ve temizle.
            #    Bu metot, her ikisindeki NaN değerleri tutarlı bir şekilde temizleyerek
            #    X (sadece özellikler) ve y (sadece etiketler) dizilerini döndürür.
            X, y = feature_engine.prepare_for_training(features_df, regime_labels)
            
            # Artık X, sadece özellikleri içeriyor ve doğru sayıda sütuna sahip.
            # === ÇÖZÜM SONU ===

            if X.shape[0] > 100:
                logger.info(f"Model {X.shape[1]} özellik ile eğitilecek. (Örnek sayısı: {X.shape[0]})")
                regime_trainer = RegimeModelTrainer()
                # `train_ensemble_models` metodu zaten doğru imzaya sahip, sadece X ve y göndermek yeterli.
                training_results = regime_trainer.train_ensemble_models(X, y)
                logger.info(f"✅ Rejim modelleri eğitildi ve kaydedildi.")
            else:
                logger.warning("Rejim modellerini eğitmek için yeterli veri bulunamadı.")
        else:
            logger.error(f"Rejim eğitimi için {regime_training_tf} verisi bulunamadı.")
    else:
        logger.error(f"Rejim eğitimi için {SYMBOLS_TO_TRAIN[0]} sembolüne ait {regime_training_tf} verisi bulunamadı.")


    # 2. FİYAT TAHMİN MODELLERİ EĞİTİMİ (Bu kısım doğru çalışıyor, değişiklik gerekmiyor)
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    logger.info("="*60)
    
    if SYMBOLS_TO_TRAIN[0] in training_data:
        symbol_data_values = list(training_data[SYMBOLS_TO_TRAIN[0]].values())
        if not symbol_data_values:
            logger.error("Fiyat modeli eğitimi için hiç veri bulunamadı. Bu adım atlanıyor.")
        else:
            sample_features = feature_engine.extract_features(symbol_data_values[0])
            input_feature_size = sample_features.shape[1]
            logger.info(f"Fiyat tahmin modelleri için tutarlı girdi boyutu: {input_feature_size}")

            transformer_d_model = input_feature_size
            if transformer_d_model % 2 != 0:
                transformer_d_model += 1
                logger.warning(f"Transformer d_model tek sayı ({input_feature_size}) olamaz. "
                               f"{transformer_d_model}'e yükseltildi.")

            timeframe_models = {}
            for tf in TIMEFRAMES_TO_TRAIN:
                base_models = {
                    'lstm': LSTMPricePredictor(input_size=input_feature_size),
                    'transformer': TransformerPricePredictor(d_model=transformer_d_model)
                }
                timeframe_models[tf] = EnsemblePricePredictor(base_models)
            
            multi_tf_predictor = MultiTimeframePricePredictor(timeframe_models)
            price_engine = AdvancedPricePredictionEngine(multi_tf_predictor)
            
            price_engine.train_and_save_models(training_data)
    else:
        logger.error(f"Fiyat modeli eğitimi için {SYMBOLS_TO_TRAIN[0]} sembolüne ait veri bulunamadı.")


    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    if "ML_ENABLED" not in os.environ:
        os.environ["ML_ENABLED"] = "true"
        print("ML_ENABLED ortam değişkeni 'true' olarak ayarlandı.")
    
    asyncio.run(main())
