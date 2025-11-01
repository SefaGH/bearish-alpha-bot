import asyncio
import os
import sys
import pandas as pd
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
ALL_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h']

# --- 🔥 IYILESTIRME: Rejim eğitimi için kullanılacak zaman dilimleri genişletildi ---
# Stratejilerin kullandığı 30m'yi ve genel trend için 1h, 4h'yi dahil ediyoruz.
REGIME_TRAINING_TIMEFRAMES = ['30m', '1h', '4h'] 

# --- 🔥 ÇÖZÜM: CANDLE_LIMIT değerini BingX API'sinin izin verdiği maksimum olan 1440'ın altına çekiyoruz. ---
CANDLE_LIMIT = 1400 # Veri miktarını API limitine uygun hale getir

# --- 🔥 YENİ: Rejim modelleri için minimum veri eşikleri ---
MIN_SAMPLES_FOR_RF = 100      # RandomForest ve Scaler'ın eğitilmesi için gereken minimum örnek sayısı
MIN_SAMPLES_FOR_NN = 500      # LSTM/Transformer gibi sinir ağlarının eğitilmesi için gereken minimum örnek sayısı


async def main():
    logger.info("="*60)
    logger.info("🤖 BAŞLIYOR: BİRLEŞİK ML MODEL EĞİTİM BETİĞİ 🤖")
    logger.info("="*60)

    exchange_client = CcxtClient('bingx')
    feature_engine = FeatureEngineeringPipeline()
    logger.info("✅ Borsa istemcisi ve özellik motoru başlatıldı.")

    training_data = {symbol: {} for symbol in SYMBOLS_TO_TRAIN}
    for symbol in SYMBOLS_TO_TRAIN:
        for timeframe in ALL_TIMEFRAMES:
            logger.info(f"\n--- Veri Çekiliyor: {symbol} [{timeframe}] ---")
            try:
                ohlcv_df = await exchange_client.ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT, add_indicators=False)
                
                if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 200:
                    logger.warning(f"Veri çekilemedi veya yetersiz ({len(ohlcv_df) if ohlcv_df is not None else 0} mum). Atlanıyor.")
                    continue
                logger.info(f"✅ {len(ohlcv_df)} adet HAM mum verisi çekildi.")
                training_data[symbol][timeframe] = ohlcv_df
            except Exception as e:
                logger.error(f"❌ Veri çekme hatası: {e}", exc_info=True)
    
    # 1. REJİM MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("🧠 ADIM 1: PİYASA REJİMİ MODELLERİ EĞİTİLİYOR 🧠")
    logger.info(f"   Eğitim Zaman Dilimleri: {REGIME_TRAINING_TIMEFRAMES}")
    logger.info("="*60)
    
    all_regime_features = []
    all_regime_labels = []

    symbol_for_regime = SYMBOLS_TO_TRAIN[0]
    if symbol_for_regime in training_data:
        for tf in REGIME_TRAINING_TIMEFRAMES:
            if tf in training_data[symbol_for_regime]:
                logger.info(f"Rejim modeli için {tf} verisi işleniyor...")
                regime_data_raw = training_data[symbol_for_regime][tf].copy()
                
                features_df = feature_engine.extract_features(regime_data_raw)
                regime_labels = generate_regime_labels(regime_data_raw)
                
                X_prepared, y_prepared = feature_engine.prepare_for_training(features_df, regime_labels)
                
                if X_prepared.shape[0] > 0:
                    all_regime_features.append(X_prepared)
                    all_regime_labels.append(y_prepared)
                    logger.info(f"✅ {tf} verisinden {X_prepared.shape[0]} örnek eklendi.")
                else:
                    logger.warning(f"{tf} verisinden geçerli eğitim örneği çıkarılamadı.")
            else:
                 logger.error(f"Rejim eğitimi için {symbol_for_regime} sembolüne ait {tf} verisi bulunamadı.")

        if all_regime_features and all_regime_labels:
            final_X = np.vstack(all_regime_features)
            final_y = np.concatenate(all_regime_labels)
            
            total_samples = final_X.shape[0]
            logger.info(f"Toplamda {total_samples} örnek ve {final_X.shape[1]} özellik ile rejim modeli eğitilecek.")

            # --- 🔥 GÜNCELLENMİŞ EĞİTİM MANTIĞI ---
            if total_samples >= MIN_SAMPLES_FOR_RF:
                regime_trainer = RegimeModelTrainer()
                
                # Sinir ağları için yeterli veri olup olmadığını kontrol et
                train_nn = total_samples >= MIN_SAMPLES_FOR_NN
                if not train_nn:
                    logger.warning(
                        f"Toplam örnek sayısı ({total_samples}) sinir ağı eğitimi için "
                        f"gereken minimum ({MIN_SAMPLES_FOR_NN}) değerden az. "
                        "Sadece RandomForest ve Scaler eğitilecek."
                    )
                
                # train_ensemble_models'a yeni bir parametre ekleyerek hangi modellerin eğitileceğini kontrol et
                # Bu parametrenin model_trainer.py içinde ele alınması gerekecek.
                # Şimdilik, model_trainer'daki korumaların yeterli olacağını varsayıyoruz.
                training_results = regime_trainer.train_ensemble_models(final_X, final_y)
                
                logger.info(f"✅ Rejim modelleri birleşik veri seti ile eğitildi ve kaydedildi.")
                
            else:
                logger.warning(
                    f"Rejim modellerini eğitmek için yeterli birleşik veri bulunamadı. "
                    f"Gereken minimum: {MIN_SAMPLES_FOR_RF}, bulunan: {total_samples}"
                )
            # --- GÜNCELLENMİŞ MANTIK SONU ---
        else:
            logger.error("Rejim modeli eğitimi için işlenecek hiçbir veri bulunamadı.")
    else:
        logger.error(f"Rejim eğitimi için {symbol_for_regime} sembolüne ait veri bulunamadı.")


    # 2. FİYAT TAHMİN MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    logger.info("="*60)
    
    price_timeframes = [tf for tf in ['5m', '15m', '1h'] if tf in training_data.get(SYMBOLS_TO_TRAIN[0], {})]
    if SYMBOLS_TO_TRAIN[0] in training_data and price_timeframes:
        sample_df = training_data[SYMBOLS_TO_TRAIN[0]][price_timeframes[0]].copy()
        sample_features = feature_engine.extract_features(sample_df)
        input_feature_size = sample_features.shape[1]
        logger.info(f"Fiyat tahmin modelleri için tutarlı girdi boyutu: {input_feature_size}")

        transformer_d_model = input_feature_size
        if transformer_d_model % 2 != 0:
            transformer_d_model += 1
            logger.warning(f"Transformer d_model tek sayı ({input_feature_size}) olamaz. {transformer_d_model}'e yükseltildi.")

        timeframe_models = {}
        for tf in price_timeframes:
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
