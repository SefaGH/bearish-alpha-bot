import asyncio
import os
import sys

# --- HATA DÜZELTMESİ BAŞLANGICI ---
# Bu blok, betiğin projenin ana dizinini tanımasını sağlar.
# Bu sayede 'src' gibi klasörlerden import işlemi başarılı olur.
# Bu kod, betiğin kendi konumundan bir üst dizine çıkarak ana yolu bulur.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- HATA DÜZELTMESİ SONU ---

import pandas as pd

from src.core.ccxt_client import CcxtClient
from src.core.logger import setup_logger
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.model_trainer import RegimeModelTrainer
from src.ml.price_predictor import AdvancedPricePredictionEngine, MultiTimeframePricePredictor, EnsemblePricePredictor, LSTMPricePredictor, TransformerPricePredictor
from src.ml.label_generator import generate_regime_labels

logger = setup_logger("model-trainer", log_to_file=True, log_filename="logs/training.log")

# --- EĞİTİM PARAMETRELERİ ---
SYMBOLS_TO_TRAIN = ['BTC/USDT']
TIMEFRAMES_TO_TRAIN = ['1h', '4h']
CANDLE_LIMIT = 2000

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
                ohlcv_df = exchange_client.ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT, add_indicators=True)
                if ohlcv_df is None or ohlcv_df.empty:
                    logger.warning(f"Veri çekilemedi. Atlanıyor.")
                    continue
                logger.info(f"✅ {len(ohlcv_df)} adet mum verisi çekildi.")
                training_data[symbol][timeframe] = ohlcv_df
            except Exception as e:
                logger.error(f"❌ Veri çekme hatası: {e}", exc_info=True)
    
    # REJİM MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("🧠 ADIM 1: PİYASA REJİMİ MODELLERİ EĞİTİLİYOR 🧠")
    logger.info("="*60)
    
    regime_training_tf = TIMEFRAMES_TO_TRAIN[-1]
    regime_training_data = training_data[SYMBOLS_TO_TRAIN[0]].get(regime_training_tf)

    if regime_training_data is not None and not regime_training_data.empty:
        regime_labels = generate_regime_labels(regime_training_data)
        features_df = feature_engine.extract_features(regime_training_data)
        
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)
        
        X, y = feature_engine.prepare_for_training(features_df, regime_labels)

        if X.shape[0] > 100:
            regime_trainer = RegimeModelTrainer()
            training_results = regime_trainer.train_ensemble_models(X, y)
            logger.info(f"✅ Rejim modelleri eğitildi ve kaydedildi.")
        else:
            logger.warning("Rejim modellerini eğitmek için yeterli veri bulunamadı.")
    else:
        logger.error(f"Rejim eğitimi için {regime_training_tf} verisi bulunamadı.")

    # FİYAT TAHMİN MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    logger.info("="*60)
    
    sample_features = feature_engine.extract_features(next(iter(training_data[SYMBOLS_TO_TRAIN[0]].values())))
    input_feature_size = sample_features.shape[1]
    logger.info(f"Fiyat tahmin modelleri için dinamik girdi boyutu: {input_feature_size}")

    timeframe_models = {}
    for tf in TIMEFRAMES_TO_TRAIN:
        base_models = {
            'lstm': LSTMPricePredictor(input_size=input_feature_size),
            'transformer': TransformerPricePredictor(d_model=input_feature_size)
        }
        timeframe_models[tf] = EnsemblePricePredictor(base_models)
    
    multi_tf_predictor = MultiTimeframePricePredictor(timeframe_models)
    price_engine = AdvancedPricePredictionEngine(multi_tf_predictor)
    
    price_engine.train_and_save_models(training_data)

    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    if "ML_ENABLED" not in os.environ:
        os.environ["ML_ENABLED"] = "true"
        print("ML_ENABLED ortam değişkeni 'true' olarak ayarlandı.")
    asyncio.run(main())
