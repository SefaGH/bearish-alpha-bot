import asyncio
import os
import sys
import pandas as pd

# Projenin ana dizinini path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ccxt_client import CcxtClient
from src.ml.price_predictor import AdvancedPricePredictionEngine, MultiTimeframePricePredictor, EnsemblePricePredictor, LSTMPricePredictor, TransformerPricePredictor
from src.core.logger import setup_logger

logger = setup_logger("model-trainer", log_to_file=True, log_filename="logs/training.log")

# --- EĞİTİM PARAMETRELERİ ---
SYMBOLS_TO_TRAIN = ['BTC/USDT'] # Eğitilecek semboller
TIMEFRAMES_TO_TRAIN = ['1h', '4h'] # Eğitim için kullanılacak zaman dilimleri
CANDLE_LIMIT = 2000 # Her sembol ve zaman dilimi için ne kadar geçmiş veri çekileceği

async def main():
    """
    ML modellerini eğitmek ve kaydetmek için ana fonksiyon.
    """
    logger.info("="*50)
    logger.info("🤖 STARTING MODEL TRAINING SCRIPT 🤖")
    logger.info("="*50)

    # 1. Borsa istemcisini başlat (sadece veri çekmek için)
    exchange_client = CcxtClient('bingx')
    logger.info("✅ Exchange client initialized.")

    # 2. Fiyat tahmin motorunu ve içindeki modelleri başlat
    # Bu yapı, live_trading_launcher.py'deki ile aynı olmalı
    timeframe_models = {}
    for tf in TIMEFRAMES_TO_TRAIN:
        # Her zaman dilimi için LSTM ve Transformer modellerini oluştur
        base_models = {
            'lstm': LSTMPricePredictor(),
            'transformer': TransformerPricePredictor(d_model=50) # Örnek d_model, özellik sayısına göre ayarlanmalı
        }
        timeframe_models[tf] = EnsemblePricePredictor(base_models)
    
    multi_tf_predictor = MultiTimeframePricePredictor(timeframe_models)
    price_engine = AdvancedPricePredictionEngine(multi_tf_predictor)
    logger.info("✅ Price prediction engine and its sub-models initialized.")

    # 3. Her sembol ve zaman dilimi için veriyi topla
    training_data = {symbol: {} for symbol in SYMBOLS_TO_TRAIN}
    for symbol in SYMBOLS_TO_TRAIN:
        for timeframe in TIMEFRAMES_TO_TRAIN:
            logger.info(f"\n--- Fetching data for: {symbol} [{timeframe}] ---")
            try:
                # Geçmiş veriyi çek
                logger.info(f"Fetching {CANDLE_LIMIT} historical candles...")
                # Not: ccxt_client.ohlcv senkron bir metot ve DataFrame döndürüyor.
                ohlcv_df = exchange_client.ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT)
                
                if ohlcv_df is None or ohlcv_df.empty:
                    logger.warning(f"No historical data returned for {symbol} [{timeframe}].")
                    continue

                logger.info(f"✅ Fetched {len(ohlcv_df)} candles.")
                training_data[symbol][timeframe] = ohlcv_df

            except Exception as e:
                logger.error(f"❌ Failed to fetch data for {symbol} [{timeframe}]: {e}", exc_info=True)
    
    # 4. Toplanan veriyle modelleri eğit ve kaydet
    if any(training_data.values()):
         price_engine.train_and_save_models(training_data)
    else:
        logger.error("No data was fetched for training. Aborting.")


    logger.info("\n="*50)
    logger.info("✅ MODEL TRAINING SCRIPT COMPLETE ✅")
    logger.info("="*50)

if __name__ == "__main__":
    # Not: Bu betik asenkron metotlar içermiyorsa asyncio.run'a gerek olmayabilir.
    # Ancak ccxt kütüphanesinin gelecekteki async kullanımı için uyumlu bırakmak iyidir.
    # Şimdilik, ccxt_client senkron olduğu için doğrudan çalıştırabiliriz.
    # main() asenkron olduğu için asyncio.run kullanıyoruz.
    asyncio.run(main())
