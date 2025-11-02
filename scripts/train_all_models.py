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
# --- YENİ: RL EĞİTİMİ İÇİN GEREKLİ IMPORT'LAR ---
from src.ml.reinforcement_learning import TradingRLAgent, ExperienceReplay
from src.ml.rl_trading_env import RLTradingEnv
from src.ml.rl_model_trainer import RLModelTrainer
# --- YENİ IMPORT'LAR SONU ---

# Logger kurulumu
logger = setup_logger("model-trainer", level=logging.DEBUG, log_to_file=True, log_filename="training.log")

# --- EĞİTİM PARAMETRELERİ ---
SYMBOLS_TO_TRAIN = ['BTC/USDT']
ALL_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h']
REGIME_TRAINING_TIMEFRAMES = ['30m', '1h', '4h'] 
CANDLE_LIMIT = 2000 # Daha fazla veri, daha iyi eğitim

MIN_SAMPLES_FOR_RF = 100
MIN_SAMPLES_FOR_NN = 500

# --- YENİ: RL EĞİTİM PARAMETRELERİ ---
RL_TRAINING_TIMEFRAME = '15m'  # RL eğitimi için kullanılacak zaman dilimi
RL_NUM_EPISODES = 250          # Ajanın kaç bölüm (episode) boyunca eğitileceği
RL_BATCH_SIZE = 64             # Her öğrenme adımında kullanılacak deneyim sayısı
RL_BUFFER_SIZE = 100000        # Deneyim tekrarı belleğinin kapasitesi
# --- YENİ PARAMETRELER SONU ---


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
    
    # 1. REJİM MODELLERİ EĞİTİMİ (Mevcut kodunuz aynı kalıyor)
    logger.info("\n" + "="*60)
    logger.info("🧠 ADIM 1: PİYASA REJİMİ MODELLERİ EĞİTİLİYOR 🧠")
    # ... (Bu bölümün tamamı sizin kodunuzla aynı)
    # ...
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
                
                X_prepared, y_prepared = feature_engine.prepare_for_training(
                    features_df, 
                    regime_labels
                )
                
                if X_prepared.shape[0] > 0:
                    all_regime_features.append(X_prepared)
                    all_regime_labels.append(y_prepared)
                    logger.info(f"✅ {tf} verisinden {X_prepared.shape[0]} örnek eklendi.")
        
        if all_regime_features and all_regime_labels:
            final_X = np.vstack(all_regime_features)
            final_y = np.concatenate(all_regime_labels)
            
            if final_X.shape[0] >= MIN_SAMPLES_FOR_RF:
                regime_trainer = RegimeModelTrainer()
                regime_trainer.train_ensemble_models(final_X, final_y)
                logger.info(f"✅ Rejim modelleri birleşik veri seti ile eğitildi ve kaydedildi.")

    # 2. FİYAT TAHMİN MODELLERİ EĞİTİMİ (Mevcut kodunuz aynı kalıyor)
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    # ... (Bu bölümün tamamı sizin kodunuzla aynı)
    # ...
    logger.info("="*60)
    
    price_timeframes = [tf for tf in ['5m', '15m', '1h'] if tf in training_data.get(SYMBOLS_TO_TRAIN[0], {})]
    if SYMBOLS_TO_TRAIN[0] in training_data and price_timeframes:
        sample_df = training_data[SYMBOLS_TO_TRAIN[0]][price_timeframes[0]].copy()
        sample_features = feature_engine.extract_features(sample_df)
        input_feature_size = sample_features.shape[1]
        
        transformer_d_model = input_feature_size
        if transformer_d_model % 2 != 0:
            transformer_d_model += 1

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

    # --- YENİ: ADIM 3 ---
    # 3. REINFORCEMENT LEARNING AJANI EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("🤖 ADIM 3: REINFORCEMENT LEARNING AJANI EĞİTİLİYOR 🤖")
    logger.info(f"   Eğitim Zaman Dilimi: {RL_TRAINING_TIMEFRAME}, Bölüm Sayısı: {RL_NUM_EPISODES}")
    logger.info("="*60)

    symbol_for_rl = SYMBOLS_TO_TRAIN[0]
    if symbol_for_rl in training_data and RL_TRAINING_TIMEFRAME in training_data[symbol_for_rl]:
        logger.info(f"RL ajanı için {RL_TRAINING_TIMEFRAME} verisi hazırlanıyor...")
        
        rl_data_raw = training_data[symbol_for_rl][RL_TRAINING_TIMEFRAME].copy()
        rl_features_df = feature_engine.extract_features(rl_data_raw)
        
        # NaN değerleri temizle
        rl_features_df.fillna(method='ffill', inplace=True)
        rl_features_df.fillna(method='bfill', inplace=True)
        rl_features_df.dropna(inplace=True)
        
        if rl_features_df.empty:
            logger.error("RL eğitimi için özellik çıkarıldıktan sonra veri kalmadı.")
        else:
            logger.info(f"✅ RL eğitimi için {len(rl_features_df)} adet kullanılabilir veri noktası hazırlandı.")
            
            # RL Ortamını ve Ajanını Başlat
            env = RLTradingEnv(df=rl_features_df)
            state_dim = env.state_dim
            action_dim = env.action_dim
            
            agent = TradingRLAgent(state_size=state_dim, action_size=action_dim)
            experience_replay = ExperienceReplay(buffer_size=RL_BUFFER_SIZE)
            
            # RL Eğiticisini Başlat ve Eğitimi Başlat
            rl_trainer = RLModelTrainer(agent, env, experience_replay)
            
            try:
                rl_trainer.train(num_episodes=RL_NUM_EPISODES, batch_size=RL_BATCH_SIZE)
                logger.info("✅ RL Ajanı başarıyla eğitildi ve kaydedildi.")
            except Exception as e:
                logger.error(f"❌ RL eğitimi sırasında bir hata oluştu: {e}", exc_info=True)

    else:
        logger.error(f"RL eğitimi için gerekli olan {symbol_for_rl} sembolüne ait {RL_TRAINING_TIMEFRAME} verisi bulunamadı.")
    # --- YENİ ADIM 3 SONU ---


    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    if "ML_ENABLED" not in os.environ:
        os.environ["ML_ENABLED"] = "true"
        print("ML_ENABLED ortam değişkeni 'true' olarak ayarlandı.")
    
    asyncio.run(main())
