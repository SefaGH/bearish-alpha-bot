"""
Unified ML Model Training Script for Bearish Alpha Bot.

This script trains all ML models (Regime Prediction, Price Prediction, RL Agent)
with architecture parameters synchronized from config.example.yaml.

CRITICAL: All model architectures MUST be synchronized with config.example.yaml:
  - Regime LSTM: hidden_size=64, num_layers=2 (from ml.regime_prediction.model_params.lstm_regime)
  - Price models: parameters from ml.price_prediction.model_params
  - RL Agent: parameters from ml.reinforcement_learning

This ensures:
  1. No size mismatch errors during model loading
  2. Consistent architecture across training and inference
  3. Reduced overfitting risk with smaller, safer models
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
import logging
import yaml

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

# === DÜZELTME: CANDLE_LIMIT, BingX API limitine (1440) uyacak şekilde güncellendi ===
CANDLE_LIMIT = 1440 # Daha fazla veri, daha iyi eğitim

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

    # Load configuration from config.example.yaml
    config_path = os.path.join(project_root, 'config', 'config.example.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_path}")
        logger.error("Please ensure config.example.yaml exists in the config/ directory.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"❌ Error parsing configuration file: {e}")
        raise
    
    ml_config = config.get('ml', {})
    regime_pred_config = ml_config.get('regime_prediction', {})
    price_pred_config = ml_config.get('price_prediction', {})
    rl_config = ml_config.get('reinforcement_learning', {})
    
    logger.info(f"✅ Configuration loaded from {config_path}")
    logger.info(f"   Regime LSTM params: {regime_pred_config.get('model_params', {}).get('lstm_regime', {})}")

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
                # Pass regime_prediction config to trainer so it uses correct architecture
                regime_trainer = RegimeModelTrainer(config=regime_pred_config)
                regime_trainer.train_ensemble_models(final_X, final_y)
                logger.info(f"✅ Rejim modelleri birleşik veri seti ile eğitildi ve kaydedildi.")

    # 2. FİYAT TAHMİN MODELLERİ EĞİTİMİ
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    logger.info("="*60)
    
    # ✔️ KESİN ÇÖZÜM: 'AdvancedPricePredictionEngine' sınıfını, ana uygulamadaki gibi
    # doğru konfigürasyon bloğuyla başlatıyoruz. Bu sınıf, artık kendi içinde
    # MultiTimeframePricePredictor ve diğer alt modelleri kendi inşa edecektir.
    # Bu, hem TypeError hatasını çözer hem de yapısal senkronizasyonu sağlar.

    if SYMBOLS_TO_TRAIN[0] in training_data:
        try:
            logger.info("AdvancedPricePredictionEngine konfigürasyona göre başlatılıyor...")
            price_engine = AdvancedPricePredictionEngine(
                market_data_pipeline=None,      # Eğitim sırasında pipeline gerekmez
                feature_pipeline=feature_engine,  # Önceden oluşturulan özellik motorunu ver
                config=price_pred_config          # Sadece fiyat tahminine özel konfigürasyonu ver
            )
            logger.info("✅ Fiyat tahmin motoru, eğitim için başarıyla başlatıldı.")
            
            # Ana sınıf üzerinden eğitimi ve kaydetmeyi tetikle
            logger.info("Model eğitimi ve kaydetme süreci başlatılıyor...")
            price_engine.train_and_save_models(training_data)
            logger.info("✅ Fiyat tahmin modellerinin eğitimi ve kaydı tamamlandı.")

        except Exception as e:
            logger.error(f"❌ Fiyat tahmin modelleri eğitimi sırasında kritik hata: {e}", exc_info=True)

    else:
        logger.warning("Fiyat tahmini eğitimi için veri bulunamadı, bu adım atlanıyor.")

    # 3. REINFORCEMENT LEARNING AJANI EĞİTİLİYOR
    logger.info("\n" + "="*60)
    logger.info("🤖 ADIM 3: REINFORCEMENT LEARNING AJANI EĞİTİLİYOR 🤖")
    logger.info(f"   Eğitim Zaman Dilimi: {RL_TRAINING_TIMEFRAME}, Bölüm Sayısı: {RL_NUM_EPISODES}")
    logger.info("="*60)

    symbol_for_rl = SYMBOLS_TO_TRAIN[0]
    if symbol_for_rl in training_data and RL_TRAINING_TIMEFRAME in training_data[symbol_for_rl]:
        logger.info(f"RL ajanı için {RL_TRAINING_TIMEFRAME} verisi hazırlanıyor...")
        
        # ... (RL için veri hazırlama kısmı aynı kalır)
        rl_data_raw = training_data[symbol_for_rl][RL_TRAINING_TIMEFRAME].copy()
        rl_features_df = feature_engine.extract_features(rl_data_raw)
        
        common_index = rl_data_raw.index.intersection(rl_features_df.index)
        rl_data_raw = rl_data_raw.loc[common_index]
        rl_features_df = rl_features_df.loc[common_index]
        rl_features_df.ffill(inplace=True)
        rl_features_df.bfill(inplace=True)
        rl_features_df.dropna(inplace=True)
        
        final_index = rl_features_df.index
        rl_data_raw = rl_data_raw.loc[final_index]
        
        if rl_features_df.empty:
            logger.error("RL eğitimi için özellik çıkarıldıktan sonra veri kalmadı.")
        else:
            logger.info(f"✅ RL eğitimi için {len(rl_features_df)} adet kullanılabilir veri noktası hazırlandı.")
            
            env = RLTradingEnv(features_df=rl_features_df, raw_df=rl_data_raw)
            state_dim = env.state_dim
            action_dim = env.action_dim
            
            logger.info(f"✅ RL Ortamı oluşturuldu. State boyutu: {state_dim}, Aksiyon boyutu: {action_dim}")

            # ✔️ KESİN ÇÖZÜM: 'TradingRLAgent' sınıfı, `__init__` metodunda 'batch_size' argümanı beklemiyor.
            # Bunun yerine, tüm 'reinforcement_learning' konfigürasyon bloğunu ('rl_config') bekliyor.
            # 'batch_size' parametresini kaldırıp, 'config' parametresini ekliyoruz.
            agent = TradingRLAgent(
                state_size=state_dim, 
                action_size=action_dim,
                config=rl_config  # <-- `batch_size` yerine bu satırı ekliyoruz.
            )

            # Deneyim belleğini doğru buffer boyutu ile başlat
            experience_replay = ExperienceReplay(buffer_size=RL_BUFFER_SIZE)
            
            # Eğiticiyi başlat
            rl_trainer = RLModelTrainer(agent, env, experience_replay)
            
            try:
                rl_trainer.train(num_episodes=RL_NUM_EPISODES)
                logger.info("✅ RL Ajanı başarıyla eğitildi ve kaydedildi.")
            except Exception as e:
                logger.error(f"❌ RL eğitimi sırasında bir hata oluştu: {e}", exc_info=True)
    else:
        logger.error(f"RL eğitimi için gerekli olan {symbol_for_rl} sembolüne ait {RL_TRAINING_TIMEFRAME} verisi bulunamadı.")


    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    if "ML_ENABLED" not in os.environ:
        os.environ["ML_ENABLED"] = "true"
        print("ML_ENABLED ortam değişkeni 'true' olarak ayarlandı.")
    
    asyncio.run(main())
