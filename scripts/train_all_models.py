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
import json
from datetime import datetime

# --- YOL AYARLAMASI (IMPORT HATALARINI ÖNLER) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- YOL AYARLAMASI SONU ---

# --- YENİ: Merkezi Config Import ---
from src.config.live_trading_config import LiveTradingConfiguration
# --- YENİ IMPORT SONU ---

# Gerekli modüllerin import edilmesi
from src.core.ccxt_client import CcxtClient
from src.core.logger import setup_logger
from src.core.market_data_pipeline import MarketDataPipeline
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

# --- PERFORMANCE TRACKING ---
from scripts.utils.model_performance_tracker import ModelPerformanceTracker
# --- PERFORMANCE TRACKING SONU ---

from scripts.utils.training_validator import TrainingConfigValidator

# Logger kurulumu
logger = setup_logger("model-trainer", level=logging.INFO, log_to_file=True, log_filename="training.log")

# --- EĞİTİM PARAMETRELERİ ---
SYMBOLS_TO_TRAIN = ['BTC/USDT']
ALL_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']  # 1d eklendi - regime için gerekli
REGIME_TRAINING_TIMEFRAMES = ['15m', '30m', '1h', '4h', '1d']  # 5 timeframe - regime detection için optimal

# === DÜZELTME: CANDLE_LIMIT, BingX API limitine (1440) uyacak şekilde güncellendi ===
CANDLE_LIMIT = 1440  # BingX limiti (değişmez)

MIN_SAMPLES_FOR_RF = 100
MIN_SAMPLES_FOR_NN = 1000  # Daha stabil model eğitimi için artırıldı

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
    
    # Report GPU availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
        logger.info(f"CUDA Available: {cuda_available} | Device: {device_name}")
    except ImportError:
        logger.info("PyTorch not installed, GPU check skipped")
    
    # Initialize performance tracker
    tracker = ModelPerformanceTracker()
    logger.info("✅ Performance tracker initialized")
    
    # Initialize metrics tracking
    start_time = datetime.now()
    training_metrics = {
        'start_time': start_time.isoformat(),
        'symbols': SYMBOLS_TO_TRAIN,
        'timeframes': ALL_TIMEFRAMES,
        'regime_models': {},
        'price_models': {},
        'rl_models': {}
    }

    # =========================================================================
    # CONFIGURATION LOADING (Using Centralized System)
    # =========================================================================
    # Use LiveTradingConfiguration for consistent config loading across
    # training and live trading. This ensures:
    #   1. Environment variable overrides work correctly
    #   2. Config validation is applied
    #   3. Type casting is automatic
    #   4. Single source of truth
    # =========================================================================
    
    logger.info("Loading configuration using centralized system...")
    
    try:
        # Use centralized config loader (handles env vars, validation, etc.)
        # Suppress duplicate logging since we'll log training-specific details
        config = LiveTradingConfiguration.load(log_summary=False)
        logger.info("✅ Configuration loaded successfully via centralized system")
        
        # Extract ML configuration blocks
        ml_config = config.get('ml', {})
        regime_pred_config = ml_config.get('regime_prediction', {})
        price_pred_config = ml_config.get('price_prediction', {})
        rl_config = ml_config.get('reinforcement_learning', {})
        
        # =====================================================================
        # TRAINING-SPECIFIC CONFIGURATION LOGGING
        # =====================================================================
        logger.info("="*60)
        logger.info("🎓 TRAINING CONFIGURATION")
        logger.info("="*60)
        
        # RL Training Mode Validation
        rl_training_mode = rl_config.get('training_mode', False)
        logger.info(f"   RL Training Mode: {rl_training_mode}")
        
        if not rl_training_mode:
            logger.warning("="*60)
            logger.warning("⚠️  WARNING: RL training_mode is False in config!")
            logger.warning("⚠️  This may be due to:")
            logger.warning("    1. config.example.yaml has training_mode: false")
            logger.warning("    2. ML_RL_TRAINING_MODE env var is not set/false")
            logger.warning("⚠️  Forcing training_mode=True for this training session")
            logger.warning("="*60)
            rl_config['training_mode'] = True
            logger.info(f"   RL Training Mode (forced): {rl_config['training_mode']}")
        
        # RL Epsilon Parameters Check
        epsilon_params = {
            'epsilon_start': rl_config.get('epsilon_start'),
            'epsilon_decay': rl_config.get('epsilon_decay'),
            'epsilon_min': rl_config.get('epsilon_min')
        }
        logger.info("   RL Epsilon Schedule:")
        for param, value in epsilon_params.items():
            if value is None:
                logger.warning(f"      {param}: NOT SET (will use default)")
            else:
                logger.info(f"      {param}: {value}")
        
        # Regime LSTM Parameters
        lstm_params = regime_pred_config.get('model_params', {}).get('lstm_regime', {})
        logger.info("   Regime LSTM Parameters (from config):")
        if lstm_params:
            logger.info(f"      hidden_size: {lstm_params.get('hidden_size', 'NOT SET')}")
            logger.info(f"      num_layers: {lstm_params.get('num_layers', 'NOT SET')}")
            logger.info(f"      dropout: {lstm_params.get('dropout', 'NOT SET')}")
        else:
            logger.warning("      ⚠️  LSTM params not found in config (will use defaults)")
        
        # Training Symbols
        logger.info(f"   Training Symbols: {', '.join(SYMBOLS_TO_TRAIN)}")
        logger.info("="*60)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Configuration file not found: {e}")
        logger.error("Please ensure config/config.example.yaml exists.")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}", exc_info=True)
        raise
    
    # =========================================================================
    # END CONFIGURATION LOADING
    # ==========================================================================
    
    # =========================================================================
    # PRE-TRAINING VALIDATION
    # =========================================================================
    # Validate configuration before starting expensive training process
    # This catches common issues early and provides clear error messages
    # =========================================================================
    
    logger.info("\n" + "="*60)
    logger.info("🔍 VALIDATING TRAINING CONFIGURATION")
    logger.info("="*60)
    
    # Run validation
    is_valid, issues = TrainingConfigValidator.validate(config)
    TrainingConfigValidator.log_validation_results(is_valid, issues)
    
    # Check for critical issues
    critical_issues = [i for i in issues if i.startswith("CRITICAL:")]
    if critical_issues:
        logger.error("❌ Critical validation errors found. Aborting training.")
        for issue in critical_issues:
            logger.error(f"   - {issue}")
        raise ValueError(f"Training validation failed with {len(critical_issues)} critical issues")
    
    # Check parameter synchronization
    logger.info("Checking parameter synchronization between config and code...")
    sync_issues = TrainingConfigValidator.validate_model_params_sync(config)
    
    if sync_issues:
        logger.warning("="*60)
        logger.warning("⚠️  PARAMETER SYNCHRONIZATION ISSUES DETECTED")
        logger.warning("="*60)
        for issue in sync_issues:
            logger.warning(f"   - {issue}")
        logger.warning("⚠️  Training will use config values (config takes precedence)")
        logger.warning("⚠️  Consider updating model_trainer.py constants to match")
        logger.warning("="*60)
    else:
        logger.info("✅ Config and code parameters are synchronized")
    
    logger.info("="*60)
    logger.info("✅ VALIDATION COMPLETE - PROCEEDING WITH TRAINING")
    logger.info("="*60 + "\n")
    
    # =========================================================================
    # END VALIDATION
    # =========================================================================
    
    exchange_client = CcxtClient('bingx')
    feature_engine = FeatureEngineeringPipeline()
    
    # Create MarketDataPipeline for price predictor (avoids warning during initialization)
    # During training we don't actually use it, but passing it prevents the warning
    market_pipeline = MarketDataPipeline(
        exchanges={'bingx': exchange_client},
        config=config
    )
    
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
    logger.info("="*60)
    logger.info("🧠 REGIME MODEL TRAINING CONFIGURATION")
    logger.info(f"   Timeframes: {REGIME_TRAINING_TIMEFRAMES}")
    logger.info(f"   Candle limit per timeframe: {CANDLE_LIMIT}")
    logger.info(f"   Expected total samples: {len(REGIME_TRAINING_TIMEFRAMES) * CANDLE_LIMIT}")
    logger.info(f"   Minimum NN samples: {MIN_SAMPLES_FOR_NN}")
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
            else:
                logger.warning(f"⚠️ {tf} için veri bulunamadı, atlanıyor...")
        
        if all_regime_features and all_regime_labels:
            final_X = np.vstack(all_regime_features)
            final_y = np.concatenate(all_regime_labels)
            
            logger.info("="*60)
            logger.info(f"✅ Total training samples: {len(final_X)} (from {len(REGIME_TRAINING_TIMEFRAMES)} timeframes)")
            logger.info("="*60)
            
            if final_X.shape[0] >= MIN_SAMPLES_FOR_RF:
                # Pass regime_prediction config to trainer so it uses correct architecture
                regime_training_start = datetime.now()
                regime_trainer = RegimeModelTrainer(config=regime_pred_config)
                results = regime_trainer.train_ensemble_models(final_X, final_y)
                regime_training_time = (datetime.now() - regime_training_start).total_seconds()
                
                logger.info(f"✅ Rejim modelleri birleşik veri seti ile eğitildi ve kaydedildi.")
                
                # Store regime model metrics (safely handle None results)
                if results:
                    training_metrics['regime_models'] = {
                        'total_samples': final_X.shape[0],
                        'feature_count': final_X.shape[1],
                        'metrics': results.get('metrics', {})
                    }
                    
                    # Record to performance tracker
                    try:
                        tracker.record_training(
                            model_type="regime",
                            model_name=f"{symbol_for_regime.replace('/', '-')}_ensemble",
                            metrics=results.get('metrics', {}),
                            data_info={
                                'total_samples': final_X.shape[0],
                                'train_samples': final_X.shape[0],
                                'features': final_X.shape[1],
                                'timeframes': ','.join(REGIME_TRAINING_TIMEFRAMES),
                                'symbol': symbol_for_regime,
                                'timeframe_count': len(REGIME_TRAINING_TIMEFRAMES)  # EKLE: Kaç timeframe kullanıldı
                            },
                            training_time=regime_training_time
                        )
                    except Exception as e:
                        logger.error(f"Failed to record regime training metrics: {e}")
                else:
                    training_metrics['regime_models'] = {
                        'total_samples': final_X.shape[0],
                        'feature_count': final_X.shape[1],
                        'metrics': {}
                    }

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
            price_training_start = datetime.now()
            
            price_engine = AdvancedPricePredictionEngine(
                market_data_pipeline=market_pipeline,  # Pass MarketDataPipeline to avoid warning
                feature_pipeline=feature_engine,  # Önceden oluşturulan özellik motorunu ver
                config=price_pred_config          # Sadece fiyat tahminine özel konfigürasyonu ver
            )
            logger.info("✅ Fiyat tahmin motoru, eğitim için başarıyla başlatıldı.")
            
            # Ana sınıf üzerinden eğitimi ve kaydetmeyi tetikle
            logger.info("Model eğitimi ve kaydetme süreci başlatılıyor...")
            price_engine.train_and_save_models(training_data)
            price_training_time = (datetime.now() - price_training_start).total_seconds()
            
            logger.info("✅ Fiyat tahmin modellerinin eğitimi ve kaydı tamamlandı.")
            
            # Store price model metrics
            training_metrics['price_models'] = {
                'status': 'completed',
                'models_trained': ['LSTM', 'Transformer', 'Ensemble']
            }
            
            # Record to performance tracker (generic metrics since we don't have detailed results)
            try:
                tracker.record_training(
                    model_type="price",
                    model_name=f"{SYMBOLS_TO_TRAIN[0].replace('/', '-')}_ensemble",
                    metrics={
                        'status': 'completed',
                        'training_time_seconds': price_training_time
                    },
                    data_info={
                        'symbol': SYMBOLS_TO_TRAIN[0],
                        'timeframes': ','.join(ALL_TIMEFRAMES),
                        'models': ['LSTM', 'Transformer', 'Ensemble']
                    },
                    training_time=price_training_time
                )
            except Exception as e:
                logger.error(f"Failed to record price training metrics: {e}")

        except Exception as e:
            logger.error(f"❌ Fiyat tahmin modelleri eğitimi sırasında kritik hata: {e}", exc_info=True)
            training_metrics['price_models'] = {
                'status': 'failed',
                'error': str(e)
            }

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
                rl_training_start = datetime.now()
                rl_trainer.train(num_episodes=RL_NUM_EPISODES)
                rl_training_time = (datetime.now() - rl_training_start).total_seconds()
                
                logger.info("✅ RL Ajanı başarıyla eğitildi ve kaydedildi.")
                
                # Store RL model metrics
                training_metrics['rl_models'] = {
                    'status': 'completed',
                    'num_episodes': RL_NUM_EPISODES,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'training_samples': len(rl_features_df)
                }
                
                # Record to performance tracker
                try:
                    tracker.record_training(
                        model_type="rl",
                        model_name=f"{symbol_for_rl.replace('/', '-')}_{RL_TRAINING_TIMEFRAME}",
                        metrics={
                            'num_episodes': RL_NUM_EPISODES,
                            'state_dim': state_dim,
                            'action_dim': action_dim
                        },
                        data_info={
                            'training_samples': len(rl_features_df),
                            'symbol': symbol_for_rl,
                            'timeframe': RL_TRAINING_TIMEFRAME
                        },
                        training_time=rl_training_time
                    )
                except Exception as e:
                    logger.error(f"Failed to record RL training metrics: {e}")
            except Exception as e:
                logger.error(f"❌ RL eğitimi sırasında bir hata oluştu: {e}", exc_info=True)
                training_metrics['rl_models'] = {
                    'status': 'failed',
                    'error': str(e)
                }
    else:
        logger.error(f"RL eğitimi için gerekli olan {symbol_for_rl} sembolüne ait {RL_TRAINING_TIMEFRAME} verisi bulunamadı.")
        training_metrics['rl_models'] = {
            'status': 'skipped',
            'reason': 'missing_data'
        }


    # Save training metrics to files
    end_time = datetime.now()
    training_metrics['end_time'] = end_time.isoformat()
    training_metrics['duration_seconds'] = (end_time - start_time).total_seconds()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Save metrics as JSON
    metrics_json_path = 'logs/training_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    logger.info(f"✅ Saved training metrics: {metrics_json_path}")
    
    # Save metrics as CSV (flattened version)
    metrics_csv_path = 'logs/training_metrics.csv'
    csv_data = {
        'timestamp': [training_metrics['start_time']],
        'duration_seconds': [training_metrics['duration_seconds']],
        'symbols': [','.join(training_metrics['symbols'])],
        'regime_samples': [training_metrics.get('regime_models', {}).get('total_samples', 0)],
        'price_status': [training_metrics.get('price_models', {}).get('status', 'unknown')],
        'rl_status': [training_metrics.get('rl_models', {}).get('status', 'unknown')],
        'rl_episodes': [training_metrics.get('rl_models', {}).get('num_episodes', 0)]
    }
    pd.DataFrame(csv_data).to_csv(metrics_csv_path, index=False)
    logger.info(f"✅ Saved training metrics: {metrics_csv_path}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    if "ML_ENABLED" not in os.environ:
        os.environ["ML_ENABLED"] = "true"
        print("ML_ENABLED ortam değişkeni 'true' olarak ayarlandı.")
    
    asyncio.run(main())
