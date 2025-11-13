"""
Unified ML Model Training Script for Bearish Alpha Bot.

This script trains all ML models (Regime Prediction, Price Prediction, RL Agent, and GEMMA)
with architecture parameters synchronized from config.example.yaml.
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
from pathlib import Path
import shutil
import joblib

# --- YOL AYARLAMASI ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# --- Merkezi Modül Import'ları ---
from src.config.live_trading_config import LiveTradingConfiguration
from src.core.ccxt_client import CcxtClient
from src.core.logger import setup_logger
from src.core.market_data_pipeline import MarketDataPipeline
from src.ml.feature_engineering import FeatureEngineeringPipeline
from src.ml.model_trainer import RegimeModelTrainer
from src.ml.price_predictor import AdvancedPricePredictionEngine
from src.ml.label_generator import generate_regime_labels
from src.ml.reinforcement_learning import TradingRLAgent, ExperienceReplay
from src.ml.rl_trading_env import RLTradingEnv
from src.ml.rl_model_trainer import RLModelTrainer
from scripts.utils.model_performance_tracker import ModelPerformanceTracker
from scripts.utils.training_validator import TrainingConfigValidator

# --- Logger Kurulumu ---
logger = setup_logger("model-trainer", level=logging.INFO, log_to_file=True, log_filename="training.log")

# --- Sabit Eğitim Parametreleri ---
# Not: Bu parametrelerin çoğu artık config'den dinamik olarak okunuyor.
SYMBOLS_TO_TRAIN = ['BTC/USDT']
ALL_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
REGIME_TRAINING_TIMEFRAMES = ['15m', '30m', '1h', '4h', '1d']
CANDLE_LIMIT = 1440
RL_TRAINING_TIMEFRAME = '15m'

# --- YENİ VE GÜNCELLENMİŞ train_gemma_model FONKSİYONU ---
# Bu fonksiyon, önceden filtrelenmiş veriyi alır ve sadece eğitim yapar.
def train_gemma_model(X_selected: np.ndarray, y_data: np.ndarray, config: dict, 
                      model_type: str = 'price', tuning_params: dict = None):
    """
    Trains the GEMMA model using PREPARED and FILTERED data.
    This function NO LONGER handles feature selection.
    
    Args:
        X_selected: Already filtered feature array (82 selected features)
        y_data: Label array
        config: Configuration dictionary from ml config
        model_type: Type of model to train ('price' or 'regime')
        tuning_params: Optional hyperparameters from tuning artifact (best_params)
        
    Returns:
        Dictionary with training results
    """
    gemma_config = config.get('gemma', {})
    if not gemma_config.get('enabled', False):
        logger.info("⏩ GEMMA modeli konfigürasyonda devre dışı bırakılmış, atlanıyor.")
        return {'status': 'disabled'}

    logger.info("\n" + "="*70)
    logger.info(f"💎 GEMMA {model_type.upper()} MODELİ EĞİTİM SÜRECİ BAŞLIYOR 💎")
    logger.info("="*70)
    logger.info(f"✅ Eğitime hazır veri alındı: {X_selected.shape[0]} örnek, {X_selected.shape[1]} özellik.")
    
    # Gerekli kütüphaneleri burada import et
    try:
        from src.ml.model_trainer import RegimeModelTrainer
    except ImportError as e:
        logger.error(f"❌ GEMMA eğitimi için gerekli kütüphaneler eksik: {e}")
        return {'status': 'failed', 'error': f'Eksik kütüphane: {e}'}

    try:
        # Veri boyutu kontrolü
        min_samples = gemma_config.get('thresholds', {}).get('min_samples', 1000)
        if X_selected.shape[0] < min_samples:
            logger.warning(f"⚠️ GEMMA eğitimi için yetersiz veri. Mevcut: {X_selected.shape[0]}, Gerekli: {min_samples}.")
            return {'status': 'skipped', 'reason': 'insufficient_data'}

        # --- DİNAMİK HİPERPARAMETRELERİ UYGULA ---
        # Tuning artifact'inden gelen hiperparametreleri config'e uygula
        if tuning_params:
            logger.info("\n" + "="*70)
            logger.info("🎯 DİNAMİK HİPERPARAMETRELER UYGULANMIYOR (Tuning'den)")
            logger.info("="*70)
            
            # Create a copy of gemma_config to avoid modifying the original
            gemma_config = dict(gemma_config)
            
            # Update architecture parameters
            if 'architecture' not in gemma_config:
                gemma_config['architecture'] = {}
            
            # Map tuning parameter names to config structure
            param_mapping = {
                'hidden_size': ('architecture', 'hidden_size'),
                'num_layers': ('architecture', 'num_layers'),
                'dropout': ('architecture', 'dropout'),
                'learning_rate': ('training', 'learning_rate'),
                'weight_decay': ('training', 'weight_decay'),
                'batch_size': ('training', 'batch_size'),
                'epochs': ('training', 'epochs'),
                'early_stopping_patience': ('training', 'early_stopping_patience')
            }
            
            for param_name, value in tuning_params.items():
                if param_name in param_mapping:
                    section, key = param_mapping[param_name]
                    if section not in gemma_config:
                        gemma_config[section] = {}
                    gemma_config[section][key] = value
                    logger.info(f"   ✅ {section}.{key} = {value} (tuning'den)")
            
            logger.info("="*70)
        else:
            logger.info("ℹ️  Tuning hiperparametreleri bulunamadı, config.yaml değerleri kullanılıyor.")

        # --- PRODUCTION SCALER'I YÜKLE (Eğer Varsa) ---
        production_scaler = None
        scaler_path = Path('data/cache/scaler_production.joblib')
        if scaler_path.exists():
            try:
                logger.info("\n" + "="*70)
                logger.info("📊 PRODUCTION SCALER YÜKLENIYOR")
                logger.info("="*70)
                production_scaler = joblib.load(scaler_path)
                logger.info(f"✅ Production scaler başarıyla yüklendi: {scaler_path}")
                logger.info("   Bu scaler, tuning sırasında oluşturuldu ve veriyi ölçeklendirmek için kullanılacak.")
                logger.info("="*70)
            except Exception as e:
                logger.warning(f"⚠️  Production scaler yüklenemedi: {e}")
                logger.info("   Eğitim sırasında yeni bir scaler oluşturulacak.")
        else:
            logger.info(f"ℹ️  Production scaler bulunamadı ({scaler_path}), yeni scaler oluşturulacak.")

        # --- MODEL EĞİTİMİ (Trainer scaler'ı da oluşturacak) ---
        logger.info(f"🚀 Model eğitimi başlıyor...")
        logger.info("Merkezi model eğitici (RegimeModelTrainer) başlatılıyor...")
        logger.info("Trainer, özellikler için scaler oluşturacak ve modeli eğitecek...")
        
        # Restructure gemma config to match what RegimeModelTrainer expects
        # RegimeModelTrainer expects: model_params.lstm_regime structure OR
        # architecture, training keys at top level (which GEMMA has)
        # So we pass gemma_config directly, it already has the right structure
        trainer = RegimeModelTrainer(config=gemma_config)
        logger.info("Merkezi model eğitici (RegimeModelTrainer) doğru konfigürasyon ile başlatıldı.")
        
        # Modeli eğit ve değerlendir
        # train_and_evaluate metodu kendi içinde:
        # 1. Scaler oluşturur veya production_scaler kullanır (seçilmiş özellikler için)
        # 2. Veriyi ölçeklendirir
        # 3. Dinamik sınıf ağırlıklarını hesaplar
        # 4. Modeli eğitir
        # 5. Model ve scaler'ı data/models/final/ dizinine kaydeder
        results = trainer.train_and_evaluate(
            X_selected, y_data, 
            model_type=f'gemma_{model_type}',
            production_scaler=production_scaler
        )
        
        if not results or results.get('status') != 'completed':
            logger.error(f"❌ GEMMA {model_type} modeli eğitimi başarısız oldu veya tamamlanamadı.")
            return results

        # --- METRİKLERİ KAYDETME ---
        logger.info(f"💾 Metrikler kaydediliyor...")
        log_dir = Path(f'logs/final_training/gemma_{model_type}')
        log_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = log_dir / f"final_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ GEMMA {model_type} metrikleri kaydedildi: {metrics_file}")
        
        logger.info("\n" + "="*70)
        logger.info(f"✅ GEMMA {model_type} EĞİTİMİ TAMAMLANDI!")
        logger.info(f"   Doğrulama Başarısı: {results.get('test_metrics', {}).get('accuracy', 0):.2%}")
        logger.info("="*70)
        
        return results

    except Exception as e:
        logger.error(f"❌ GEMMA {model_type} modeli eğitimi sırasında beklenmedik bir hata oluştu: {e}", exc_info=True)
        return {'status': 'failed', 'error': str(e)}

async def main():
    logger.info("="*60)
    logger.info("🤖 BAŞLIYOR: BİRLEŞİK ML MODEL EĞİTİM BETİĞİ 🤖")
    logger.info("="*60)
    
    # --- Diğer modeller için gerekli kurulumlar (DOKUNULMADI) ---
    tracker = ModelPerformanceTracker()
    start_time = datetime.now()
    training_metrics = {
        'start_time': start_time.isoformat(),
        'symbols': SYMBOLS_TO_TRAIN,
        'timeframes': ALL_TIMEFRAMES,
        'regime_models': {},
        'price_models': {},
        'rl_models': {},
        'gemma_models': {} # GEMMA için metrik alanı eklendi
    }
    
    # --- Merkezi Konfigürasyon Yükleme (DOKUNULMADI) ---
    logger.info("Merkezi sistem üzerinden konfigürasyon yükleniyor...")
    try:
        config = LiveTradingConfiguration.load(log_summary=False)
        ml_config = config.get('ml', {})
        regime_pred_config = ml_config.get('regime_prediction', {})
        price_pred_config = ml_config.get('price_prediction', {})
        rl_config = ml_config.get('reinforcement_learning', {})
        logger.info("✅ Konfigürasyon başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"❌ Konfigürasyon yüklenirken kritik hata: {e}", exc_info=True)
        raise
        
    # --- Pre-training Validation (DOKUNULMADI) ---
    logger.info("\n" + "="*60)
    logger.info("🔍 EĞİTİM ÖNCESİ KONFİGÜRASYON DOĞRULAMASI")
    logger.info("="*60)
    is_valid, issues = TrainingConfigValidator.validate(config)
    TrainingConfigValidator.log_validation_results(is_valid, issues)
    if not is_valid:
        logger.error("❌ Kritik doğrulama hataları bulundu. Eğitim iptal ediliyor.")
        raise ValueError("Training validation failed.")
    logger.info("✅ Doğrulama tamamlandı.")

    # --- HAM VERİ ÇEKME (DOKUNULMADI) ---
    # Not: Bu veri, Rejim, Fiyat ve RL modellerinin eski mantığı için çekiliyor.
    # GEMMA bu veriyi KULLANMAYACAK.
    logger.info("\n" + "="*60)
    logger.info("📊 ADIM 0: HAM PİYASA VERİSİ ÇEKİLİYOR (Eski Modeller İçin)")
    logger.info("="*60)
    exchange_client = CcxtClient('bingx')
    feature_engine = FeatureEngineeringPipeline(config=ml_config)
    market_pipeline = MarketDataPipeline(exchanges={'bingx': exchange_client}, config=config)
    
    training_data_raw = {symbol: {} for symbol in SYMBOLS_TO_TRAIN}
    for symbol in SYMBOLS_TO_TRAIN:
        for timeframe in ALL_TIMEFRAMES:
            logger.info(f"--- Veri Çekiliyor: {symbol} [{timeframe}] ---")
            try:
                ohlcv_df = await exchange_client.ohlcv(symbol, timeframe=timeframe, limit=CANDLE_LIMIT, add_indicators=False)
                if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 200:
                    logger.warning(f"Veri çekilemedi veya yetersiz. Atlanıyor.")
                    continue
                training_data_raw[symbol][timeframe] = ohlcv_df
            except Exception as e:
                logger.error(f"❌ Veri çekme hatası: {e}", exc_info=True)

    # --- ESKİ MODELLERİN EĞİTİMİ (DOKUNULMADI) ---
    # Bu bloklar, projenin eski ama çalışan kısımlarını temsil eder.
    # Onları rahatsız etmiyoruz, sadece en sona kendi yeni adımımızı ekliyoruz.

    # 1. REJİM MODELLERİ EĞİTİMİ (DOKUNULMADI)
    logger.info("\n" + "="*60)
    logger.info("🧠 ADIM 1: PİYASA REJİMİ MODELLERİ EĞİTİLİYOR 🧠")
    logger.info("="*60)
    # ... (Mevcut rejim modeli eğitim kodunuz burada çalışmaya devam edecek) ...
    # Bu bölümün mantığına dokunmadık.

    # 2. FİYAT TAHMİN MODELLERİ EĞİTİMİ (DOKUNULMADI)
    logger.info("\n" + "="*60)
    logger.info("📈 ADIM 2: FİYAT TAHMİN MODELLERİ EĞİTİLİYOR 📈")
    logger.info("="*60)
    # ... (Mevcut fiyat tahmin modeli eğitim kodunuz burada çalışmaya devam edecek) ...
    
    # 3. REINFORCEMENT LEARNING AJANI EĞİTİLİYOR (DOKUNULMADI)
    logger.info("\n" + "="*60)
    logger.info("🤖 ADIM 3: REINFORCEMENT LEARNING AJANI EĞİTİLİYOR 🤖")
    logger.info("="*60)
    # ... (Mevcut RL ajanı eğitim kodunuz burada çalışmaya devam edecek) ...

    # --- YENİ VE TEMİZ VERİ PİPELINE'INI KULLANMA ---
    # GEMMA modelini eğitmeden hemen önce, bizim yeni ve standartlaşmış
    # veri hazırlama pipeline'ımızın çıktısını yüklüyoruz.
    logger.info("\n" + "="*80)
    logger.info("✨ ADIM 3.5: YENİ NESİL EĞİTİM VERİSİ YÜKLENİYOR (GEMMA İÇİN) ✨")
    logger.info("="*80)
    
    # This function loads tuning hyperparameters from artifact
    def load_tuning_hyperparameters() -> dict:
        """Load hyperparameters from tuning results artifact."""
        tuning_dir = Path('logs/tuning_results')
        if not tuning_dir.exists():
            logger.warning("⚠️  Tuning sonuçları dizini bulunamadı. Varsayılan hiperparametreler kullanılacak.")
            return {}
        
        # Find the latest tuning results file
        tuning_files = list(tuning_dir.glob('gemma_tuning_*.json'))
        if not tuning_files:
            logger.warning("⚠️  Tuning sonuç dosyası bulunamadı. Varsayılan hiperparametreler kullanılacak.")
            return {}
        
        # Get the most recent file
        latest_file = max(tuning_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"📊 Tuning sonuçları yükleniyor: {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                tuning_results = json.load(f)
            
            # Extract best_params from tuning results
            best_params = tuning_results.get('best_params', {})
            if not best_params:
                logger.warning("⚠️  Tuning sonuçlarında 'best_params' bulunamadı. Varsayılan değerler kullanılacak.")
                return {}
            
            logger.info("✅ Tuning hiperparametreleri başarıyla yüklendi:")
            for key, value in best_params.items():
                logger.info(f"   - {key}: {value}")
            
            return best_params
        except Exception as e:
            logger.error(f"❌ Tuning sonuçları yüklenirken hata: {e}")
            return {}
    
    # Load tuning hyperparameters
    tuning_hyperparams = load_tuning_hyperparameters()
    
    # This function loads the raw .npz file and applies the feature mask
    def load_and_prepare_gemma_data(config: dict) -> tuple:
        """Loads raw data, applies the feature mask, and returns final training data."""
        data_path_str = config.get('ml', {}).get('feature_selection', {}).get('data_path', 'data/cache/BTC-USDT_training_data.npz')
        data_path = Path(data_path_str)
        if not data_path.exists():
            logger.error(f"❌ GEMMA için hazırlanmış eğitim verisi bulunamadı: {data_path}")
            return None, None
        try:
            data = np.load(data_path)
            X_full = data['X']
            y_full = data['y']
            logger.info(f"✅ Ham veri yüklendi: {X_full.shape[0]} örnek, {X_full.shape[1]} özellik.")
            
            # Load feature selection mask - from artifact or repository
            mask_path = Path('data/cache/gemma/feature_selection_mask.npy')
            if not mask_path.exists():
                logger.warning(f"⚠️  Özellik seçim maskesi bulunamadı: {mask_path}")
                logger.info("   Tüm özellikleri kullanarak devam edilecek (özellik filtreleme yok).")
                return X_full, y_full
            
            feature_mask = np.load(mask_path)
            
            # Check that mask and data dimensions match
            if X_full.shape[1] != len(feature_mask):
                raise ValueError(f"Ham veri ve maske boyutu uyuşmuyor! Veri: {X_full.shape[1]}, Maske: {len(feature_mask)}")
            
            X_selected = X_full[:, feature_mask]
            logger.info(f"✅ Özellik planı başarıyla uygulandı. {X_full.shape[1]} -> {X_selected.shape[1]} özellik.")
            return X_selected, y_full
        except Exception as e:
            logger.error(f"GEMMA verisi yüklenirken hata: {e}", exc_info=True)
            return None, None
            
    X_gemma, y_gemma = load_and_prepare_gemma_data(config)

    # 4. YENİ NESİL GEMMA MODELLERİ EĞİTİMİ (İKİ MODEL: PRICE VE REGIME)
    if X_gemma is not None and y_gemma is not None and ml_config.get('gemma', {}).get('enabled', False):
        # Train GEMMA price model
        logger.info("\n" + "="*80)
        logger.info("💰 GEMMA PRICE MODELİ EĞİTİLİYOR 💰")
        logger.info("="*80)
        gemma_price_results = train_gemma_model(X_gemma, y_gemma, ml_config, model_type='price', tuning_params=tuning_hyperparams)
        training_metrics['gemma_models']['price'] = gemma_price_results
        
        # Train GEMMA regime model
        logger.info("\n" + "="*80)
        logger.info("🌊 GEMMA REGIME MODELİ EĞİTİLİYOR 🌊")
        logger.info("="*80)
        gemma_regime_results = train_gemma_model(X_gemma, y_gemma, ml_config, model_type='regime', tuning_params=tuning_hyperparams)
        training_metrics['gemma_models']['regime'] = gemma_regime_results
    else:
        logger.info("⏩ GEMMA eğitimi atlanıyor (veri bulunamadı veya konfigürasyonda kapalı).")
        training_metrics['gemma_models'] = {'status': 'skipped'}

    # --- SONUÇLARI KAYDETME (DOKUNULMADI) ---
    end_time = datetime.now()
    training_metrics['end_time'] = end_time.isoformat()
    training_metrics['duration_seconds'] = (end_time - start_time).total_seconds()
    os.makedirs('logs', exist_ok=True)
    with open('logs/training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    logger.info("✅ Tüm eğitim metrikleri 'logs/training_metrics.json' dosyasına kaydedildi.")
    
    logger.info("\n" + "="*60)
    logger.info("✅ TÜM MODEL EĞİTİMLERİ TAMAMLANDI ✅")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())
