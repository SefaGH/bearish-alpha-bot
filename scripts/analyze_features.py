"""
Feature Analysis Tool for ML Feature Quality Assessment

This tool analyzes feature quality by calculating variance and correlation,
and generates a feature selection mask based on configurable thresholds
from the main project config.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import yaml  # EKLENDİ: Merkezi konfigürasyon için
from scipy import stats

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- YENİ: Merkezi Konfigürasyon Yükleme ---
def load_config():
    """Loads the main YAML configuration file."""
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.example.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Kritik Hata: Konfigürasyon dosyası bulunamadı: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Konfigürasyon dosyası okunurken hata: {e}")
        sys.exit(1)

class FeatureAnalyzer:
    """Analyzes feature quality based on project configuration."""

    def __init__(self, config: Dict):
        """
        Initialize the Feature Analyzer with project config.
        """
        self.config = config.get('ml', {})
        self.fs_config = self.config.get('feature_selection', {})
        
        # --- DÜZELTİLDİ: Değerler artık config'den okunuyor ---
        self.data_path = Path(self.fs_config.get('data_path', 'data/cache/BTC-USDT_training_data.npz'))
        self.variance_threshold = self.fs_config.get('variance_threshold', 0.00005)
        self.correlation_threshold = self.fs_config.get('correlation_threshold', 0.005)
        
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.feature_names: List[str] = self.config.get('feature_engineering', {}).get('active_features', [])
        
        self.variances: Optional[np.ndarray] = None
        self.correlations: Optional[np.ndarray] = None
        self.feature_mask: Optional[np.ndarray] = None

        logger.info("FeatureAnalyzer başlatıldı. Eşik değerleri:")
        logger.info(f"  - Varyans Eşiği: {self.variance_threshold}")
        logger.info(f"  - Korelasyon Eşiği: {self.correlation_threshold}")

    def load_data(self) -> bool:
        """Loads training data from the NPZ file."""
        try:
            if not self.data_path.exists():
                logger.error(f"Veri dosyası bulunamadı: {self.data_path}")
                return False
            
            logger.info(f"{self.data_path} dosyasından veriler yükleniyor...")
            data = np.load(self.data_path)
            self.features = data['X']
            self.labels = data['y']
            
            # --- DÜZELTİLDİ: Özellik isimleri artık NPZ'den değil, config'den geliyor ---
            if not self.feature_names or len(self.feature_names) != self.features.shape[1]:
                logger.warning("Config'deki özellik isimleri veri ile uyuşmuyor! Jenerik isimler kullanılacak.")
                self.feature_names = [f"feature_{i}" for i in range(self.features.shape[1])]

            logger.info(f"✅ Veri başarıyla yüklendi: {self.features.shape[0]} örnek, {self.features.shape[1]} özellik.")
            return True
        except Exception as e:
            logger.error(f"Veri yüklenirken hata oluştu: {e}", exc_info=True)
            return False

    def run_full_analysis(self):
        """Performs variance, correlation, and feature selection."""
        logger.info("="*70)
        logger.info("📊 ÖZELLİK ANALİZİ BAŞLATILIYOR")
        logger.info("="*70)

        if not self.load_data():
            return

        self._analyze_variance()
        self._analyze_correlations()
        self._select_features()
        self.save_results()

    def _analyze_variance(self):
        """Analyzes variance for all features."""
        logger.info("\n--- Aşama 1: Varyans Analizi ---")
        self.variances = np.var(self.features, axis=0)
        low_var_count = np.sum(self.variances < self.variance_threshold)
        logger.info(f"Düşük varyanslı (< {self.variance_threshold}) özellik sayısı: {low_var_count}")

    def _analyze_correlations(self):
        """Analyzes Spearman correlation between features and labels."""
        logger.info("\n--- Aşama 2: Korelasyon Analizi ---")
        n_features = self.features.shape[1]
        self.correlations = np.zeros(n_features)
        
        for i in range(n_features):
            try:
                corr, _ = stats.spearmanr(self.features[:, i], self.labels)
                self.correlations[i] = corr if np.isfinite(corr) else 0.0
            except ValueError:
                self.correlations[i] = 0.0

        weak_corr_count = np.sum(np.abs(self.correlations) < self.correlation_threshold)
        logger.info(f"Zayıf korelasyonlu (< {self.correlation_threshold}) özellik sayısı: {weak_corr_count}")

    def _select_features(self):
        """Selects features based on variance and correlation thresholds."""
        logger.info("\n--- Aşama 3: Özellik Seçimi ---")
        variance_mask = self.variances >= self.variance_threshold
        correlation_mask = np.abs(self.correlations) >= self.correlation_threshold
        self.feature_mask = variance_mask & correlation_mask

        selected_count = np.sum(self.feature_mask)
        total_count = len(self.feature_mask)
        logger.info(f"✅ Seçim tamamlandı. Seçilen özellikler: {selected_count} / {total_count}")

    def save_results(self, output_dir: str = "data/cache"):
        """Saves the feature selection mask and metadata."""
        logger.info("\n--- Aşama 4: Sonuçları Kaydetme ---")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Maskeyi kaydet
            mask_file = output_path / "feature_selection_mask.npy"
            np.save(mask_file, self.feature_mask)
            logger.info(f"✅ Özellik seçim maskesi kaydedildi: {mask_file}")

            # Metadatayı kaydet
            metadata = {
                'n_features_original': len(self.feature_mask),
                'n_features_selected': int(np.sum(self.feature_mask)),
                'variance_threshold': self.variance_threshold,
                'correlation_threshold': self.correlation_threshold,
                'selected_features': [name for name, selected in zip(self.feature_names, self.feature_mask) if selected],
            }
            metadata_file = output_path / "feature_selection_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✅ Özellik seçim metadatası kaydedildi: {metadata_file}")

        except Exception as e:
            logger.error(f"Sonuçlar kaydedilirken hata oluştu: {e}", exc_info=True)

def main():
    """Main entry point for the feature analyzer script."""
    parser = argparse.ArgumentParser(description="Feature Analysis Tool")
    parser.add_argument('--select', action='store_true', help='Run selection and save mask.')
    parser.add_argument('--analyze', action='store_true', help='Run feature analysis.')
    parser.add_argument('--report', action='store_true', help='Generate a detailed report.')
    parser.add_argument('--variance-threshold', type=float, default=None, 
                       help='Threshold for variance (overrides config value).')
    parser.add_argument('--correlation-threshold', type=float, default=None,
                       help='Threshold for correlation (overrides config value).')
    
    args = parser.parse_args()
    
    project_config = load_config()
    analyzer = FeatureAnalyzer(config=project_config)
    
    # Override thresholds from command line if provided
    if args.variance_threshold is not None:
        analyzer.variance_threshold = args.variance_threshold
        logger.info(f"Varyans eşiği komut satırından güncellendi: {args.variance_threshold}")
    
    if args.correlation_threshold is not None:
        analyzer.correlation_threshold = args.correlation_threshold
        logger.info(f"Korelasyon eşiği komut satırından güncellendi: {args.correlation_threshold}")

    if args.select or args.analyze:
        analyzer.run_full_analysis()
    elif args.report:
        # Report functionality can be extended later
        logger.info("Rapor oluşturma özelliği henüz uygulanmadı.")
        parser.print_help()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
