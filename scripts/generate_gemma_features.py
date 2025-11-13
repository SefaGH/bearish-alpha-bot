#!/usr/bin/env python3
# scripts/generate_gemma_features.py
"""
GEMMA Feature List Generator for Bearish Alpha Bot
Creates feature metadata for production deployment.

MODIFIED FOR MLOps (Option 3):
This script now reads the mask from analyze_features.py instead of
using a hard-coded exclusion list.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging
import sys # Eklendi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GemmaFeatureGenerator:
    """Generate and manage GEMMA feature lists for Bearish Alpha Bot"""

    def __init__(self):
        self.features_dir = Path('features/gemma')
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.repo_info = {
            'repository': 'github.com/SefaGH/bearish-alpha-bot',
            'author': 'SefaGH',
            'version': 'GEMMA-1.0.0'
        }
        
        # Analist script'inin ürettiği maskenin konumu
        self.analyst_mask_path = Path('data/cache/feature_selection_mask.npy')

    def generate_full_87_features(self) -> List[str]:
        """Generate complete 87-feature list matching FeatureEngineering class"""
        features = []
        
        # Price-based features (30)
        for period in [5, 10, 15, 20, 30]:
            features.extend([
                f"sma_{period}", 
                f"ema_{period}", 
                f"rsi_{period}", 
                f"stoch_k_{period}", 
                f"stoch_d_{period}", 
                f"williams_r_{period}"
            ])
        
        # Volume-based features (15)
        for period in [5, 10, 15]:
            features.extend([
                f"volume_sma_{period}", 
                f"volume_ratio_{period}", 
                f"obv_{period}", 
                f"mfi_{period}", 
                f"vwap_{period}"
            ])
        
        # Volatility features (20)
        for period in [10, 20]:
            features.extend([
                f"bb_upper_{period}", 
                f"bb_middle_{period}", 
                f"bb_lower_{period}", 
                f"bb_width_{period}", 
                f"bb_position_{period}", 
                f"atr_{period}", 
                f"volatility_{period}", 
                f"keltner_upper_{period}", 
                f"keltner_lower_{period}", 
                f"donchian_{period}"
            ])
        
        # Trend features (12)
        features.extend([
            "macd_line", 
            "macd_signal", 
            "macd_histogram", 
            "adx_14", 
            "plus_di_14", 
            "minus_di_14", 
            "cci_20", 
            "roc_10", 
            "momentum_10", 
            "trix_15", 
            "dpo_20", 
            "vortex_pos_14"
        ])
        
        # Market structure features (10)
        features.extend([
            "support_distance", 
            "resistance_distance", 
            "pivot_point", 
            "r1_level", 
            "s1_level", 
            "fib_38", 
            "fib_50", 
            "fib_62", 
            "trend_strength", 
            "market_phase"
        ])
        
        assert len(features) == 87, f"Expected 87, got {len(features)}"
        logger.info(f"✅ Generated {len(features)} features")
        return features

    def perform_feature_selection(self, features: List[str], importance_scores: np.ndarray = None) -> Tuple[List[str], np.ndarray]:
        """
        MODIFIED: Selects features by loading the mask from analyze_features.py.
        """
        # --- MLOps Çözümü: Analist maskesini yükle ---
        
        if not self.analyst_mask_path.exists():
            logger.error(f"Kritik Hata: 'analyze_features.py' tarafından üretilen maske bulunamadı.")
            logger.error(f"Beklenen dosya: {self.analyst_mask_path}")
            logger.error("Lütfen önce 'Feature Analysis & Selection' adımının çalıştığından emin olun.")
            sys.exit(1) # Hata ile çık

        logger.info(f"Analist maskesi bulundu, yükleniyor: {self.analyst_mask_path}")
        mask = np.load(self.analyst_mask_path)
        
        if len(mask) != len(features):
            logger.error(f"Maske boyutu ({len(mask)}) ile özellik listesi ({len(features)}) uyumsuz!")
            sys.exit(1) # Hata ile çık

        selected = [f for f, m in zip(features, mask) if m]
        logger.info(f"Analiz sonucuna göre {len(selected)} özellik seçildi.")
        
        return selected, mask

    def save_feature_configurations(self) -> Dict[str, str]:
        """Save all feature configurations for production"""
        paths = {}
        
        # Generate full feature list
        full_87 = self.generate_full_87_features()
        
        # Perform feature selection (Artık analist maskesini okuyor)
        selected_features, mask = self.perform_feature_selection(full_87)
        
        # --- Dinamik olarak ayarlandı ---
        selected_count = len(selected_features)
        excluded_features = [f for f in full_87 if f not in selected_features]
        excluded_count = len(excluded_features)
        selection_method_name = "variance_correlation_analysis" # Yöntemi doğru yazalım
        
        # Save full feature list
        full_path = self.features_dir / 'selected/gemma_full_87.json'
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_config = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "type": "full", 
            "count": 87, 
            "features": full_87
        }
        with open(full_path, 'w') as f: 
            json.dump(full_config, f, indent=2)
        paths['full'] = str(full_path)
        logger.info(f"✅ Saved full feature list: {full_path}")

        # Save selected features for price model (Dinamik isim ve içerik)
        price_path = self.features_dir / f'selected/gemma_price_selected_{selected_count}.json'
        price_config = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "type": "price_prediction", 
            "count": selected_count, 
            "selection_method": selection_method_name, 
            "features": selected_features
        }
        with open(price_path, 'w') as f: 
            json.dump(price_config, f, indent=2)
        paths['price'] = str(price_path)
        logger.info(f"✅ Saved price feature list: {price_path}")

        # Save selected features for regime model (Dinamik isim ve içerik)
        regime_path = self.features_dir / f'selected/gemma_regime_selected_{selected_count}.json'
        regime_config = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "type": "regime_prediction", 
            "count": selected_count, 
            "selection_method": selection_method_name, 
            "features": selected_features
        }
        with open(regime_path, 'w') as f: 
            json.dump(regime_config, f, indent=2)
        paths['regime'] = str(regime_path)
        logger.info(f"✅ Saved regime list: {regime_path}")

        # Save feature mask (Bu, CI'ın artifact yapacağı yoldur)
        mask_path = Path('data/cache/gemma/feature_selection_mask.npy')
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        paths['mask'] = str(mask_path)
        logger.info(f"✅ Saved feature mask: {mask_path}")

        # Save metadata (Dinamik içerik)
        metadata_path = self.features_dir / 'metadata/feature_metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "statistics": {
                "full_count": 87, 
                "selected_count": selected_count, 
                "excluded_count": excluded_count, 
                "excluded_features": excluded_features
            }, 
            "paths": paths
        }
        with open(metadata_path, 'w') as f: 
            json.dump(metadata, f, indent=2)
        paths['metadata'] = str(metadata_path)
        logger.info(f"✅ Saved metadata: {metadata_path}")
        
        return paths

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🧬 GEMMA Feature List Generator (MLOps Mode)")
    logger.info("="*70)
    
    generator = GemmaFeatureGenerator()
    generated_paths = generator.save_feature_configurations()
    
    print("\n" + "="*70)
    print("✅ Feature configuration complete! (Synched with analyst mask)")
    print("="*70)
    print(json.dumps(generated_paths, indent=2))
    print("="*70)
