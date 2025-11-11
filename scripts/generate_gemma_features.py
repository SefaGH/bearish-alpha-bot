#!/usr/bin/env python3
# scripts/generate_gemma_features.py
"""
GEMMA Feature List Generator for Bearish Alpha Bot
Creates feature metadata for production deployment
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging

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
        """Select top 82 features from 87"""
        if importance_scores is not None:
            # Use importance scores if provided
            indices = np.argsort(importance_scores)[::-1][:82]
            mask = np.zeros(87, dtype=bool)
            mask[indices] = True
            selected = [features[i] for i in indices]
            logger.info("Selected features based on importance scores")
        else:
            # Default: exclude 5 specific features
            excluded = ["dpo_20", "vortex_pos_14", "trix_15", "donchian_10", "donchian_20"]
            mask = np.array([f not in excluded for f in features])
            selected = [f for f in features if f not in excluded]
            logger.info(f"Selected features by excluding: {excluded}")
        
        assert len(selected) == 82, f"Expected 82, got {len(selected)}"
        logger.info(f"✅ Selected {len(selected)} features for production")
        return selected, mask

    def save_feature_configurations(self) -> Dict[str, str]:
        """Save all feature configurations for production"""
        paths = {}
        
        # Generate full feature list
        full_87 = self.generate_full_87_features()
        
        # Perform feature selection
        selected_82, mask = self.perform_feature_selection(full_87)

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

        # Save selected features for price model
        price_path = self.features_dir / 'selected/gemma_price_selected_82.json'
        price_config = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "type": "price_prediction", 
            "count": 82, 
            "selection_method": "f_classif", 
            "features": selected_82
        }
        with open(price_path, 'w') as f: 
            json.dump(price_config, f, indent=2)
        paths['price'] = str(price_path)
        logger.info(f"✅ Saved price feature list: {price_path}")

        # Save selected features for regime model
        regime_path = self.features_dir / 'selected/gemma_regime_selected_82.json'
        regime_config = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "type": "regime_prediction", 
            "count": 82, 
            "selection_method": "mutual_info_classif", 
            "features": selected_82
        }
        with open(regime_path, 'w') as f: 
            json.dump(regime_config, f, indent=2)
        paths['regime'] = str(regime_path)
        logger.info(f"✅ Saved regime feature list: {regime_path}")

        # Save feature mask
        mask_path = Path('data/cache/gemma/feature_selection_mask.npy')
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        paths['mask'] = str(mask_path)
        logger.info(f"✅ Saved feature mask: {mask_path}")

        # Save metadata
        metadata_path = self.features_dir / 'metadata/feature_metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.repo_info, 
            "created": datetime.now().isoformat(), 
            "statistics": {
                "full_count": 87, 
                "selected_count": 82, 
                "excluded_count": 5, 
                "excluded_features": [f for f in full_87 if f not in selected_82]
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
    logger.info("🧬 GEMMA Feature List Generator")
    logger.info("="*70)
    
    generator = GemmaFeatureGenerator()
    generated_paths = generator.save_feature_configurations()
    
    print("\n" + "="*70)
    print("✅ Feature configuration complete!")
    print("="*70)
    print(json.dumps(generated_paths, indent=2))
    print("="*70)
