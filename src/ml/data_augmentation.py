"""
Data Augmentation for Time-Series Regime Classification

Provides:
- SMOTE (Synthetic Minority Oversampling Technique)
- Time-series jittering (Gaussian noise)
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataAugmentation:
    """Time-series data augmentation for improving model generalization"""
    
    def __init__(self):
        """Initialize data augmentation utilities"""
        try:
            from imblearn.over_sampling import SMOTE
            self.smote = SMOTE(
                sampling_strategy='not majority',
                k_neighbors=5,
                random_state=42
            )
            self.smote_available = True
            logger.info("✅ SMOTE initialized successfully")
        except ImportError:
            self.smote_available = False
            logger.warning("⚠️ imbalanced-learn not installed. SMOTE unavailable.")
    
    def augment_with_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to generate synthetic samples for minority classes"""
        if not self.smote_available:
            logger.warning("SMOTE not available, returning original data")
            return X, y
        
        logger.info("Applying SMOTE augmentation...")
        
        # Store original shape
        original_shape = X.shape
        is_3d = len(X.shape) == 3
        
        # Flatten if 3D
        if is_3d:
            N, seq_len, features = X.shape
            X_flat = X.reshape(N, -1)
        else:
            X_flat = X
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Original class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        # Apply SMOTE
        try:
            X_aug, y_aug = self.smote.fit_resample(X_flat, y)
            
            # Reshape back to 3D if needed
            if is_3d:
                X_aug = X_aug.reshape(-1, seq_len, features)
            
            # Log results
            unique_aug, counts_aug = np.unique(y_aug, return_counts=True)
            logger.info(f"Augmented class distribution: {dict(zip(unique_aug.tolist(), counts_aug.tolist()))}")
            logger.info(f"Total samples: {len(y)} → {len(y_aug)} (+{len(y_aug)-len(y)})")
            
            return X_aug, y_aug
            
        except Exception as e:
            logger.error(f"SMOTE failed: {e}. Returning original data.")
            return X, y
    
    def add_jittering(self, X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to time-series data"""
        logger.info(f"Adding jittering with noise_level={noise_level}...")
        
        noise = np.random.normal(0, noise_level, X.shape)
        X_jittered = X + noise
        
        logger.info(f"✅ Jittering applied to {X.shape[0]} samples")
        
        return X_jittered
    
    def augment_sequence_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        use_smote: bool = True,
        use_jittering: bool = True,
        jitter_noise: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple augmentation techniques"""
        logger.info("="*60)
        logger.info("🔄 DATA AUGMENTATION PIPELINE")
        logger.info(f"   Original samples: {len(X)}")
        logger.info(f"   SMOTE: {use_smote}")
        logger.info(f"   Jittering: {use_jittering} (noise={jitter_noise})")
        logger.info("="*60)
        
        X_aug, y_aug = X.copy(), y.copy()
        
        # 1. SMOTE
        if use_smote:
            X_aug, y_aug = self.augment_with_smote(X_aug, y_aug)
        
        # 2. Jittering
        if use_jittering:
            X_jittered = self.add_jittering(X_aug, noise_level=jitter_noise)
            X_aug = np.vstack([X_aug, X_jittered])
            y_aug = np.concatenate([y_aug, y_aug])
            logger.info(f"✅ Combined with jittered data: {len(X_aug)} total samples")
        
        logger.info("="*60)
        logger.info(f"✅ AUGMENTATION COMPLETE: {len(X)} → {len(X_aug)} samples")
        logger.info("="*60)
        
        return X_aug, y_aug
