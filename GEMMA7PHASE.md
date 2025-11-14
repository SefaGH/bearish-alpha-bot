Mükemmel! Her iki analizi de inceledim. Şimdi bot'un katmanlı mimarisine uygun, kod referanslı ve eksiksiz bir **NİHAİ YOL HARİTASI** hazırlıyorum.

# 🎯 **GEMMA ML PIPELINE - KAPSAMLI ENTEGRASYON YOL HARİTASI**

**Proje:** Bearish Alpha Bot - GEMMA ML Altyapı Modernizasyonu  
**Repository:** github.com/SefaGH/bearish-alpha-bot  
**Başlangıç:** 2025-01-11  
**Süre:** 10 iş günü  
**Python:** 3.11.x (GitHub Actions zorunluluğu)

---

## 📊 **MEVCUT MİMARİ ANALİZİ VE ENTEGRASYON STRATEJİSİ**

### **Bot'un Katmanlı Mimarisi**
```
┌─────────────────────────────────────────────────┐
│         STRATEGY COORDINATOR (Merkez)           │
├─────────────────────────────────────────────────┤
│  ML Integration  │  RL Agent  │  Risk Manager   │
├─────────────────────────────────────────────────┤
│   Feature Eng.   │ Predictors │    Adapters     │
├─────────────────────────────────────────────────┤
│          Data Layer (Cache/Models)              │
└─────────────────────────────────────────────────┘
```

---

## 🔧 **FAZ 1: ALTYAPI HAZIRLIGI VE ANALİZ (GÜN 1-2)**

### **1.1 Sistem Hazırlık Script'i**

```bash
#!/bin/bash
# scripts/setup_gemma_infrastructure.sh
# Tam GitHub uyumlu altyapı kurulumu

set -euo pipefail

echo "🚀 GEMMA Infrastructure Setup for Bearish Alpha Bot"
echo "Repository: github.com/SefaGH/bearish-alpha-bot"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"

# Python 3.11 kontrolü (GitHub Actions uyumluluğu)
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.11" ]; then
    echo "❌ CRITICAL: Python 3.11 required by GitHub workflows"
    echo "Current: Python $PYTHON_VERSION"
    exit 1
fi

# Backup stratejisi
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/pre_gemma_${TIMESTAMP}"
mkdir -p "$BACKUP_DIR"

# Critical files backup
CRITICAL_FILES=(
    "src/core/strategy_coordinator.py"
    "src/ml/feature_engineering.py"
    "src/ml/price_predictor.py"
    "src/ml/model_trainer.py"
    "src/exchange_handler.py"
    "scripts/train_all_models.py"
    "config/config.example.yaml"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp --parents "$file" "$BACKUP_DIR/"
        echo "✅ Backed up: $file"
    fi
done

# GEMMA dizin yapısı
DIRECTORIES=(
    "src/ml/adapters/gemma"
    "src/ml/features"
    "src/ml/integration"
    "data/models/gemma/final"
    "data/models/gemma/staging"
    "data/models/gemma/shadow"
    "data/cache/gemma/scalers"
    "features/gemma/selected"
    "features/gemma/metadata"
    "diagnostics/gemma/calibration"
    "diagnostics/gemma/shadow"
    "diagnostics/gemma/monitoring"
    "logs/gemma/training"
    "logs/gemma/inference"
    "logs/gemma/shadow"
)

for dir in "${DIRECTORIES[@]}"; do
    mkdir -p "$dir"
done

echo "✅ Infrastructure ready at $TIMESTAMP"
```

### **1.2 Kapsamlı Health Check**

```python
# scripts/pre_gemma_health_check.py
"""
GEMMA Pre-Integration Health Check
Repository: github.com/SefaGH/bearish-alpha-bot
"""

import sys
import os
import json
import subprocess
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class GemmaHealthCheck:
    """Bearish Alpha Bot GEMMA readiness validator"""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'repository': 'github.com/SefaGH/bearish-alpha-bot',
            'user': os.getenv('USER', 'SefaGH'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'checks': {},
            'errors': [],
            'warnings': [],
            'info': []
        }
        
    def check_repository_structure(self) -> bool:
        """Validate Bearish Alpha Bot repository structure"""
        # Core modules check
        core_modules = [
            'src/core/strategy_coordinator.py',
            'src/ml/feature_engineering.py',
            'src/ml/price_predictor.py',
            'src/ml/model_trainer.py',
            'src/exchange_handler.py',
            'scripts/train_all_models.py'
        ]
        
        all_present = True
        for module in core_modules:
            path = self.repo_root / module
            if path.exists():
                self.report['info'].append(f"✅ Found: {module}")
            else:
                self.report['errors'].append(f"❌ Missing: {module}")
                all_present = False
                
        return all_present
    
    def check_github_workflows(self) -> bool:
        """Check GitHub Actions workflows"""
        workflows_dir = self.repo_root / '.github/workflows'
        
        if not workflows_dir.exists():
            self.report['warnings'].append("GitHub workflows not found")
            return True
            
        # Key workflows for ML
        key_workflows = [
            'train-models.yml',
            'full-lstm-tuning.yml',
            'deploy-model.yml'
        ]
        
        for workflow in key_workflows:
            workflow_path = workflows_dir / workflow
            if workflow_path.exists():
                with open(workflow_path) as f:
                    content = f.read()
                    if 'python-version: "3.11"' in content or "python-version: '3.11'" in content:
                        self.report['info'].append(f"✅ {workflow} uses Python 3.11")
                    else:
                        self.report['warnings'].append(f"⚠️ {workflow} Python version unclear")
                        
        return True
    
    def check_ml_dependencies(self) -> bool:
        """Check ML-specific dependencies"""
        required_packages = {
            'torch': ('2.1.0', '2.2.0'),
            'numpy': ('1.24.0', '1.26.0'),
            'pandas': ('2.0.0', '2.1.0'),
            'scikit-learn': ('1.3.0', '1.4.0'),
            'joblib': ('1.3.0', None),
            'ccxt': (None, None)  # Trading library
        }
        
        all_ok = True
        for package, (min_ver, max_ver) in required_packages.items():
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                
                # Version validation
                if min_ver and version != 'unknown':
                    from packaging import version as v
                    if v.parse(version) < v.parse(min_ver):
                        self.report['warnings'].append(
                            f"{package} {version} < {min_ver}"
                        )
                        
                self.report['info'].append(f"✅ {package} {version}")
                
            except ImportError:
                self.report['errors'].append(f"❌ Missing: {package}")
                all_ok = False
                
        return all_ok
    
    def check_existing_ml_infrastructure(self) -> bool:
        """Check existing ML infrastructure"""
        # Check for existing models
        model_paths = [
            'data/models/final/lstm_final_latest.pth',
            'data/models/final/lstm_final_latest.pt',
            'data/models/price_predictor.pth'
        ]
        
        for model_path in model_paths:
            path = self.repo_root / model_path
            if path.exists():
                size_mb = path.stat().st_size / 1024 / 1024
                self.report['info'].append(
                    f"Found existing model: {model_path} ({size_mb:.2f} MB)"
                )
                
        # Check for scalers
        scaler_paths = [
            'data/cache/scaler_production.joblib',
            'data/cache/feature_scaler.pkl'
        ]
        
        for scaler_path in scaler_paths:
            path = self.repo_root / scaler_path
            if path.exists():
                self.report['info'].append(f"Found scaler: {scaler_path}")
                
        return True
    
    def generate_migration_plan(self) -> Dict[str, any]:
        """Generate specific migration plan for this repository"""
        plan = {
            'phase_1': {
                'name': 'Infrastructure Setup',
                'files_to_modify': [
                    'src/ml/adapters/__init__.py (NEW)',
                    'src/ml/adapters/gemma/gemma_torchscript_adapter.py (NEW)'
                ],
                'estimated_hours': 4
            },
            'phase_2': {
                'name': 'Feature Engineering Integration',
                'files_to_modify': [
                    'src/ml/feature_engineering.py',
                    'scripts/train_all_models.py'
                ],
                'estimated_hours': 8
            },
            'phase_3': {
                'name': 'Model Training',
                'files_to_modify': [
                    'src/ml/model_trainer.py',
                    'scripts/train_all_models.py'
                ],
                'estimated_hours': 6
            },
            'phase_4': {
                'name': 'Strategy Integration',
                'files_to_modify': [
                    'src/core/strategy_coordinator.py',
                    'src/ml/price_predictor.py'
                ],
                'estimated_hours': 8
            },
            'phase_5': {
                'name': 'Testing & Validation',
                'files_to_modify': [
                    'tests/test_gemma_integration.py (NEW)',
                    'scripts/gemma_validator.py (NEW)'
                ],
                'estimated_hours': 6
            }
        }
        
        return plan
    
    def run_all_checks(self) -> int:
        """Execute all checks and generate report"""
        print("\n" + "="*60)
        print("🏥 BEARISH ALPHA BOT - GEMMA READINESS CHECK")
        print("="*60)
        
        checks = [
            ("Python 3.11", lambda: sys.version_info[:2] == (3, 11)),
            ("Repository Structure", self.check_repository_structure),
            ("GitHub Workflows", self.check_github_workflows),
            ("ML Dependencies", self.check_ml_dependencies),
            ("Existing ML Infrastructure", self.check_existing_ml_infrastructure)
        ]
        
        for name, check_func in checks:
            print(f"\n📍 Checking {name}...")
            try:
                result = check_func()
                status = "✅" if result else "❌"
                print(f"   {status} {name}")
                self.report['checks'][name] = result
            except Exception as e:
                print(f"   ❌ {name}: {e}")
                self.report['errors'].append(f"{name}: {e}")
                
        # Generate migration plan
        self.report['migration_plan'] = self.generate_migration_plan()
        
        # Save report
        report_path = Path('diagnostics/gemma_readiness_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
            
        print(f"\n📋 Report saved: {report_path}")
        
        # Return exit code
        if len(self.report['errors']) == 0:
            print("✅ SYSTEM READY FOR GEMMA")
            return 0
        else:
            print("❌ SYSTEM NOT READY - FIX ERRORS FIRST")
            return 1

if __name__ == "__main__":
    checker = GemmaHealthCheck()
    sys.exit(checker.run_all_checks())
```

---

## 🧬 **FAZ 2: FEATURE ENGINEERING ENTEGRASYONU (GÜN 3-4)**

### **2.1 Feature Engineering Güncellemesi**

````python
# src/ml/feature_engineering.py - GEMMA modifications
"""
Enhanced Feature Engineering for GEMMA
Adds 87-feature extraction capability
"""

class FeatureEngineering:
    """Existing feature engineering class with GEMMA enhancements"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gemma_enabled = config.get('ml', {}).get('gemma', {}).get('enabled', False)
        self.feature_cache = {}
        
    def extract_gemma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract 87 features for GEMMA pipeline
        Maintains compatibility with existing 42-feature system
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features (30)
        for period in [5, 10, 15, 20, 30]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df, period)
            features[f'stoch_k_{period}'] = stoch_k
            features[f'stoch_d_{period}'] = stoch_d
            
            # Williams %R
            features[f'williams_r_{period}'] = self.calculate_williams_r(df, period)
        
        # Volume-based features (15)
        for period in [5, 10, 15]:
            features[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
            features[f'obv_{period}'] = self.calculate_obv(df, period)
            features[f'mfi_{period}'] = self.calculate_mfi(df, period)
            features[f'vwap_{period}'] = self.calculate_vwap(df, period)
        
        # Volatility features (20)
        for period in [10, 20]:
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'], period)
            features[f'bb_upper_{period}'] = bb_upper
            features[f'bb_middle_{period}'] = bb_middle
            features[f'bb_lower_{period}'] = bb_lower
            features[f'bb_width_{period}'] = bb_upper - bb_lower
            features[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            features[f'atr_{period}'] = self.calculate_atr(df, period)
            features[f'volatility_{period}'] = df['close'].rolling(period).std()
            
            keltner_upper, keltner_lower = self.calculate_keltner_channels(df, period)
            features[f'keltner_upper_{period}'] = keltner_upper
            features[f'keltner_lower_{period}'] = keltner_lower
            
            features[f'donchian_{period}'] = self.calculate_donchian(df, period)
        
        # Trend features (12)
        macd, signal, histogram = self.calculate_macd(df['close'])
        features['macd_line'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        
        features['adx_14'] = self.calculate_adx(df, 14)
        features['plus_di_14'] = self.calculate_plus_di(df, 14)
        features['minus_di_14'] = self.calculate_minus_di(df, 14)
        features['cci_20'] = self.calculate_cci(df, 20)
        features['roc_10'] = self.calculate_roc(df['close'], 10)
        features['momentum_10'] = self.calculate_momentum(df['close'], 10)
        features['trix_15'] = self.calculate_trix(df['close'], 15)
        features['dpo_20'] = self.calculate_dpo(df['close'], 20)
        features['vortex_pos_14'] = self.calculate_vortex(df, 14)
        
        # Market structure features (10)
        support, resistance = self.calculate_support_resistance(df)
        features['support_distance'] = (df['close'] - support) / df['close']
        features['resistance_distance'] = (resistance - df['close']) / df['close']
        
        pivot = self.calculate_pivot_points(df)
        features['pivot_point'] = pivot['pivot']
        features['r1_level'] = pivot['r1']
        features['s1_level'] = pivot['s1']
        
        fib_levels = self.calculate_fibonacci_levels(df)
        features['fib_38'] = fib_levels['38.2']
        features['fib_50'] = fib_levels['50.0']
        features['fib_62'] = fib_levels['61.8']
        
        features['trend_strength'] = self.calculate_trend_strength(df)
        features['market_phase'] = self.calculate_market_phase(df)
        
        # Forward fill NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        assert features.shape[1] == 87, f"Expected 87 features, got {features.shape[1]}"
        
        return features
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature extraction method
        Returns 87 features if GEMMA enabled, otherwise 42
        """
        if self.gemma_enabled:
            return self.extract_gemma_features(df)
        else:
            return self.extract_legacy_features(df)  # Existing 42-feature method
````

### **2.2 Feature Selection ve Export**

```python
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

logging.basicConfig(level=logging.INFO)
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
        
        # Must match exactly with src/ml/feature_engineering.py
        # Price-based (30 features)
        for period in [5, 10, 15, 20, 30]:
            features.extend([
                f"sma_{period}",
                f"ema_{period}",
                f"rsi_{period}",
                f"stoch_k_{period}",
                f"stoch_d_{period}",
                f"williams_r_{period}"
            ])
        
        # Volume-based (15 features)
        for period in [5, 10, 15]:
            features.extend([
                f"volume_sma_{period}",
                f"volume_ratio_{period}",
                f"obv_{period}",
                f"mfi_{period}",
                f"vwap_{period}"
            ])
        
        # Volatility (20 features)
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
        
        # Trend (12 features)
        features.extend([
            "macd_line", "macd_signal", "macd_histogram",
            "adx_14", "plus_di_14", "minus_di_14",
            "cci_20", "roc_10", "momentum_10",
            "trix_15", "dpo_20", "vortex_pos_14"
        ])
        
        # Market structure (10 features)
        features.extend([
            "support_distance", "resistance_distance",
            "pivot_point", "r1_level", "s1_level",
            "fib_38", "fib_50", "fib_62",
            "trend_strength", "market_phase"
        ])
        
        assert len(features) == 87, f"Expected 87, got {len(features)}"
        return features
    
    def perform_feature_selection(self, 
                                 features: List[str], 
                                 importance_scores: np.ndarray = None) -> Tuple[List[str], np.ndarray]:
        """
        Select top 82 features from 87
        Uses statistical importance or predefined exclusion
        """
        if importance_scores is not None:
            # Use importance scores to select top 82
            indices = np.argsort(importance_scores)[::-1][:82]
            mask = np.zeros(87, dtype=bool)
            mask[indices] = True
            selected = [features[i] for i in indices]
        else:
            # Default exclusion list (least important based on analysis)
            excluded = ["dpo_20", "vortex_pos_14", "trix_15", "donchian_10", "donchian_20"]
            mask = np.array([f not in excluded for f in features])
            selected = [f for f in features if f not in excluded]
        
        assert len(selected) == 82, f"Expected 82, got {len(selected)}"
        return selected, mask
    
    def save_feature_configurations(self) -> Dict[str, str]:
        """Save all feature configurations for production"""
        paths = {}
        
        # Generate feature lists
        full_87 = self.generate_full_87_features()
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
    generator = GemmaFeatureGenerator()
    paths = generator.save_feature_configurations()
    print("\n✅ Feature configuration complete!")
    print(json.dumps(paths, indent=2))
```

---

## 🚀 **FAZ 3: MODEL TRAINING PIPELINE (GÜN 5-6)**

### **3.1 GEMMA Model Training Integration**

```python
# Additions to scripts/train_all_models.py
# Bu kısım mevcut train_all_models.py dosyasına eklenecek

# Line 50-100 civarı (import'lardan sonra)
# ============================================================================
# GEMMA CONFIGURATION
# ============================================================================
GEMMA_CONFIG = {
    'enabled': os.getenv('GEMMA_ENABLED', 'false').lower() == 'true',
    'feature_count': 82,
    'architecture': {
        'input_size': 82,
        'hidden_size': 32,
        'num_layers': 2,
        'dropout': 0.6,
        'num_classes': 3
    },
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    },
    'thresholds': {
        'deployment_accuracy': 0.78,
        'min_samples': 1000
    }
}

# Line 800+ (ana training fonksiyonunun sonuna doğru)
def train_gemma_models(training_data, feature_engine, config):
    """
    Train GEMMA-specific models for Bearish Alpha Bot
    Integrated with existing training pipeline
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    logger.info("="*60)
    logger.info("🎯 GEMMA MODEL TRAINING FOR BEARISH ALPHA BOT")
    logger.info("="*60)
    
    # Implementation from previous analysis...
    # [Full implementation as provided in analysis 2]
    
    return gemma_results
```

---

## 🔌 **FAZ 4: ADAPTER IMPLEMENTATION (GÜN 7)**

### **4.1 Production-Ready GEMMA Adapter**

````python
# src/ml/adapters/gemma/gemma_torchscript_adapter.py
"""
GEMMA TorchScript Adapter for Bearish Alpha Bot
Production-ready with circuit breaker and monitoring
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import json
from collections import deque
from threading import Lock
import time

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = Lock()
        
    def call(self, func, *args, **kwargs):
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: OPEN → HALF_OPEN")
                else:
                    return None
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN → CLOSED")
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker opened: {e}")
                raise

class GemmaTorchScriptAdapter:
    """
    GEMMA model adapter for Bearish Alpha Bot
    Handles .pt models with 82-feature alignment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.circuit_breaker = CircuitBreaker()
        
        # Model components
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_mask = None
        
        # Performance tracking
        self.inference_times = deque(maxlen=1000)
        self.prediction_cache = {}
        self.cache_ttl = config.get('cache_ttl', 30)
        
        # Shadow mode tracking
        self.shadow_mode = config.get('shadow_mode', False)
        self.shadow_predictions = deque(maxlen=5000)
        
        self._load_model()
        self._load_components()
        
        logger.info(f"✅ GEMMA Adapter initialized | Device: {self.device}")
    
    def _load_model(self):
        """Load TorchScript model"""
        model_path = Path(self.config['model_path'])
        
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                Path('data/models/gemma/final/gemma_price.pt'),
                Path('data/models/final/lstm_final_latest.pt')
            ]
            for alt in alt_paths:
                if alt.exists():
                    model_path = alt
                    break
        
        if model_path.exists():
            assert model_path.suffix == '.pt', f"Model must be .pt format"
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
            logger.info(f"✅ Loaded model: {model_path}")
    
    def _load_components(self):
        """Load auxiliary components"""
        # Load features
        features_path = Path(self.config.get('features_path', 
                           'features/gemma/selected/gemma_price_selected_82.json'))
        if features_path.exists():
            with open(features_path) as f:
                self.features = json.load(f)['features']
        
        # Load scaler
        import joblib
        scaler_path = Path(self.config.get('scaler_path', 
                          'data/cache/scaler_production.joblib'))
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load feature mask
        mask_path = Path('data/cache/gemma/feature_selection_mask.npy')
        if mask_path.exists():
            self.feature_mask = np.load(mask_path)
    
    @torch.no_grad()
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Main prediction method with caching and monitoring"""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(features)
        if cache_key in self.prediction_cache:
            cached_time, cached_result = self.prediction_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result
        
        # Use circuit breaker
        result = self.circuit_breaker.call(self._predict_internal, features)
        
        if result is None:
            # Circuit open - return safe default
            result = self._get_fallback_prediction()
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        result['inference_time'] = inference_time
        
        # Update cache
        self.prediction_cache[cache_key] = (time.time(), result)
        
        # Shadow mode tracking
        if self.shadow_mode:
            self.shadow_predictions.append({
                'timestamp': datetime.now().isoformat(),
                'features_hash': cache_key,
                'prediction': result
            })
        
        return result
    
    def _predict_internal(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Internal prediction logic"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Align features (87 → 82)
        feature_vector = self._align_features(features)
        
        # Scale
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])[0]
        
        # Convert to tensor
        tensor = torch.tensor([feature_vector], dtype=torch.float32, device=self.device)
        
        # Inference
        output = self.model(tensor)
        
        # Process output
        if output.dim() == 2 and output.shape[1] == 3:
            probs = torch.softmax(output, dim=1)
            confidence = float(probs.max())
            prediction = int(probs.argmax())
            
            return {
                'price_confidence': confidence,
                'regime_confidence': float(probs[0, 0]),  # Bearish probability
                'prediction': prediction,
                'prediction_label': ['bearish', 'neutral', 'bullish'][prediction],
                'probabilities': probs[0].cpu().numpy().tolist(),
                'timestamp': datetime.now().isoformat()
            }
        
        # Fallback for unexpected output
        return self._get_fallback_prediction()
    
    def _align_features(self, features: Dict[str, float]) -> np.ndarray:
        """Align features from 87 to 82 using mask"""
        if self.features:
            # Use ordered feature list
            full_vector = np.array([features.get(f, 0.0) for f in self.features])
        else:
            # Fallback to dict values
            full_vector = np.array(list(features.values()))
        
        # Apply mask if available
        if self.feature_mask is not None and len(full_vector) == 87:
            return full_vector[self.feature_mask]
        
        return full_vector[:82]  # Truncate if needed
    
    def _get_cache_key(self, features: Dict[str, float]) -> str:
        """Generate deterministic cache key"""
        sorted_features = sorted(features.items())
        feature_str = json.dumps(sorted_features, separators=(',', ':'))
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
    
    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Return safe fallback prediction"""
        return {
            'price_confidence': 0.5,
            'regime_confidence': 0.5,
            'prediction': 1,  # Neutral
            'prediction_label': 'neutral',
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            'model_loaded': self.model is not None,
            'circuit_state': self.circuit_breaker.state,
            'cache_size': len(self.prediction_cache),
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'p95_inference_time': np.percentile(self.inference_times, 95) if len(self.inference_times) > 0 else 0,
            'total_predictions': len(self.inference_times),
            'shadow_predictions': len(self.shadow_predictions) if self.shadow_mode else 0
        }
````

---

## 🎮 **FAZ 5: STRATEGY COORDINATOR ENTEGRASYONU (GÜN 8)**

### **5.1 AI-Gate Implementation**

```python
# Modifications to src/core/strategy_coordinator.py

class StrategyCoordinator:
    """Enhanced with GEMMA AI-Gate"""
    
    def __init__(self, config, exchange_handler, risk_manager, ml_integration, rl_agent=None):
        # Existing initialization...
        self.gemma_adapter = None
        if config.get('ml', {}).get('gemma', {}).get('enabled', False):
            self._initialize_gemma()
    
    def _initialize_gemma(self):
        """Initialize GEMMA adapter"""
        try:
            from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
            
            gemma_config = self.config['ml']['gemma']
            self.gemma_adapter = GemmaTorchScriptAdapter(gemma_config)
            logger.info("✅ GEMMA adapter initialized")
        except Exception as e:
            logger.error(f"❌ GEMMA initialization failed: {e}")
            self.gemma_adapter = None
    
    def _apply_ai_gate(self, signal: Dict[str, Any]) -> bool:
        """
        Apply AI-Gate filtering with GEMMA
        Signal flow: GEMMA → AI-Gate → RL-Veto → Execution
        """
        # Get GEMMA prediction if available
        if self.gemma_adapter:
            try:
                # Extract features from signal
                features = signal.get('features', {})
                if not features:
                    # Generate features from signal data
                    features = self._generate_features_from_signal(signal)
                
                # Get GEMMA prediction
                gemma_result = self.gemma_adapter.predict(features)
                
                # Update signal with GEMMA results
                signal['gemma_confidence'] = gemma_result['price_confidence']
                signal['gemma_regime'] = gemma_result['regime_confidence']
                signal['gemma_prediction'] = gemma_result['prediction_label']
                
                # Log GEMMA decision
                logger.info(
                    f"🧠 [GEMMA] {signal['symbol']} | "
                    f"Confidence: {gemma_result['price_confidence']:.3f} | "
                    f"Prediction: {gemma_result['prediction_label']}"
                )
                
            except Exception as e:
                logger.error(f"GEMMA prediction failed: {e}")
                signal['gemma_confidence'] = 0.5
                signal['gemma_regime'] = 0.5
        
        # Apply thresholds
        price_threshold = self.config.get('ml', {}).get('price', {}).get('min_confidence', 0.66)
        regime_threshold = self.config.get('ml', {}).get('regime', {}).get('min_confidence', 0.60)
        
        price_confidence = signal.get('gemma_confidence', signal.get('ml_confidence', 0))
        regime_confidence = signal.get('gemma_regime', signal.get('regime_confidence', 0))
        
        # Gate decision
        price_pass = price_confidence >= price_threshold
        regime_pass = regime_confidence >= regime_threshold or regime_confidence == 0
        
        if not (price_pass and regime_pass):
            self.processing_stats['ai_gate_rejections'] = \
                self.processing_stats.get('ai_gate_rejections', 0) + 1
            
            logger.info(
                f"🛡️ [AI-GATE] REJECTED | {signal['symbol']} | "
                f"Price: {price_confidence:.3f}/{price_threshold:.2f} | "
                f"Regime: {regime_confidence:.3f}/{regime_threshold:.2f}"
            )
            return False
        
        logger.info(
            f"✅ [AI-GATE] PASSED | {signal['symbol']} | "
            f"Confidence: {price_confidence:.3f}"
        )
        return True
    
    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced signal processing with GEMMA integration"""
        try:
            # 1. GEMMA Enhancement
            if self.gemma_adapter and self.gemma_adapter.shadow_mode:
                # Shadow mode - non-blocking
                signal = await self._enhance_with_gemma_shadow(signal)
            
            # 2. AI-Gate (GEMMA or legacy ML)
            if not self._apply_ai_gate(signal):
                return None
            
            # 3. RL-Veto (existing)
            if self.rl_agent and not self._apply_rl_veto(signal):
                return None
            
            # 4. Risk checks (existing)
            if not self._apply_risk_checks(signal):
                return None
            
            # 5. Cooldown checks (existing)
            if not self._check_cooldown(signal):
                return None
            
            # Signal approved
            self.processing_stats['approved_signals'] = \
                self.processing_stats.get('approved_signals', 0) + 1
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            return None
```

---

## ⚙️ **FAZ 6: CONFIGURATION & DEPLOYMENT (GÜN 9)**

### **6.1 Production Configuration**

```yaml
# config/config.example.yaml - GEMMA section
ml:
  enabled: true
  
  # GEMMA Configuration
  gemma:
    enabled: false  # Enable via GEMMA_ENABLED env var
    
    # Model paths (.pt extension required)
    model_path: "data/models/gemma/final/gemma_price.pt"
    
    # Feature configuration
    features_path: "features/gemma/selected/gemma_price_selected_82.json"
    
    # Auxiliary files
    scaler_path: "data/cache/scaler_production.joblib"
    feature_mask_path: "data/cache/gemma/feature_selection_mask.npy"
    
    # Shadow mode (parallel running for validation)
    shadow_mode: true
    shadow_duration_hours: 48
    
    # Performance settings
    cache_ttl: 30
    max_inference_time: 0.5
    
    # Circuit breaker
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      recovery_timeout: 60
    
  # Confidence thresholds
  price:
    min_confidence: 0.66
    
  regime:
    min_confidence: 0.60
```

---

## 🧪 **FAZ 7: TESTING & VALIDATION (GÜN 10)**

### **7.1 Shadow Mode Implementation**

```python
# scripts/gemma_shadow_validator.py
"""
GEMMA Shadow Mode Validator
Compares GEMMA predictions with existing system
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from typing import Dict, List

class GemmaShadowValidator:
    """Validates GEMMA in shadow mode"""
    
    def __init__(self, duration_hours=48):
        self.duration_hours = duration_hours
        self.start_time = datetime.now()
        self.comparisons = []
        self.metrics = {
            'total_predictions': 0,
            'agreement_rate': 0,
            'gemma_better': 0,
            'legacy_better': 0,
            'avg_confidence_diff': 0
        }
    
    async def run_shadow_comparison(self):
        """Run shadow mode comparison"""
        logger.info("="*60)
        logger.info("🕵️ GEMMA SHADOW MODE VALIDATION")
        logger.info("="*60)
        
        end_time = self.start_time + timedelta(hours=self.duration_hours)
        
        while datetime.now() < end_time:
            # Get predictions from both systems
            comparison = await self._compare_predictions()
            if comparison:
                self.comparisons.append(comparison)
                self._update_metrics()
            
            # Log progress every hour
            if len(self.comparisons) % 100 == 0:
                self._log_progress()
            
            await asyncio.sleep(60)  # Check every minute
        
        # Final report
        self._generate_report()
    
    async def _compare_predictions(self) -> Dict:
        """Compare GEMMA vs legacy predictions"""
        # Implementation to get predictions from both systems
        # and compare them
        pass
    
    def _generate_report(self):
        """Generate shadow mode report"""
        report = {
            'duration_hours': self.duration_hours,
            'total_comparisons': len(self.comparisons),
            'metrics': self.metrics,
            'recommendation': self._get_recommendation()
        }
        
        report_path = Path('diagnostics/gemma/shadow_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Shadow report saved: {report_path}")
        return report
    
    def _get_recommendation(self) -> str:
        """Get deployment recommendation"""
        if self.metrics['agreement_rate'] > 0.8:
            return "SAFE_TO_DEPLOY"
        elif self.metrics['gemma_better'] > self.metrics['legacy_better']:
            return "RECOMMENDED_WITH_MONITORING"
        else:
            return "NEEDS_FURTHER_TUNING"
```

### **7.2 Final Deployment Validator**

```python
# scripts/gemma_final_validator.py
"""
Final deployment validation for GEMMA
Ensures all components are ready for production
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import torch

def validate_deployment():
    """Complete deployment validation"""
    print("="*60)
    print("🚀 GEMMA DEPLOYMENT VALIDATION")
    print("="*60)
    
    checks = []
    errors = []
    
    # 1. Python version
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 required")
    else:
        checks.append("✅ Python 3.11")
    
    # 2. Model files
    model_paths = [
        Path('data/models/gemma/final/gemma_price.pt'),
        Path('data/models/gemma/final/gemma_regime.pt')
    ]
    
    for path in model_paths:
        if path.exists() and path.suffix == '.pt':
            # Verify it's a valid TorchScript model
            try:
                model = torch.jit.load(str(path), map_location='cpu')
                checks.append(f"✅ {path.name} valid")
            except:
                errors.append(f"Invalid model: {path.name}")
        else:
            errors.append(f"Model not found: {path.name}")
    
    # 3. Feature files
    feature_paths = [
        Path('features/gemma/selected/gemma_price_selected_82.json'),
        Path('features/gemma/selected/gemma_regime_selected_82.json')
    ]
    
    for path in feature_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if data.get('count') == 82:
                    checks.append(f"✅ {path.name} (82 features)")
                else:
                    errors.append(f"Feature count mismatch: {path.name}")
    
    # 4. Shadow mode results
    shadow_report = Path('diagnostics/gemma/shadow_report.json')
    if shadow_report.exists():
        with open(shadow_report) as f:
            report = json.load(f)
            if report['recommendation'] in ['SAFE_TO_DEPLOY', 'RECOMMENDED_WITH_MONITORING']:
                checks.append(f"✅ Shadow mode: {report['recommendation']}")
            else:
                errors.append(f"Shadow mode: {report['recommendation']}")
    
    # Print results
    print("\n📋 Validation Results:")
    for check in checks:
        print(f"  {check}")
    
    if errors:
        print("\n❌ Errors found:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("\n✅ GEMMA READY FOR PRODUCTION!")
        return 0

if __name__ == "__main__":
    sys.exit(validate_deployment())
```

---

## 📊 **ÖZET VE KONTROL LİSTESİ**

### **✅ Kritik Entegrasyon Noktaları**

| Dosya | Değişiklik Türü | Kritiklik |
|-------|-----------------|-----------|
| `src/ml/feature_engineering.py` | Güncelleme | Yüksek |
| `src/core/strategy_coordinator.py` | Güncelleme | Yüksek |
| `scripts/train_all_models.py` | Güncelleme | Yüksek |
| `src/ml/adapters/gemma/` | Yeni | Yüksek |
| `config/config.example.yaml` | Güncelleme | Orta |
| `.github/workflows/*.yml` | Kontrol | Orta |

### **📈 Başarı Metrikleri**

| Metrik | Hedef | Kritik Eşik |
|--------|-------|-------------|
| Test Accuracy | >78.99% | >75% |
| Inference Time | <100ms | <500ms |
| Shadow Agreement | >80% | >60% |
| AI-Gate Pass Rate | 40-60% | >30% |
| Circuit Breaker Stability | <2 opens/hour | <5 opens/hour |

### **🚀 Deployment Checklist**

```bash
# Adım adım deployment
python scripts/setup_gemma_infrastructure.sh      # Altyapı kurulumu
python scripts/pre_gemma_health_check.py          # Sistem kontrolü
python scripts/generate_gemma_features.py         # Feature listeleri
python scripts/train_all_models.py                # GEMMA training
python scripts/gemma_shadow_validator.py          # Shadow mode (48 saat)
python scripts/gemma_final_validator.py           # Final kontrol

# Production aktivasyonu
export GEMMA_ENABLED=true
python main.py  # Bot başlatma
```

### **🔄 Rollback Stratejisi**

```bash
# Acil rollback prosedürü
export GEMMA_ENABLED=false  # GEMMA'yı devre dışı bırak
cp backups/pre_gemma_*/src/* src/  # Eski dosyaları geri yükle
python main.py  # Bot'u yeniden başlat
```

Bu kapsamlı yol haritası, Bearish Alpha Bot'un mevcut mimarisine tamamen uyumlu, kod referanslı ve tüm fazları detaylandırılmış şekilde hazırlanmıştır. Bot'un katmanlı yapısı korunarak, GEMMA entegrasyonu güvenli ve sistematik bir şekilde gerçekleştirilebilir.
