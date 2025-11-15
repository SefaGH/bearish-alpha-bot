"""Legacy 42-feature sistem için manifest oluştur"""

import json
from pathlib import Path
import joblib
import torch
import numpy as np

def analyze_legacy_system():
    """Mevcut sistemi analiz et ve manifest oluştur"""
    
    # 1. Scaler'dan feature count al
    scaler_path = Path("data/models/regime/scaler.pkl")
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        feature_count = scaler.n_features_in_
    else:
        feature_count = 42  # Default
    
    # 2. Legacy feature names (hardcoded from existing system)
    legacy_features = [
        "close", "volume", "high", "low", "open",
        "returns", "log_returns", "volatility",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "sma_20", "sma_50", "ema_12", "ema_26",
        "atr_14", "adx_14", "cci_20", "mfi_14",
        "obv", "vwap", "pivot", "r1", "s1",
        "stoch_k", "stoch_d", "williams_r",
        "momentum_10", "roc_10", "trix_15",
        "vortex_pos", "vortex_neg",
        "keltner_upper", "keltner_lower",
        "donchian_high", "donchian_low",
        "fractal_dim"
    ]
    
    # Feature count ile uyumlu hale getir
    if len(legacy_features) < feature_count:
        # Eksik feature'ları generic isimlerle doldur
        for i in range(len(legacy_features), feature_count):
            legacy_features.append(f"feature_{i}")
    elif len(legacy_features) > feature_count:
        legacy_features = legacy_features[:feature_count]
    
    # 3. Manifest oluştur
    manifest = {
        "version": "1.0-legacy",
        "created_at": datetime.now().isoformat(),
        "mode": "legacy",
        "feature_count": feature_count,
        "feature_names_ordered": legacy_features,
        
        # Model paths (absolute paths for legacy)
        "price_scaler_path": "data/models/regime/scaler.pkl",
        "regime_scaler_path": "data/models/regime/scaler.pkl",
        "regime_model_path": "data/models/regime/random_forest.pkl",
        "lstm_model_path": "data/models/regime/lstm_regime.pth",
        
        # RL configuration
        "rl_state_size": feature_count,
        "rl_model_path": "data/models/rl_agent_final.pth",
        
        # No feature selection in legacy
        "selected_features_price": list(range(feature_count)),
        "selected_features_regime": list(range(feature_count)),
        
        # Metadata
        "metadata": {
            "system": "legacy",
            "migration_ready": True,
            "validated": False
        }
    }
    
    # 4. Bundle klasörü oluştur ve kaydet
    bundle_path = Path("artifacts/legacy")
    bundle_path.mkdir(parents=True, exist_ok=True)
    
    manifest_path = bundle_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Legacy manifest created: {manifest_path}")
    print(f"   Feature count: {feature_count}")
    print(f"   Feature names: {legacy_features[:5]}...")
    
    return manifest_path

def validate_legacy_manifest():
    """Legacy manifest'i validate et"""
    manifest_path = Path("artifacts/legacy/manifest.json")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    errors = []
    warnings = []
    
    # Check scaler
    scaler_path = Path(manifest["price_scaler_path"])
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if scaler.n_features_in_ != manifest["feature_count"]:
            errors.append(f"Scaler feature mismatch: {scaler.n_features_in_} != {manifest['feature_count']}")
    else:
        warnings.append(f"Scaler not found: {scaler_path}")
    
    # Check RL model
    rl_path = Path(manifest["rl_model_path"])
    if rl_path.exists():
        checkpoint = torch.load(rl_path, map_location='cpu')
        # Check first layer input size
        if 'q_network_state_dict' in checkpoint:
            first_layer_weight = checkpoint['q_network_state_dict']['network.0.weight']
            input_size = first_layer_weight.shape[1]
            if input_size != manifest["rl_state_size"]:
                errors.append(f"RL input size mismatch: {input_size} != {manifest['rl_state_size']}")
    else:
        warnings.append(f"RL model not found: {rl_path}")
    
    if errors:
        print("❌ Validation FAILED:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Validation PASSED")
        
    if warnings:
        print("⚠️ Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    return len(errors) == 0

if __name__ == "__main__":
    analyze_legacy_system()
    validate_legacy_manifest()
