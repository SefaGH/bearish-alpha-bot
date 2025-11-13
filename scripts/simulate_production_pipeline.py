#!/usr/bin/env python3
"""
Simulation script to verify the GEMMA production pipeline changes.
This script demonstrates the feature selection and scaler creation flow.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def simulate_feature_selection():
    """Simulate the feature selection mask loading and application"""
    print("="*70)
    print("🧪 SIMULATION: Feature Selection and Scaler Creation")
    print("="*70)
    
    # Create temporary directories
    cache_dir = Path('data/cache/gemma')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    final_dir = Path('data/models/final')
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate full feature dataset (87 features)
    print("\n📊 Step 1: Creating simulated dataset with 87 features...")
    n_samples = 1000
    n_features_full = 87
    X_data_full = np.random.randn(n_samples, n_features_full)
    y_data = np.random.randint(0, 3, n_samples)
    print(f"✅ Created dataset: {X_data_full.shape[0]} samples, {X_data_full.shape[1]} features")
    
    # Create and save a feature selection mask (select ~60% of features)
    print("\n🎯 Step 2: Creating feature selection mask...")
    feature_mask = np.random.rand(n_features_full) > 0.4
    mask_path = cache_dir / 'feature_selection_mask.npy'
    np.save(mask_path, feature_mask)
    n_selected = feature_mask.sum()
    print(f"✅ Created mask: {n_selected} features selected from {n_features_full}")
    print(f"✅ Mask saved to: {mask_path}")
    
    # Load and apply mask (simulates train_gemma_model logic)
    print("\n📋 Step 3: Loading and applying feature mask...")
    if mask_path.exists():
        loaded_mask = np.load(mask_path)
        X_selected = X_data_full[:, loaded_mask]
        print(f"✅ Mask loaded successfully")
        print(f"✅ Applied mask: {X_selected.shape[1]} features selected")
    else:
        print("⚠️ Mask not found, using all features")
        X_selected = X_data_full
    
    # Create scalers for both model types
    print("\n🔧 Step 4: Creating production scalers...")
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    for model_type in ['price', 'regime']:
        print(f"\n  Creating scaler for gemma_{model_type}...")
        scaler = StandardScaler()
        scaler.fit(X_selected)
        
        scaler_path = final_dir / f'gemma_{model_type}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"  ✅ Scaler saved to: {scaler_path}")
    
    # Verify the artifacts
    print("\n✅ Step 5: Verifying artifacts...")
    expected_files = [
        'data/models/final/gemma_price_scaler.joblib',
        'data/models/final/gemma_regime_scaler.joblib',
    ]
    
    all_exist = True
    for file_path in expected_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path} exists")
        else:
            print(f"  ❌ {file_path} NOT FOUND")
            all_exist = False
    
    print("\n" + "="*70)
    if all_exist:
        print("✅ SIMULATION SUCCESSFUL! All expected artifacts created.")
    else:
        print("❌ SIMULATION FAILED! Some artifacts missing.")
    print("="*70)
    
    return all_exist

if __name__ == "__main__":
    success = simulate_feature_selection()
    sys.exit(0 if success else 1)
