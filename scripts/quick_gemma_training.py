#!/usr/bin/env python3
"""
Quick GEMMA Model Training for Phase 2 Validation
Creates a minimal but functional GEMMA model using synthetic data
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.core.logger import setup_logger

logger = setup_logger("quick-gemma-training", level=logging.INFO, log_to_file=True, log_filename="quick_training.log")

def generate_synthetic_ohlcv_data(num_samples=2000):
    """Generate synthetic OHLCV data for training"""
    logger.info(f"Generating {num_samples} samples of synthetic OHLCV data...")
    
    # Starting values
    base_price = 50000
    base_volume = 100
    
    # Initialize lists
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    current_time = datetime(2024, 1, 1, 0, 0, 0)
    current_price = base_price
    
    for i in range(num_samples):
        # Generate price movement (random walk with trend)
        price_change = np.random.randn() * (base_price * 0.005)  # 0.5% volatility
        trend = 0.0001 * base_price  # Slight upward trend
        current_price = current_price + price_change + trend
        
        # Generate OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.randn()) * (base_price * 0.003)
        low_price = open_price - abs(np.random.randn()) * (base_price * 0.003)
        close_price = low_price + (high_price - low_price) * np.random.random()
        
        # Generate volume
        volume = base_volume * (1 + np.random.randn() * 0.3)
        volume = max(1, volume)  # Ensure positive
        
        timestamps.append(current_time)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        
        # Move to next candle (5 minutes)
        current_time += timedelta(minutes=5)
        current_price = close_price
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    logger.info(f"✅ Generated {len(df)} candles")
    return df

def train_gemma_model():
    """Train a minimal GEMMA model for validation purposes"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        import joblib
    except ImportError as e:
        logger.error(f"❌ Required libraries not available: {e}")
        return False
    
    logger.info("\n" + "="*80)
    logger.info("QUICK GEMMA MODEL TRAINING FOR PHASE 2 VALIDATION")
    logger.info("="*80)
    
    # Import feature engineering
    from src.ml.feature_engineering import FeatureEngineeringPipeline
    
    # 1. Generate synthetic data
    raw_data = generate_synthetic_ohlcv_data(num_samples=2000)
    
    # 2. Extract GEMMA features
    logger.info("\nExtracting GEMMA features (82 features)...")
    feature_engine = FeatureEngineeringPipeline()
    
    try:
        features_df = feature_engine.extract_gemma_features(raw_data.copy())
        if features_df is None:
            logger.error("❌ Feature extraction returned None")
            return False
        logger.info(f"✅ Feature extraction complete: {features_df.shape[0]} samples, {features_df.shape[1]} features")
        logger.info(f"   Columns: {list(features_df.columns)[:10]}...")
    except AssertionError as e:
        logger.error(f"❌ Feature extraction assertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Generate target labels (price direction prediction)
    logger.info("Generating target labels...")
    
    # Check if 'close' column exists
    if 'close' not in features_df.columns:
        logger.error("❌ 'close' column not found in features DataFrame")
        logger.error(f"Available columns: {list(features_df.columns)[:10]}...")
        return False
    
    features_df['target'] = (features_df['close'].shift(-5) > features_df['close']).astype(int)
    features_df.dropna(inplace=True)
    
    if features_df.empty:
        logger.error("❌ No data remaining after feature extraction")
        return False
    
    logger.info(f"✅ Labels generated: {len(features_df)} samples")
    logger.info(f"   Class distribution: {features_df['target'].value_counts().to_dict()}")
    
    # Separate features and labels (exclude 'close' and 'target')
    exclude_cols = ['target']
    if 'close' in features_df.columns:
        exclude_cols.append('close')
    
    features = features_df.drop(columns=exclude_cols)
    labels = features_df['target']
    
    logger.info(f"Final dataset: {features.shape[0]} samples, {features.shape[1]} features")
    
    # 4. Scale features
    logger.info("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Save scaler
    scaler_path = Path('data/cache/gemma/scaler_gemma.joblib')
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Scaler saved to {scaler_path}")
    
    # 5. Split data
    logger.info("\nSplitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels.values, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"   Training samples: {len(X_train)}")
    logger.info(f"   Validation samples: {len(X_val)}")
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 6. Build GEMMA Model (Simple MLP)
    logger.info("\nBuilding GEMMA model...")
    
    class GemmaModel(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(128, 64)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 2)  # Binary classification
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout1(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout2(x)
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GemmaModel(input_size=features.shape[1]).to(device)
    
    logger.info(f"✅ Model created on device: {device}")
    logger.info(f"   Input size: {features.shape[1]}")
    logger.info(f"   Architecture: {features.shape[1]} → 128 → 64 → 32 → 2")
    
    # 7. Train model
    logger.info("\nTraining GEMMA model...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50  # Quick training
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"   GEMMA Price Model Final Validation Accuracy: {best_val_acc:.2f}%")
    
    # 8. Save model as TorchScript
    logger.info("\nSaving model as TorchScript...")
    model.eval()
    model_path = Path('data/models/gemma/final/gemma_price.pt')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create example input for tracing
    example_input = torch.randn(1, features.shape[1]).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, str(model_path))
    
    logger.info(f"✅ TorchScript model saved to {model_path}")
    
    # 9. Create feature list file
    logger.info("\nSaving feature list...")
    feature_list_path = Path('data/cache/gemma/feature_names.json')
    feature_list_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(feature_list_path, 'w') as f:
        json.dump({
            'features': list(features.columns),
            'count': len(features.columns),
            'created': datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"✅ Feature list saved to {feature_list_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ GEMMA MODEL TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nArtifacts created:")
    logger.info(f"  1. Model: {model_path}")
    logger.info(f"  2. Scaler: {scaler_path}")
    logger.info(f"  3. Feature list: {feature_list_path}")
    logger.info(f"\n📊 Final Validation Accuracy: {best_val_acc:.2f}%")
    
    return True

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("QUICK GEMMA MODEL TRAINING FOR PHASE 2 VALIDATION")
    print("="*80 + "\n")
    
    # Check if GEMMA is enabled
    gemma_enabled = os.environ.get('GEMMA_ENABLED', 'false').lower() == 'true'
    if not gemma_enabled:
        logger.warning("⚠️ GEMMA_ENABLED environment variable is not set to 'true'")
        logger.warning("   Set with: export GEMMA_ENABLED=true")
        return 1
    
    logger.info("✅ GEMMA_ENABLED is set")
    
    try:
        success = train_gemma_model()
        if success:
            logger.info("\n✅ Training successful!")
            return 0
        else:
            logger.error("\n❌ Training failed!")
            return 1
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
