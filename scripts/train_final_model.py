"""
Final Model Training with Aggressive Class Weights
Fixes class imbalance issue for Bearish detection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.ml.models import SimpleLSTM, create_model_from_params

# Configuration
TUNING_RESULTS_DIR = Path('logs/tuning_results')
TRAINING_DATA_PATH = Path('data/cache/BTC-USDT_training_data.npz')
MODEL_OUTPUT_DIR = Path('data/models/final')
METRICS_OUTPUT_DIR = Path('logs/final_training')


def load_best_hyperparameters():
    """Load best hyperparameters from tuning results"""
    tuning_files = list(TUNING_RESULTS_DIR.glob('lstm_tuning_*.json'))
    if not tuning_files:
        raise FileNotFoundError("No tuning results found!")
    
    latest_file = max(tuning_files, key=lambda p: p.stat().st_mtime)
    print(f"✅ Loading hyperparameters from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results['best_params'], results


def load_training_data():
    """Load preprocessed training data"""
    print(f"✅ Loading training data from: {TRAINING_DATA_PATH}")
    data = np.load(TRAINING_DATA_PATH)
    X, y = data['X'], data['y']
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
    return X, y


def stratified_split(X, y, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
        shuffle=True
    )
    
    print("\n📊 Stratified Split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def calculate_aggressive_class_weights(y_train, device):
    """
    Calculate aggressive class weights to fix minority class detection
    
    Strategy:
    1. Calculate baseline balanced weights
    2. Amplify Bearish weight aggressively (3x-6x)
    3. Amplify Bullish weight moderately (1.5x-2x)
    4. Keep Neutral weight low
    
    Args:
        y_train: Training labels
        device: Torch device (cuda/cpu)
    
    Returns:
        Tensor of class weights [Bullish, Neutral, Bearish]
    """
    print("\n" + "="*70)
    print("⚖️  CALCULATING AGGRESSIVE CLASS WEIGHTS")
    print("="*70)
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    num_classes = len(unique)
    
    print(f"\n📊 Training Set Distribution:")
    class_names = ['Bullish', 'Neutral', 'Bearish']
    for cls_id, count in zip(unique, counts):
        pct = count / total_samples * 100
        print(f"   {class_names[cls_id]:10s}: {count:,} samples ({pct:.1f}%)")
    
    # ============================================================
    # BASELINE: Balanced weights (inverse frequency)
    # ============================================================
    baseline_weights = np.zeros(num_classes, dtype=np.float32)
    for cls_id, count in zip(unique, counts):
        baseline_weights[cls_id] = total_samples / (num_classes * count)
    
    print(f"\n⚖️  Baseline Weights (Balanced):")
    for cls_id in unique:
        print(f"   {class_names[cls_id]:10s}: {baseline_weights[cls_id]:.4f}")
    
    # ============================================================
    # AGGRESSIVE: Amplify minority classes
    # ============================================================
    aggressive_weights = baseline_weights.copy()
    
    # Amplification factors (TUNE THESE!)
    BULLISH_AMPLIFY = 1.75  # Bullish needs moderate boost
    BEARISH_AMPLIFY = 2.60  # Bearish needs MASSIVE boost
    NEUTRAL_REDUCE = 0.80   # Neutral can afford slight reduction
    
    aggressive_weights[0] *= BULLISH_AMPLIFY  # Bullish
    aggressive_weights[1] *= NEUTRAL_REDUCE   # Neutral
    aggressive_weights[2] *= BEARISH_AMPLIFY  # Bearish
    
    print(f"\n🚀 Aggressive Weights (Amplified):")
    for cls_id in unique:
        boost = aggressive_weights[cls_id] / baseline_weights[cls_id]
        print(f"   {class_names[cls_id]:10s}: {aggressive_weights[cls_id]:.4f} ({boost:.2f}x)")
    
    # ============================================================
    # CONVERT TO TORCH TENSOR
    # ============================================================
    class_weights_tensor = torch.tensor(aggressive_weights, dtype=torch.float32).to(device)
    
    print(f"\n✅ Class weights ready on device: {device}")
    print(f"   Weights tensor: {class_weights_tensor}")
    
    return class_weights_tensor


def train_final_model(X_train, y_train, params, class_weights):
    """Train final model with aggressive class weights"""
    print("\n" + "="*70)
    print("🚀 TRAINING FINAL MODEL (AGGRESSIVE CLASS WEIGHTS)")
    print("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Create model
    model = create_model_from_params(params)
    model = model.to(device)
    
    print(f"\n📐 Model Architecture:")
    print(f"   Input: {params['input_size']} features")
    print(f"   Hidden: {params['hidden_size']}")
    print(f"   Layers: {params['num_layers']}")
    print(f"   Dropout: {params['dropout']}")
    print(f"   Output: {params['num_classes']} classes")
    
    # ============================================================
    # CRITICAL: Loss with Class Weights
    # ============================================================
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n🎯 Loss Function: CrossEntropyLoss with class weights")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Prepare data
    from torch.utils.data import TensorDataset, DataLoader
    
    # Split train into train/val (80/20)
    val_split = int(len(X_train) * 0.8)
    X_t, X_v = X_train[:val_split], X_train[val_split:]
    y_t, y_v = y_train[:val_split], y_train[val_split:]
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_t),
        torch.LongTensor(y_t)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_v),
        torch.LongTensor(y_v)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
    
    # Training loop
    num_epochs = 50
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n🔄 Training for up to {num_epochs} epochs...")
    print(f"   Early stopping patience: {patience}")
    print(f"   Batch size: {params['batch_size']}")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model on hold-out test set"""
    print("\n" + "="*70)
    print("📊 EVALUATING ON HOLD-OUT TEST SET")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
    
    # Calculate metrics
    accuracy = (predicted == y_test).mean()
    
    # Per-class accuracy
    class_names = ['Bullish', 'Neutral', 'Bearish']
    
    print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📊 Per-Class Performance:")
    
    per_class_acc = {}
    for class_id in sorted(np.unique(y_test)):
        mask = y_test == class_id
        class_acc = (predicted[mask] == y_test[mask]).mean()
        count = mask.sum()
        per_class_acc[class_names[class_id]] = float(class_acc)
        print(f"  {class_names[class_id]}: {class_acc:.4f} ({count} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, predicted)
    
    print("\n📊 Confusion Matrix:")
    print(cm)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, predicted, target_names=class_names))
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_acc
    }


def save_model_torchscript(model, params, metrics, metadata):
    """Save model using TorchScript"""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Set model to eval mode
    model.eval()
    model.cpu()  # Move to CPU for saving
    
    # Create dummy input
    dummy_input = torch.randn(1, params['input_size'])
    
    try:
        # Trace model
        print(f"\n🔄 Tracing model with input shape: {dummy_input.shape}")
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Verify
        print("🔍 Verifying traced model...")
        with torch.no_grad():
            original_out = model(dummy_input)
            traced_out = traced_model(dummy_input)
            
            if torch.allclose(original_out, traced_out, rtol=1e-3):
                print("✅ Traced model verification PASSED (outputs match)")
            else:
                max_diff = (original_out - traced_out).abs().max().item()
                print(f"⚠️  WARNING: Traced model outputs differ (max diff: {max_diff:.6f})")
                
                if max_diff > 1e-2:
                    raise ValueError(f"Traced model verification FAILED: max diff {max_diff:.6f}")
        
        # Save with timestamp
        script_path = MODEL_OUTPUT_DIR / f'lstm_final_{timestamp}.ptc'
        traced_model.save(str(script_path))
        print(f"✅ TorchScript model saved: {script_path}")
        
        # Also save as "latest"
        latest_path = MODEL_OUTPUT_DIR / 'lstm_final_latest.ptc'
        traced_model.save(str(latest_path))
        print(f"✅ Latest model saved: {latest_path}")
        
    except Exception as e:
        print(f"❌ Error tracing/saving model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save metadata
    metadata_path = METRICS_OUTPUT_DIR / f'metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'hyperparameters': params,
            'test_metrics': metrics,
            'metadata': metadata,
            'timestamp': timestamp,
            'format': 'torchscript'
        }, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")
    
    # Also save as "latest" metadata
    latest_metadata_path = METRICS_OUTPUT_DIR / 'metadata_latest.json'
    with open(latest_metadata_path, 'w') as f:
        json.dump({
            'hyperparameters': params,
            'test_metrics': metrics,
            'metadata': metadata,
            'timestamp': timestamp,
            'format': 'torchscript'
        }, f, indent=2)
    print(f"✅ Latest metadata saved: {latest_metadata_path}")
    
    return script_path, metadata_path


def main():
    print("="*70)
    print("🚀 FINAL MODEL TRAINING (AGGRESSIVE CLASS WEIGHTS FIX)")
    print("="*70)
    print(f"⏰ Timestamp: {datetime.utcnow().isoformat()}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load best hyperparameters
    best_params, tuning_results = load_best_hyperparameters()
    
    print("\n📋 Best Hyperparameters:")
    for key, value in best_params.items():
        if key not in ['input_size', 'num_classes', 'class_weights']:
            print(f"  {key}: {value}")
    
    # Load training data
    X, y = load_training_data()
    
    # Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    
    # ============================================================
    # CRITICAL: Calculate Aggressive Class Weights
    # ============================================================
    class_weights = calculate_aggressive_class_weights(y_train, device)
    
    # Update params with actual values
    best_params['input_size'] = X.shape[1]
    best_params['num_classes'] = len(np.unique(y))
    
    # Train model with aggressive class weights
    model = train_final_model(X_train, y_train, best_params, class_weights)
    
    # Evaluate on hold-out
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Prepare metadata
    metadata = {
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'total_samples': int(len(X)),
        'num_features': int(X.shape[1]),
        'tuning_cv_score': float(tuning_results['cv_score']),
        'tuning_holdout_score': float(tuning_results['holdout_score']),
        'tuning_gap': float(tuning_results['gap']),
        'final_test_accuracy': float(test_metrics['accuracy']),
        'split_strategy': 'stratified',
        'export_format': 'torchscript',
        'class_weights_used': class_weights.cpu().tolist(),  # Save weights used
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Save model
    save_model_torchscript(model, best_params, test_metrics, metadata)
    
    print("\n" + "="*70)
    print("✅ FINAL MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"📊 Tuning CV Score: {tuning_results['cv_score']:.4f}")
    print(f"📊 Tuning Hold-out: {tuning_results['holdout_score']:.4f}")
    print(f"💾 Export Format: TorchScript (.ptc)")
    print("="*70)
    
    # ============================================================
    # SUCCESS CRITERIA CHECK
    # ============================================================
    bearish_acc = test_metrics['per_class_accuracy'].get('Bearish', 0)
    bullish_acc = test_metrics['per_class_accuracy'].get('Bullish', 0)
    
    print("\n" + "="*70)
    print("🎯 PRODUCTION READINESS CHECK")
    print("="*70)
    
    print(f"\n📊 Minority Class Performance:")
    print(f"   Bearish: {bearish_acc:.4f} ({bearish_acc*100:.2f}%)")
    print(f"   Bullish: {bullish_acc:.4f} ({bullish_acc*100:.2f}%)")
    
    # New criteria
    BEARISH_MIN = 0.50  # 50% minimum
    BULLISH_MIN = 0.60  # 60% minimum
    
    bearish_ok = bearish_acc >= BEARISH_MIN
    bullish_ok = bullish_acc >= BULLISH_MIN
    
    if bearish_ok and bullish_ok:
        print(f"\n✅ MODEL IS PRODUCTION READY!")
        print(f"   Bearish: {bearish_acc:.2%} >= {BEARISH_MIN:.0%} ✅")
        print(f"   Bullish: {bullish_acc:.2%} >= {BULLISH_MIN:.0%} ✅")
    else:
        print(f"\n⚠️  MODEL NEEDS FURTHER IMPROVEMENT")
        if not bearish_ok:
            print(f"   Bearish: {bearish_acc:.2%} < {BEARISH_MIN:.0%} ❌")
            print(f"   → Need to increase Bearish class weight further")
        if not bullish_ok:
            print(f"   Bullish: {bullish_acc:.2%} < {BULLISH_MIN:.0%} ❌")
            print(f"   → Need to increase Bullish class weight further")
    
    print("="*70)


if __name__ == '__main__':
    main()
