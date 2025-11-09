"""
Final Model Training with Tuned Hyperparameters
Uses stratified split and best params from tuning results
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration
TUNING_RESULTS_DIR = Path('logs/tuning_results')
TRAINING_DATA_PATH = Path('data/cache/BTC-USDT_training_data.npz')
MODEL_OUTPUT_DIR = Path('data/models/final')
METRICS_OUTPUT_DIR = Path('logs/final_training')

def load_best_hyperparameters():
    """Load best hyperparameters from tuning results"""
    # Find latest tuning results
    tuning_files = list(TUNING_RESULTS_DIR.glob('lstm_tuning_*.json'))
    if not tuning_files:
        raise FileNotFoundError("No tuning results found!")
    
    latest_file = max(tuning_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading hyperparameters from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results['best_params'], results

def load_training_data():
    """Load preprocessed training data"""
    print(f"Loading training data from: {TRAINING_DATA_PATH}")
    data = np.load(TRAINING_DATA_PATH)
    X, y = data['X'], data['y']
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
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
    
    # Log distributions
    print("\nStratified Split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

class SimpleLSTM(nn.Module):
    """LSTM Model Architecture"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def train_final_model(X_train, y_train, params, class_weights):
    """Train final model with best hyperparameters"""
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    # Create model
    model = SimpleLSTM(
        input_size=params['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        dropout=params['dropout']
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
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
    
    print(f"\nTraining for up to {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
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
            print(f"Epoch {epoch+1}/{num_epochs}: "
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
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model on hold-out test set"""
    print("\n" + "="*70)
    print("EVALUATING ON HOLD-OUT TEST SET")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.numpy()
    
    # Calculate metrics
    accuracy = (predicted == y_test).mean()
    
    # Per-class accuracy
    from collections import Counter
    class_names = ['Bullish', 'Neutral', 'Bearish']
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nPer-Class Performance:")
    
    for class_id in sorted(np.unique(y_test)):
        mask = y_test == class_id
        class_acc = (predicted[mask] == y_test[mask]).mean()
        count = mask.sum()
        print(f"  {class_names[class_id]}: {class_acc:.4f} ({count} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, predicted)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predicted, target_names=class_names))
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {
            class_names[i]: float((predicted[y_test == i] == y_test[y_test == i]).mean())
            for i in sorted(np.unique(y_test))
        }
    }

def save_model(model, params, metrics, metadata):
    """Save model with all metadata"""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    
    # Save PyTorch model
    model_path = MODEL_OUTPUT_DIR / f'lstm_final_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': params,
        'metadata': metadata
    }, model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Save metrics
    metrics_path = METRICS_OUTPUT_DIR / f'metrics_{timestamp}.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'hyperparameters': params,
            'test_metrics': metrics,
            'metadata': metadata,
            'timestamp': timestamp
        }, f, indent=2)
    print(f"✅ Metrics saved: {metrics_path}")
    
    # Save as pickle for easy loading
    pickle_path = MODEL_OUTPUT_DIR / 'lstm_final_latest.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'model_state_dict': model.state_dict(),
            'hyperparameters': params,
            'metadata': metadata
        }, f)
    print(f"✅ Pickle saved: {pickle_path}")

def main():
    print("="*70)
    print("🚀 FINAL MODEL TRAINING")
    print("="*70)
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    # Load best hyperparameters
    best_params, tuning_results = load_best_hyperparameters()
    
    print("\nBest Hyperparameters:")
    for key, value in best_params.items():
        if key not in ['input_size', 'num_classes', 'class_weights']:
            print(f"  {key}: {value}")
    
    # Load training data
    X, y = load_training_data()
    
    # Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print(f"\nClass weights: {class_weights}")
    
    # Update params with actual values
    best_params['input_size'] = X.shape[1]
    best_params['num_classes'] = len(np.unique(y))
    best_params['class_weights'] = class_weights.tolist()
    
    # Train model
    model = train_final_model(X_train, y_train, best_params, class_weights)
    
    # Evaluate on hold-out
    test_metrics = evaluate_model(model, X_test, y_test)
    
    # Save everything
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
        'timestamp': datetime.utcnow().isoformat()
    }
    
    save_model(model, best_params, test_metrics, metadata)
    
    print("\n" + "="*70)
    print("✅ FINAL MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Tuning CV Score: {tuning_results['cv_score']:.4f}")
    print(f"Tuning Hold-out: {tuning_results['holdout_score']:.4f}")
    print("="*70)

if __name__ == '__main__':
    main()
