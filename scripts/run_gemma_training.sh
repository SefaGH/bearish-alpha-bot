#!/usr/bin/env bash
set -euo pipefail

log_section() {
  echo "========================================"
  echo "$1"
  echo "========================================"
}

echo "========================================"
echo "GEMMA MODEL TRAINING - LOCAL"
echo "========================================"
echo "Start: $(date -u +%Y-%m-%d_%H:%M:%S)"
echo

# Step 1: Create necessary directories
echo "[INFO] Creating directories..."
mkdir -p data/cache/gemma
mkdir -p data/models/final
mkdir -p features/gemma/selected
mkdir -p logs/final_training/gemma_price
mkdir -p logs/final_training/gemma_regime
mkdir -p logs/tuning_results
echo "[INFO] Directories ready"
echo

# Step 2: Prepare training data
echo "[INFO] Preparing training data for BTC/USDT..."
if ! python scripts/prepare_training_data.py \
    --symbol "BTC/USDT" \
    --no-feature-selection; then
  echo "[ERROR] Data preparation failed!"
  exit 1
fi
echo "[INFO] Training data prepared"
echo

# Step 3: Create default GEMMA feature mask (82 features)
log_section "Creating GEMMA feature selection mask..."
python - <<'PY'
import json
import numpy as np
from pathlib import Path

cache_file = Path("data/cache/BTC-USDT_training_data.npz")
if not cache_file.exists():
    raise SystemExit("Training data cache missing at data/cache/BTC-USDT_training_data.npz")

data = np.load(cache_file)
X = data["X"]
feature_count = X.shape[1]
print(f"Total features: {feature_count}")

if feature_count < 82:
    print(f"[WARN] Not enough features ({feature_count} < 82)")
    raise SystemExit(1)

mask = np.zeros(feature_count, dtype=bool)
mask[:82] = True

mask_dir = Path("data/cache/gemma")
mask_dir.mkdir(parents=True, exist_ok=True)
mask_file = mask_dir / "feature_selection_mask.npy"
np.save(mask_file, mask)
print(f"[INFO] Feature mask saved: {mask_file}")
print(f"   Selected {mask.sum()} out of {len(mask)} features")

feature_names = [f"feature_{i}" for i in range(82)]
feature_config = {
    "version": "1.0-gemma",
    "total_features": 82,
    "selected_features": feature_names,
    "features": feature_names,
}
feature_dir = Path("features/gemma/selected")
feature_dir.mkdir(parents=True, exist_ok=True)
feature_file = feature_dir / "gemma_price_selected_82.json"
feature_file.write_text(json.dumps(feature_config, indent=2))
print(f"[INFO] Feature config saved: {feature_file}")
PY
echo

# Step 4: Create mock tuning results (default hyperparameters)
echo "[INFO] Creating default tuning configuration..."
cat > logs/tuning_results/gemma_tuning_local.json <<'JSON'
{
  "best_params": {
    "hidden_size": 55,
    "num_layers": 3,
    "dropout": 0.3243,
    "learning_rate": 0.000133,
    "batch_size": 32,
    "epochs": 50,
    "early_stopping_patience": 10
  },
  "best_score": 0.70,
  "timestamp": "2025-11-15T17:54:00Z",
  "note": "Default hyperparameters for local training"
}
JSON
echo "[INFO] Tuning config created"
echo

# Step 5: Enable GEMMA in config
echo "[INFO] Enabling GEMMA in config..."
CONFIG_FILE="config/config.example.yaml"
BACKUP_FILE="${CONFIG_FILE}.gemma_backup"
python - <<PY
from pathlib import Path
import yaml

config_path = Path("config/config.example.yaml")
backup_path = Path("config/config.example.yaml.gemma_backup")
if not config_path.exists():
    raise SystemExit("[WARN] GEMMA section not found in config: config/config.example.yaml missing")

backup_path.write_bytes(config_path.read_bytes())
config = yaml.safe_load(config_path.read_text())

ml_config = config.get("ml", {})
if "gemma" not in ml_config:
    print("[WARN] GEMMA section not found in config")
    raise SystemExit(0)

gemma_config = ml_config["gemma"]
if isinstance(gemma_config, dict):
    gemma_config["enabled"] = True

config_path.write_text(yaml.safe_dump(config, sort_keys=False))
print("[INFO] GEMMA enabled in config")
PY
echo

# Step 6: Train all models including GEMMA
echo "[INFO] Training models (this may take 5-10 minutes)..."
export ML_ENABLED=true
export GEMMA_ENABLED=true
export ML_RL_TRAINING_MODE=true

if ! python scripts/train_all_models.py; then
  echo "[ERROR] Model training failed!"
  mv "${BACKUP_FILE}" "$CONFIG_FILE" 2>/dev/null || true
  exit 1
fi

# Step 7: Restore config
if [ -f "$BACKUP_FILE" ]; then
  mv "$BACKUP_FILE" "$CONFIG_FILE"
  echo "[INFO] Config restored"
fi
echo

# Step 8: Verify GEMMA models
log_section "Verifying GEMMA models..."
models_found=0
for model in gemma_price gemma_regime; do
  if [ -f "data/models/final/${model}.pt" ]; then
    echo "   [OK] ${model}.pt"
    ((models_found++))
  else
    echo "   [ERROR] ${model}.pt NOT FOUND"
  fi

  if [ -f "data/models/final/${model}_scaler.joblib" ]; then
    echo "   [OK] ${model}_scaler.joblib"
    ((models_found++))
  else
    echo "   [WARN] ${model}_scaler.joblib not found (optional)"
  fi
done

echo
if [ "$models_found" -ge 2 ]; then
  log_section "GEMMA TRAINING SUCCESSFUL!"
  echo "Models created:"
  ls -la data/models/final/gemma*.pt || true
  echo
  echo "Next steps:"
  echo "1. Test models: python scripts/test_gemma_models.py"
  echo "2. Activate GEMMA: ./scripts/activate_gemma.sh"
  touch gemma_trained.flag
else
  echo "[ERROR] Some GEMMA models are missing!"
  exit 1
fi

echo "End: $(date -u +%Y-%m-%d_%H:%M:%S)"
