#!/usr/bin/env bash
set -euo pipefail

declare -a PYTHON_CMD=()
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8

select_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_CMD=($PYTHON_BIN)
    return
  fi

  for candidate in python python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_CMD=($candidate)
      return
    fi
  done

  if command -v py >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
    return
  fi

  echo "[ERROR] No Python interpreter found on PATH." >&2
  exit 1
}

select_python
echo "[INFO] Using Python interpreter: ${PYTHON_CMD[*]}"

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
if ! "${PYTHON_CMD[@]}" scripts/prepare_training_data.py \
    --symbol "BTC/USDT" \
    --no-feature-selection; then
  echo "[ERROR] Data preparation failed!"
  exit 1
fi
echo "[INFO] Training data prepared"
echo

# Step 3: Create default GEMMA feature mask (82 features)
log_section "Creating GEMMA feature selection mask..."
"${PYTHON_CMD[@]}" - <<'PY'
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

select_count = min(82, feature_count)
if feature_count < 82:
  print(f"[WARN] Not enough features ({feature_count} < 82); selecting all available features instead.")

mask = np.zeros(feature_count, dtype=bool)
mask[:select_count] = True

mask_dir = Path("data/cache/gemma")
mask_dir.mkdir(parents=True, exist_ok=True)
mask_file = mask_dir / "feature_selection_mask.npy"
np.save(mask_file, mask)
print(f"[INFO] Feature mask saved: {mask_file}")
print(f"   Selected {mask.sum()} out of {len(mask)} features")

feature_names = [f"feature_{i}" for i in range(select_count)]
feature_config = {
  "version": "1.0-gemma",
  "total_features": select_count,
  "selected_features": feature_names,
  "features": feature_names,
}
feature_dir = Path("features/gemma/selected")
feature_dir.mkdir(parents=True, exist_ok=True)
feature_file = feature_dir / f"gemma_price_selected_{select_count}.json"
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
"${PYTHON_CMD[@]}" - <<'PY'
from pathlib import Path

config_path = Path("config/config.example.yaml")
backup_path = Path("config/config.example.yaml.gemma_backup")
if not config_path.exists():
  raise SystemExit("[WARN] GEMMA section not found in config: config/config.example.yaml missing")

backup_path.write_bytes(config_path.read_bytes())
text = config_path.read_text(encoding="utf-8")
lines = text.splitlines()

in_gemma = False
gemma_indent = None
enabled_updated = False

for index, line in enumerate(lines):
  stripped = line.lstrip()
  indent = len(line) - len(stripped)

  if stripped.startswith("gemma:") and (indent >= 0):
    in_gemma = True
    gemma_indent = indent
    continue

  if in_gemma:
    if indent <= gemma_indent and stripped and not stripped.startswith("#"):
      break
    if stripped.startswith("enabled:"):
      if "true" not in stripped:
        lines[index] = " " * indent + "enabled: true"
        enabled_updated = True
      else:
        enabled_updated = True
      break

if not enabled_updated:
  print("[WARN] GEMMA enabled flag not updated; please verify config structure.")
else:
  config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print("[INFO] GEMMA enabled in config")
PY
echo

# Step 6: Train all models including GEMMA
echo "[INFO] Training models (this may take 5-10 minutes)..."
export ML_ENABLED=true
export GEMMA_ENABLED=true
export ML_RL_TRAINING_MODE=true

if ! "${PYTHON_CMD[@]}" scripts/train_all_models.py; then
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
    ((models_found+=1))
  else
    echo "   [ERROR] ${model}.pt NOT FOUND"
  fi

  if [ -f "data/models/final/${model}_scaler.joblib" ]; then
    echo "   [OK] ${model}_scaler.joblib"
    ((models_found+=1))
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
