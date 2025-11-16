#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/gemma/final"
LEGACY_DIR="${ROOT_DIR}/artifacts/legacy"
MODELS_DIR="${ROOT_DIR}/data/models/final"
FEATURE_LIST="${ROOT_DIR}/features/gemma/selected/gemma_price_selected_82.json"
REGIME_FEATURE_LIST="${ROOT_DIR}/features/gemma/selected/gemma_regime_selected_82.json"

PYTHON_CMD=(python)
if ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
  elif command -v py >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
  else
    log "Python interpreter not found in PATH" "ERROR"
    exit 1
  fi
fi

log() {
  local message="$1"
  local level="${2:-INFO}"
  printf '%s [%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "${message}"
}

copy_if_exists() {
  local src="$1"
  local dest="$2"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${dest}"
    log "Copied $(basename "${src}") -> ${dest}"
  else
    log "Missing expected file: ${src}" "WARN"
  fi
}

log "========================================"
log "SETTING UP GEMMA ARTIFACTS"
log "========================================"

log "Creating artifact directories..."
mkdir -p "${ARTIFACT_DIR}" "${LEGACY_DIR}"
log "Directories ready"

log "Copying GEMMA models..."
copy_if_exists "${MODELS_DIR}/gemma_price.pt" "${ARTIFACT_DIR}/gemma_price.pt"
copy_if_exists "${MODELS_DIR}/gemma_regime.pt" "${ARTIFACT_DIR}/gemma_regime.pt"

log "Copying scalers..."
for scaler in gemma_price_scaler.joblib gemma_regime_scaler.joblib; do
  copy_if_exists "${MODELS_DIR}/${scaler}" "${ARTIFACT_DIR}/${scaler}"
done

log "Setting up manifest..."
MANIFEST_PATH="${ARTIFACT_DIR}/manifest.json"
"${PYTHON_CMD[@]}" - "${MANIFEST_PATH}" "${ROOT_DIR}" "${FEATURE_LIST}" "${REGIME_FEATURE_LIST}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
price_features_path = Path(sys.argv[3])
regime_features_path = Path(sys.argv[4])

def load_feature_list(path: Path):
  if not path.exists():
    return []
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features") or []
    if isinstance(features, list):
      return [str(f) for f in features]
  except Exception:
    return []
  return []

price_features = load_feature_list(price_features_path)
regime_features = load_feature_list(regime_features_path)

combined_features = []
seen = set()
for name in price_features + regime_features:
  if name not in seen:
    combined_features.append(name)
    seen.add(name)

DEFAULT_COUNT = 82
if not combined_features:
  combined_features = [f"feature_{i}" for i in range(DEFAULT_COUNT)]

feature_index = {name: idx for idx, name in enumerate(combined_features)}

def map_indices(source):
  indices = [feature_index[name] for name in source if name in feature_index]
  return indices if indices else list(range(len(combined_features)))

selected_price = map_indices(price_features)
selected_regime = map_indices(regime_features)

feature_count = len(combined_features)

metadata = {
  "generated_at": datetime.now(timezone.utc).isoformat(),
  "feature_sources": {
    "price": str(price_features_path.relative_to(root_dir)) if price_features else None,
    "regime": str(regime_features_path.relative_to(root_dir)) if regime_features else None,
  },
  "notes": "Auto-generated via setup_gemma_artifacts.sh"
}

bundle_rel = manifest_path.parent.relative_to(root_dir)

manifest = {
  "bundle": str(bundle_rel).replace("\\", "/"),
  "version": "GEMMA-2.0.0",
  "mode": "gemma",
  "feature_count": feature_count,
  "feature_names_ordered": combined_features,
  "selected_features_price": selected_price,
  "selected_features_regime": selected_regime,
  "rl_state_size": feature_count,
  "active_features_path": str(price_features_path.relative_to(root_dir)) if price_features else None,
  "gemma_price_model_path": "gemma_price.pt",
  "gemma_price_scaler_path": "gemma_price_scaler.joblib",
  "price_model_path": "gemma_price.pt",
  "price_scaler_path": "gemma_price_scaler.joblib",
  "regime_model_path": "gemma_regime.pt",
  "regime_scaler_path": "gemma_regime_scaler.joblib",
  "metadata": metadata
}

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Manifest created at {manifest_path}")
PY
log "Manifest created at ${MANIFEST_PATH}"

log "Creating active bundle symlink..."
ARTIFACTS_ROOT="${ROOT_DIR}/artifacts"
pushd "${ARTIFACTS_ROOT}" >/dev/null
if [[ -L active || -d active ]]; then
  rm -rf active
fi
if ln -s gemma/final active 2>/dev/null; then
  log "Symlink created: artifacts/active -> gemma/final"
else
  log "Symlink creation failed; copying directory as fallback" "WARN"
  cp -a gemma/final active
fi
popd >/dev/null

log ""
log "Final artifact structure:"
ls -la "${ARTIFACTS_ROOT}"
ls -la "${ARTIFACT_DIR}"

log ""
log "GEMMA artifacts setup complete!"
