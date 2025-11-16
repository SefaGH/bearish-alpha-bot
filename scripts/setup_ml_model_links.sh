#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINAL_DIR="${ROOT_DIR}/data/models/final"

log() {
  local message="$1"
  local level="${2:-INFO}"
  printf '%s [%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "${message}"
}

link_model() {
  local source_file="$1"
  local target_link="$2"
  if [[ ! -f "${source_file}" ]]; then
    log "Source model missing: ${source_file}" "WARN"
    return
  fi
  mkdir -p "$(dirname "${target_link}")"
  if ln -sf "${source_file}" "${target_link}" 2>/dev/null; then
    log "Linked ${target_link} -> ${source_file}"
  else
    cp -f "${source_file}" "${target_link}"
    log "Symlink unsupported, copied model to ${target_link}" "WARN"
  fi
}

log "========================================"
log "SETTING UP ML MODEL LINKS"
log "========================================"

for tf in 5m 15m 1h; do
  link_model "${FINAL_DIR}/gemma_regime.pt" "${ROOT_DIR}/data/models/regime/${tf}/model.pt"
  link_model "${FINAL_DIR}/gemma_price.pt" "${ROOT_DIR}/data/models/price/${tf}/model.pt"

done

log "Listing linked models:"
find "${ROOT_DIR}/data/models" -type l -name "model.pt" -print 2>/dev/null || true

log "ML model link setup complete"
