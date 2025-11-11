#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  local level="${2:-INFO}"
  printf '[%s] [%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "${level}" "$1"
}

require_python_311() {
  local candidates=()
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    candidates+=("${PYTHON_BIN}")
  fi
  candidates+=("python" "python3")

  local resolved=""
  local resolved_version=""
  for candidate in "${candidates[@]}"; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      local candidate_version
      candidate_version="$(${candidate} -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
      if [[ -n "${candidate_version}" ]]; then
        resolved="${candidate}"
        resolved_version="${candidate_version}"
        break
      fi
    fi
  done

  if [[ -z "${resolved}" ]]; then
    log "Python interpreter could not be located." "ERROR"
    exit 1
  fi

  if [[ "${resolved_version}" != "3.11" ]]; then
    log "Python 3.11 is required; detected ${resolved} -> ${resolved_version}." "ERROR"
    exit 1
  fi

  echo "${resolved}"
}

perform_backups() {
  local python_bin="$1"
  local timestamp
  timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  local backup_dir="${ROOT_DIR}/backups/pre_gemma_${timestamp}"
  mkdir -p "${backup_dir}"

  local critical_paths=(
    "pyproject.toml"
    "config"
    "src/ml"
    "scripts/train_all_models.py"
  )

  local copied=()
  for rel in "${critical_paths[@]}"; do
    local src="${ROOT_DIR}/${rel}"
    if [[ -e "${src}" ]]; then
      cp -a "${src}" "${backup_dir}/"
      copied+=("${rel}")
    else
      log "Critical path missing and skipped during backup: ${rel}." "WARN"
    fi
  done

  log "Backup captured at ${backup_dir} (${#copied[@]} entries). Python interpreter: ${python_bin}."
}

init_directories() {
  local -a gemma_dirs=(
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

  local created=0
  for rel in "${gemma_dirs[@]}"; do
    local dir="${ROOT_DIR}/${rel}"
    if [[ ! -d "${dir}" ]]; then
      mkdir -p "${dir}"
      ((created+=1))
      log "Created: ${rel}."
    else
      log "Verified existing directory: ${rel}." "DEBUG"
    fi
  done

  log "GEMMA directory scaffolding complete (created ${created} of ${#gemma_dirs[@]})."
}

main() {
  log "Starting GEMMA infrastructure bootstrap."
  local python_bin
  python_bin="$(require_python_311)"
  perform_backups "${python_bin}"
  init_directories
  log "Infrastructure bootstrap finished. Review logs/gemma/ for follow-up actions."
}

main "$@"
