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

  local fallback_interpreter=""
  local fallback_version=""

  for candidate in "${candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
      continue
    fi

    local candidate_version
    candidate_version="$(${candidate} -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
    if [[ -z "${candidate_version}" ]]; then
      continue
    fi

    if [[ "${candidate_version}" == "3.11" ]]; then
      echo "${candidate}"
      return
    fi

    if [[ -z "${fallback_interpreter}" ]]; then
      fallback_interpreter="${candidate}"
      fallback_version="${candidate_version}"
    fi
  done

  if [[ -n "${fallback_interpreter}" ]]; then
    log "Python 3.11 is required; detected ${fallback_interpreter} -> ${fallback_version}." "ERROR"
  else
    log "Python interpreter could not be located." "ERROR"
  fi
  exit 1
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
