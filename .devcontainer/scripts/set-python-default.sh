#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.11"

resolve_python_path() {
  local exe="$1"
  if ! command -v "$exe" >/dev/null 2>&1; then
    return 1
  fi
  "$exe" -c 'import sys; print(sys.executable)'
}

if ! PYTHON_TARGET="$(resolve_python_path "python${PYTHON_VERSION}")"; then
  echo "Unable to locate python${PYTHON_VERSION}." >&2
  exit 1
fi

configure_pyenv() {
  if ! command -v pyenv >/dev/null 2>&1; then
    echo "pyenv not found; skipping shim configuration."
    return
  fi

  local pyenv_root
  pyenv_root="$(pyenv root)"
  local versions_dir="${pyenv_root}/versions"

  if [[ ! -d "${versions_dir}" ]]; then
    echo "pyenv versions directory does not exist; skipping shim configuration." >&2
    return
  fi
  local candidate_dir
  candidate_dir="$(find "${versions_dir}" -maxdepth 1 -mindepth 1 -type d -name "${PYTHON_VERSION}*" | sort -V | tail -n1)"

  if [[ -z "${candidate_dir}" ]]; then
    echo "pyenv does not have a python${PYTHON_VERSION} installation; skipping shim configuration." >&2
    return
  fi

  local version_name
  version_name="$(basename "${candidate_dir}")"

  if [[ "$(pyenv global)" != "${version_name}" ]]; then
    pyenv global "${version_name}"
  fi

  pyenv rehash

  cat <<MSG
Configured pyenv global version to ${version_name}.
pyenv shim verification:
MSG
  pyenv exec python --version
  pyenv exec python3 --version
}

configure_pyenv

CURRENT_PYTHON3="$(resolve_python_path "python3" || true)"

sudo update-alternatives --install /usr/bin/python python "${PYTHON_TARGET}" 1
sudo update-alternatives --install /usr/bin/python3 python3 "${PYTHON_TARGET}" 1
sudo update-alternatives --set python "${PYTHON_TARGET}"
sudo update-alternatives --set python3 "${PYTHON_TARGET}"

cat <<MSG
Configured system python symlinks to ${PYTHON_TARGET}.
Configured system python3 symlinks to ${PYTHON_TARGET}.
Previously active python3 interpreter: ${CURRENT_PYTHON3:-"(none)"}
Verification of selected interpreters:
MSG

python --version
python3 --version
python${PYTHON_VERSION} --version
