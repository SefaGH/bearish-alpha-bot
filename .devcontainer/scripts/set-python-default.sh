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

PYTHON_TARGET="$(resolve_python_path "python${PYTHON_VERSION}")"
if [[ -z "${PYTHON_TARGET}" ]]; then
  echo "Unable to locate python${PYTHON_VERSION}." >&2
  exit 1
fi

PYTHON3_TARGET="${PYTHON_TARGET}"
if command -v python3 >/dev/null 2>&1; then
  CURRENT_PYTHON3="$(python3 -c 'import sys; print(sys.executable)')"
else
  CURRENT_PYTHON3=""
fi

sudo update-alternatives --install /usr/bin/python python "${PYTHON_TARGET}" 1
sudo update-alternatives --install /usr/bin/python3 python3 "${PYTHON3_TARGET}" 1
sudo update-alternatives --set python "${PYTHON_TARGET}"
sudo update-alternatives --set python3 "${PYTHON3_TARGET}"

cat <<MSG
Configured system python symlinks to ${PYTHON_TARGET}.
Configured system python3 symlinks to ${PYTHON3_TARGET}.
Previously active python3 interpreter: ${CURRENT_PYTHON3:-"(none)"}
Verification of selected interpreters:
MSG

python --version
python3 --version
"python${PYTHON_VERSION}" --version
