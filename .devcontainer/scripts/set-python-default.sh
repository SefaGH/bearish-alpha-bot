#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.11"

resolve_python_path() {
  local exe="$1"
  if ! command -v "$exe" >/dev/null 2>&1; then
    echo "ERROR:command_not_found"
    return 1
  fi
  "$exe" -c 'import sys; print(sys.executable)'
}

PYTHON_TARGET_OUTPUT="$(resolve_python_path "python${PYTHON_VERSION}")"
PYTHON_TARGET_STATUS=$?
if [[ ${PYTHON_TARGET_STATUS} -ne 0 ]]; then
  if [[ "${PYTHON_TARGET_OUTPUT}" == ERROR:* ]]; then
    echo "Unable to locate python${PYTHON_VERSION}: command not found." >&2
  else
    echo "Failed to resolve python${PYTHON_VERSION}: ${PYTHON_TARGET_OUTPUT}" >&2
  fi
  exit 1
fi

PYTHON_TARGET="${PYTHON_TARGET_OUTPUT}"

if command -v python3 >/dev/null 2>&1; then
  CURRENT_PYTHON3="$(python3 -c 'import sys; print(sys.executable)')"
else
  CURRENT_PYTHON3=""
fi

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
"python${PYTHON_VERSION}" --version
