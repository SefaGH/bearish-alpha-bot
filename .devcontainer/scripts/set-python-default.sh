#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/usr/local/python/current/bin/python"
PYTHON3_BIN="/usr/local/python/current/bin/python3"

sudo update-alternatives --install /usr/bin/python python "${PYTHON_BIN}" 1
sudo update-alternatives --install /usr/bin/python3 python3 "${PYTHON3_BIN}" 1
sudo update-alternatives --set python "${PYTHON_BIN}"
sudo update-alternatives --set python3 "${PYTHON3_BIN}"

echo "Configured system python symlinks to ${PYTHON_BIN}."
echo "Configured system python3 symlinks to ${PYTHON3_BIN}."
echo "Verification of selected interpreters:"
python --version
python3 --version
