#!/bin/bash
# Fluent Bit package bootstrapper for the bearish bot VM

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root (sudo)." >&2
  exit 1
fi

REPO_ROOT=${1:-/opt/bearish-alpha-bot}
FB_ETC_DIR=/etc/fluent-bit
UNIT_DIR=/etc/systemd/system/td-agent-bit.service.d

WORKSPACE_ID=${WORKSPACE_ID:-}
WORKSPACE_KEY=${WORKSPACE_KEY:-}
LOG_TYPE=${LOG_TYPE:-bearish_events}

if [[ -z "${WORKSPACE_ID}" || -z "${WORKSPACE_KEY}" ]]; then
  cat <<'EOF' >&2
Set WORKSPACE_ID and WORKSPACE_KEY in the environment before running this script.
Example:
  sudo WORKSPACE_ID=... WORKSPACE_KEY=... LOG_TYPE=bearish_events ./install_fluent_bit.sh /home/azureuser/bearish-alpha-bot
EOF
  exit 1
fi

apt-get update
apt-get install -y td-agent-bit

mkdir -p "${FB_ETC_DIR}"
cp "${REPO_ROOT}/infra/fluent-bit/fluent-bit.conf" "${FB_ETC_DIR}/fluent-bit.conf"
cp "${REPO_ROOT}/infra/fluent-bit/parsers.conf" "${FB_ETC_DIR}/parsers.conf"
chmod 644 "${FB_ETC_DIR}/fluent-bit.conf" "${FB_ETC_DIR}/parsers.conf"
chown root:root "${FB_ETC_DIR}/fluent-bit.conf" "${FB_ETC_DIR}/parsers.conf"

mkdir -p "${UNIT_DIR}"
cat > "${UNIT_DIR}/env.conf" <<EOF
[Service]
Environment="WORKSPACE_ID=${WORKSPACE_ID}"
Environment="WORKSPACE_KEY=${WORKSPACE_KEY}"
Environment="LOG_TYPE=${LOG_TYPE}"
EOF

systemctl daemon-reload
systemctl enable td-agent-bit
systemctl restart td-agent-bit

systemctl --no-pager status td-agent-bit

echo "Fluent Bit installed. Tail service logs with: journalctl -u td-agent-bit -f"
