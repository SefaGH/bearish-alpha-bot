#!/bin/bash
set -euo pipefail

if [ -z "${EVENT_HUB_CONNECTION_STRING:-}" ]; then
    echo "Error: EVENT_HUB_CONNECTION_STRING is not set."
    exit 1
fi

# Install Fluent Bit
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sh

# Create directories
mkdir -p /etc/fluent-bit
mkdir -p /data/parsed
chown -R azureuser:azureuser /data/parsed

# Write parsers.conf
cat > /etc/fluent-bit/parsers.conf <<'EOF'
[PARSER]
    Name        json
    Format      json
    Time_Key    timestamp_utc
    Time_Format %Y-%m-%dT%H:%M:%SZ
    Time_Keep   On
EOF

# Write fluent-bit.conf
cat > /etc/fluent-bit/fluent-bit.conf <<EOF
[SERVICE]
    Flush        5
    Daemon       Off
    Log_Level    debug
    Parsers_File parsers.conf

[INPUT]
    Name         tail
    Path         /data/parsed/*.ndjson
    Tag          bearish-bot
    Parser       json
    Skip_Long_Lines On
    Read_from_Head On

[FILTER]
    Name         modify
    Match        bearish-bot
    Add          source parser

[OUTPUT]
    Name        kafka
    Match       bearish-bot
    Brokers     bearishreportingehns.servicebus.windows.net:9093
    Topics      parsed-events
    rdkafka.security.protocol   SASL_SSL
    rdkafka.sasl.mechanism      PLAIN
    rdkafka.sasl.username       \$ConnectionString
    rdkafka.sasl.password       $EVENT_HUB_CONNECTION_STRING
    rdkafka.request.required.acks 1
EOF

# Restart Fluent Bit
systemctl enable fluent-bit
systemctl restart fluent-bit
systemctl status fluent-bit --no-pager
