#!/bin/bash
# ADX Ingestion Pipeline Health Check
# Sefa icin hazirlanmis uctan uca kontrol scripti

set -euo pipefail

# ---- CONFIG ----
RG="tradebot-ops"
ADX_CLUSTER="bearish-adx"
ADX_DB="bearishdb"
CONNECTION_NAME="bearish-parsed-events-eh-conn"
EH_NAMESPACE="bearishreportingehns"
EH_NAME="parsed-events"
SUBSCRIPTION_ID="74ab10ba-c96d-449e-97cb-ee4f9c0de714"
LOCATION="westeurope"
KUSTO_RESOURCE="https://kusto.kusto.windows.net"
KUSTO_URI="https://${ADX_CLUSTER}.${LOCATION}.kusto.windows.net"

ensure_kusto_extension() {
  az extension add --name kusto --only-show-errors >/dev/null 2>&1 || true
}

kusto_rows() {
  local endpoint="$1"
  local csl="$2"
  
  # Get token
  local token
  token=$(az account get-access-token --resource "$KUSTO_RESOURCE" --query accessToken -o tsv)
  
  # Use temp file for payload to avoid quoting issues
  local payload_file
  payload_file=$(mktemp)
  # Create JSON payload using python to ensure correct escaping
  python3 -c "import json; print(json.dumps({'db': '$ADX_DB', 'csl': '''$csl'''}));" > "$payload_file"
  
  curl -s -X POST "$endpoint" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    -d "@$payload_file" | \
    python3 -c "
import sys, json
raw = sys.stdin.read()
try:
    data = json.loads(raw)
    rows = []
    if isinstance(data, dict):
        tables = data.get('Tables', [])
        if tables:
            rows = tables[0].get('Rows', [])
    print(json.dumps(rows, indent=2))
except Exception as e:
    sys.stderr.write(f'Error parsing JSON: {e}\nRaw data: {raw}\n')
"
  rm -f "$payload_file"
}

ensure_kusto_extension

for tool in az curl python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool not found on PATH" >&2
    exit 1
  fi
done
echo "=== 1. Event Hub Connection Status ==="
az kusto data-connection show \
  --cluster-name "$ADX_CLUSTER" \
  --database-name "$ADX_DB" \
  --name "$CONNECTION_NAME" \
  --resource-group "$RG" \
  --query '{state:provisioningState,eventHub:eventHubResourceId,consumerGroup:consumerGroup}'

echo "=== 2. Event Hub Incoming Messages (last 5 min) ==="
az monitor metrics list \
  --resource "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RG/providers/Microsoft.EventHub/namespaces/$EH_NAMESPACE" \
  --metric IncomingMessages \
  --interval PT1M \
  --aggregation Total \
  --filter "EntityName eq '$EH_NAME'" \
  --query "value[].timeseries[].data[].{time:timeStamp,total:total}"

echo "=== 3. ADX Ingestion Failures (last 1h) ==="
kusto_rows "$KUSTO_URI/v1/rest/mgmt" ".show ingestion failures | where Table == 'bearish_events' and FailedOn > ago(1h) | project FailedOn, FailureStatus, Details"

echo "=== 4. ADX Table Sample Rows ==="
kusto_rows "$KUSTO_URI/v1/rest/query" "bearish_events | take 10"
