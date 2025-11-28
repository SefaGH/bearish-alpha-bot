#!/bin/bash
set -e
TOKEN=$(az account get-access-token --resource https://kusto.kusto.windows.net --query accessToken -o tsv)
echo "Token length: ${#TOKEN}"

# Try Query Endpoint
echo "--- Query Endpoint ---"
curl -s -X POST "https://bearish-adx.westeurope.kusto.windows.net/v1/rest/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"db": "bearishdb", "csl": "bearish_events | count"}' | head -n 5

# Check Ingestion Failures
echo "--- Ingestion Failures ---"
curl -s -X POST "https://bearish-adx.westeurope.kusto.windows.net/v1/rest/mgmt" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"db": "bearishdb", "csl": ".show ingestion failures | where Table == '\''bearish_events'\'' | take 5"}'

# Check Table Count
echo "--- Table Count ---"
curl -s -X POST "https://bearish-adx.westeurope.kusto.windows.net/v1/rest/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"db": "bearishdb", "csl": "bearish_events | count"}'
