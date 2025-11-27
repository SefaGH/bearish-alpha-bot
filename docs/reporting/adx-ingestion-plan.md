# ADX Ingestion Plan

This document captures the recommended steps to wire Azure Data Explorer (ADX) ingestion for the reporting pipeline now that diagnostic settings and storage routing are in place.

## Objectives
- Stream the structured NDJSON files produced by the parser (`parsed-events/`) into the `bearishdb` database and `bearish_events` table.
- Optionally land Azure Monitor diagnostic logs (`insights-logs-*` containers) for supplementary auditing dashboards.

## Prerequisites
- ADX cluster `bearish-adx` and database `bearishdb` (created via `infra/step03-adx-and-export.ps1`).
- Storage account `bearishstorage` with the `parsed-events` container (created in Step 01) and diagnostic containers (created by Step 03 diagnostics).
- Managed identity `bearish-reporter-id` with read permissions on the storage account if the data connection uses MSI.

## Recommended Flow
1. **Create an Event Grid subscription on Storage**
   - Scope: `bearishstorage` account.
   - Filter: blob created events for `parsed-events/` prefix and (optional) `insights-logs-` containers.
   - Endpoint Type: ADX data connection (Event Grid).

2. **Provision ADX Event Grid Data Connection**
   - Command example:
     ```powershell
     az kusto data-connection event-grid create `
         --cluster-name bearish-adx `
         --database-name bearishdb `
         --name bearish-parsed-events-eg `
         --resource-group tradebot-ops `
         --context "StorageAccountResourceId=/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/tradebot-ops/providers/Microsoft.Storage/storageAccounts/bearishstorage" `
         --event-hub-resource-id "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/azure-eventgrid-subscriptions/providers/Microsoft.EventGrid/systemTopics/..." `
         --table-name bearish_events `
         --mapping-rule-name bearish_events_json_mapping `
         --data-format MULTIJSON
     ```
   - The command above assumes a system topic is created automatically; adapt the resource IDs once the Event Grid subscription is created.

3. **(Alternative) Use Blob Storage Data Connection**
   - If Event Grid is unavailable, schedule batch ingestion using `az kusto data-connection blob create` with the storage container SAS.
   - Configure ingestion batching (file patterns, interval) to match parser output cadence (defaults: 5 minute batches, 500 MB).

4. **Grant Storage Permissions to ADX**
   - Assign `Storage Blob Data Reader` to the ADX managed identity or data connection identity on `bearishstorage`.
   - Confirm role assignments:
     ```powershell
     az role assignment create `
         --assignee-object-id $(az kusto cluster show --name bearish-adx --resource-group tradebot-ops --query identity.principalId -o tsv) `
         --scope "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/tradebot-ops/providers/Microsoft.Storage/storageAccounts/bearishstorage" `
         --role "Storage Blob Data Reader"
     ```

5. **Validate Ingestion**
   - Drop a sample NDJSON file into `parsed-events/` and ensure an `.ingestionstatus` blob appears.
   - Run ADX validation:
     ```kql
     bearish_events
     | where timestamp_utc between (ago(1h) .. now())
     | take 10
     ```

6. **Monitor and Alert**
   - Use ADX command `.show ingestion failures` to track issues.
   - Add alerts in Azure Monitor for Event Grid dead-letter events or ADX ingestion failures if required.

## Outstanding Items
- Finalize Event Grid resource IDs once topics/subscriptions are deployed.
- Automate the steps above in a new `infra/step06-adx-ingestion.ps1` script when ready.
- Consider using Azure Data Explorer queued ingestion for high throughput scenarios.
