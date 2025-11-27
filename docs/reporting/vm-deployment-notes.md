# VM Deployment Notes for Reporting Pipeline

1. **Prerequisites**
   - Docker / Docker Compose installed (`sudo apt-get install docker-compose-plugin`).
   - Directories exist: `/mnt/bearish/logs`, `/mnt/bearish/data/parsed` (match volume mounts).
   - Managed identity on VM has access to Key Vault secrets for Log Analytics workspace key.

2. **Secrets**
   - Retrieve Log Analytics `workspaceId` and `sharedKey` from Key Vault and export as env vars:
     ```bash
     export WORKSPACE_ID=$(az keyvault secret show --vault-name bearish-kv --name log-analytics-workspace-id --query value -o tsv)
     export WORKSPACE_KEY=$(az keyvault secret show --vault-name bearish-kv --name log-analytics-shared-key --query value -o tsv)
     ```

3. **Compose file**
   - Copy `infra/docker-compose.reporting.yml` and `infra/fluent-bit/fluent-bit.conf` to `/home/azureuser/reporting/`.
   - Perform dry run:
     ```bash
     docker compose -f docker-compose.reporting.yml config
     ```

4. **Start services**
   ```bash
   docker compose -f docker-compose.reporting.yml up -d
   ```

5. **Validation**
   - Check parser logs: `docker logs -f reporting-log-parser`
   - Check Fluent Bit logs: `docker logs -f reporting-fluent-bit`
   - Verify ingestion:
     ```bash
     az monitor log-analytics query --workspace bearish-logs --analytics-query "BearishEvents_CL | take 5"
     ```

6. **Troubleshooting**
   - Ensure parser writes NDJSON lines; Fluent Bit requires newline-delimited JSON.
   - Confirm `WORKSPACE_KEY` rotation when regenerated (update Key Vault secret first).
   - Use `docker inspect` to verify volume mounts if files missing.
