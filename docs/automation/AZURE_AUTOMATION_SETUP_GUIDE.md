# Azure Automation Runbook - Deployment & Configuration Guide

## 📋 Overview

This guide provides comprehensive instructions for deploying and configuring the **Bearish Alpha Bot Azure Automation Pipeline** as specified in Issue #434.

### Architecture Components

```
┌─────────────────┐      HTTP POST      ┌──────────────┐
│   Logic App     │ ◄──────────────────► │    User      │
│  (Orchestrator) │                      │  (Mobile/Web)│
└────────┬────────┘                      └──────────────┘
         │
         │ Triggers
         ▼
┌─────────────────┐      Manages        ┌──────────────┐
│   Automation    │ ───────────────────► │   Azure VM   │
│    Runbook      │      (RunCommand)    │              │
└────────┬────────┘                      └──────┬───────┘
         │                                       │
         │ Retrieves                             │ Runs
         ▼                                       ▼
┌─────────────────┐                      ┌──────────────┐
│   Key Vault     │                      │   Docker     │
│   (Secrets)     │                      │  Container   │
└─────────────────┘                      └──────────────┘
```

---

## 🎯 Features Implemented

✅ **Concurrency Control**: Lock file mechanism prevents overlapping runs  
✅ **Secret Management**: Azure Key Vault integration with Managed Identity  
✅ **Retry Logic**: Exponential backoff with configurable attempts  
✅ **Timeout Handling**: Docker timeout enforcement + graceful shutdown  
✅ **Comprehensive Logging**: Structured logs with Azure Monitor integration  
✅ **Idempotency**: Token-based duplicate prevention  
✅ **VM Lifecycle Management**: Auto-start and deallocate  
✅ **RunCommand Resilience**: Handles Azure Agent timeout scenarios  

---

## 🚀 Quick Start Deployment

### Prerequisites

1. **Azure CLI** installed and authenticated
   ```bash
   az login
   az account set --subscription "YOUR_SUBSCRIPTION_ID"
   ```

2. **PowerShell 7+** (for deployment script)

3. **Existing Resources**:
   - Resource Group: `TradeBot`
   - Azure VM: `BearishAlphaBot-VM-01` (with SSH: 20.73.171.66)
   - Azure Container Registry: `bearishalphabot.azurecr.io`
   - Key Vault: `bearish-kv` (optional but recommended)

---

## 📦 Step 1: Deploy Automation Account & Runbook

### Option A: Using Deployment Script (Recommended)

```powershell
# Navigate to automation directory
cd c:\Users\sefaa\bearish-alpha-bot\infra\automation

# Deploy with default settings
.\Deploy-AutomationRunbook.ps1 `
  -ResourceGroup "TradeBot" `
  -AutomationAccountName "tradebot-automation"

# Or with custom settings
.\Deploy-AutomationRunbook.ps1 `
  -ResourceGroup "TradeBot" `
  -AutomationAccountName "tradebot-automation" `
  -Location "eastus" `
  -VMResourceGroup "TradeBot" `
  -VMName "BearishAlphaBot-VM-01" `
  -KeyVaultName "bearish-kv"
```

### Option B: Manual Azure CLI Deployment

```bash
# Variables
RESOURCE_GROUP="TradeBot"
AUTOMATION_ACCOUNT="tradebot-automation"
RUNBOOK_NAME="Start-BearishBot-Enhanced"
LOCATION="eastus"

# Create Automation Account
az automation account create \
  --name $AUTOMATION_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Basic

# Enable Managed Identity
az automation account update \
  --name $AUTOMATION_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --assign-identity

# Create runbook
az automation runbook create \
  --name $RUNBOOK_NAME \
  --automation-account-name $AUTOMATION_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --type PowerShell \
  --location $LOCATION

# Upload content
az automation runbook replace-content \
  --name $RUNBOOK_NAME \
  --automation-account-name $AUTOMATION_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --content @Start-BearishBot-Enhanced.ps1

# Publish
az automation runbook publish \
  --name $RUNBOOK_NAME \
  --automation-account-name $AUTOMATION_ACCOUNT \
  --resource-group $RESOURCE_GROUP
```

---

## 🔐 Step 2: Configure Permissions

### 2.1 Get Managed Identity Principal ID

```bash
PRINCIPAL_ID=$(az automation account show \
  --name tradebot-automation \
  --resource-group TradeBot \
  --query identity.principalId -o tsv)

echo "Principal ID: $PRINCIPAL_ID"
```

### 2.2 Assign VM Contributor Role

```bash
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

az role assignment create \
  --assignee-object-id $PRINCIPAL_ID \
  --assignee-principal-type ServicePrincipal \
  --role "Virtual Machine Contributor" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/TradeBot/providers/Microsoft.Compute/virtualMachines/BearishAlphaBot-VM-01"
```

### 2.3 Configure Key Vault Access

```bash
az keyvault set-policy \
  --name bearish-kv \
  --object-id $PRINCIPAL_ID \
  --secret-permissions get list
```

### 2.4 Add Secrets to Key Vault

```bash
# Add trading bot secrets
az keyvault secret set --vault-name bearish-kv --name BINGX-KEY --value "YOUR_API_KEY"
az keyvault secret set --vault-name bearish-kv --name BINGX-SECRET --value "YOUR_API_SECRET"
az keyvault secret set --vault-name bearish-kv --name TELEGRAM-BOT-TOKEN --value "YOUR_TELEGRAM_TOKEN"
```

---

## 🧪 Step 3: Test Runbook Execution

### Direct Runbook Test

```bash
# Start runbook with 10-minute test session
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=10 imageTag=vm-vmboot-9

# Check job status
JOB_ID=$(az automation job list \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --query "[0].name" -o tsv)

az automation job show \
  --job-name $JOB_ID \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot

# Get output
az automation job output \
  --job-name $JOB_ID \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot
```

---

## 🔄 Step 4: Deploy Logic App for HTTP Trigger

### 4.1 Create Logic App

```bash
# Create Logic App resource
az logic workflow create \
  --name bearish-bot-orchestrator \
  --resource-group TradeBot \
  --location eastus \
  --definition @logic-app-workflow.json
```

### 4.2 Configure Connections

The Logic App requires two connections:

#### Azure Automation Connection

```bash
# Create Automation connection via Azure Portal
# Navigate to: Logic App > API Connections > Add > Azure Automation
# Or use Azure Resource Manager template
```

#### Office 365 Connection (for email notifications)

```bash
# Create Office 365 connection via Azure Portal
# Navigate to: Logic App > API Connections > Add > Office 365 Outlook
# Authenticate with your Microsoft account
```

### 4.3 Enable Managed Identity for Logic App

```bash
# Enable system-assigned identity
az logic workflow identity assign \
  --name bearish-bot-orchestrator \
  --resource-group TradeBot

# Get Logic App principal ID
LOGIC_APP_PRINCIPAL=$(az logic workflow show \
  --name bearish-bot-orchestrator \
  --resource-group TradeBot \
  --query identity.principalId -o tsv)

# Assign Automation Operator role
az role assignment create \
  --assignee-object-id $LOGIC_APP_PRINCIPAL \
  --assignee-principal-type ServicePrincipal \
  --role "Automation Operator" \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/TradeBot/providers/Microsoft.Automation/automationAccounts/tradebot-automation"
```

### 4.4 Get HTTP Trigger URL

```bash
# Get callback URL for HTTP trigger
az logic workflow show \
  --name bearish-bot-orchestrator \
  --resource-group TradeBot \
  --query "accessEndpoint" -o tsv
```

Or via Azure Portal:
1. Go to Logic App → Logic app designer
2. Click on "When a HTTP request is received" trigger
3. Copy the **HTTP POST URL**

---

## 📱 Step 5: Test End-to-End Workflow

### Test via PowerShell

```powershell
$url = "https://prod-XX.eastus.logic.azure.com:443/workflows/.../triggers/manual/paths/invoke?..."

$body = @{
    durationMinutes = 10
    imageTag = "vm-vmboot-9"
    keyVaultName = "bearish-kv"
    kvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN"
} | ConvertTo-Json

Invoke-RestMethod -Uri $url -Method POST -Body $body -ContentType "application/json"
```

### Test via curl (from VM SSH)

```bash
curl -X POST "https://prod-XX.eastus.logic.azure.com:443/workflows/.../triggers/manual/paths/invoke?..." \
  -H "Content-Type: application/json" \
  -d '{
    "durationMinutes": 10,
    "imageTag": "vm-vmboot-9"
  }'
```

---

## 📊 Monitoring & Observability

### View Runbook Execution Logs

```bash
# List recent jobs
az automation job list \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --query "[].{Name:name, Status:status, StartTime:startTime}" \
  --output table

# Get detailed job output
az automation job output \
  --job-name JOB_ID \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot
```

### Azure Portal Monitoring

1. **Automation Account**:
   - Navigate to: Automation Account → Jobs
   - View: Status, Duration, Error Messages

2. **Logic App**:
   - Navigate to: Logic App → Overview → Runs history
   - View: Execution flow, Action outputs

3. **VM Insights**:
   - Navigate to: VM → Monitoring → Insights
   - View: CPU, Memory, Disk usage during bot runs

---

## 🔧 Troubleshooting

### Issue: "Concurrency lock detected"

**Cause**: Another runbook instance is running  
**Solution**: Wait for current job to complete or manually release lock

```bash
# Release lock via VM RunCommand
az vm run-command invoke \
  -g TradeBot \
  -n BearishAlphaBot-VM-01 \
  --command-id RunShellScript \
  --scripts "rm -f /tmp/bearish_bot_automation.lock"
```

### Issue: "Managed Identity authentication failed"

**Cause**: MSI not enabled or missing permissions  
**Solution**: Re-run deployment script or manually assign roles

```bash
# Verify identity is enabled
az automation account show \
  --name tradebot-automation \
  --resource-group TradeBot \
  --query identity

# Re-assign permissions
.\Deploy-AutomationRunbook.ps1 -ResourceGroup TradeBot -AutomationAccountName tradebot-automation
```

### Issue: "RunCommand timeout"

**Cause**: Container execution exceeded 85 minutes  
**Solution**: Reduce `durationMinutes` parameter or increase Azure Agent timeout

### Issue: "Key Vault secret not found"

**Cause**: Secret doesn't exist or MSI lacks permissions  
**Solution**: Verify secret exists and permissions are set

```bash
# List secrets
az keyvault secret list --vault-name bearish-kv --query "[].name"

# Verify access policy
az keyvault show --name bearish-kv --query properties.accessPolicies
```

---

## 🎯 Production Recommendations

### 1. Enable Azure Monitor Alerts

Create alerts for:
- Runbook failures
- VM high CPU/memory during trading
- Logic App execution failures

```bash
# Create action group for notifications
az monitor action-group create \
  --name bearish-alerts \
  --resource-group TradeBot \
  --short-name alerts \
  --email-receiver sefaasar sefaasar@hotmail.com
```

### 2. Implement Cost Controls

- **VM Auto-deallocate**: Already implemented in runbook
- **Automation Account**: Use Basic SKU (sufficient for this use case)
- **Logic App**: Standard tier (pay-per-execution)

### 3. Security Hardening

- [ ] Enable Azure Key Vault firewall
- [ ] Use Private Endpoints for VM
- [ ] Enable diagnostic logs for all resources
- [ ] Implement Azure Policy for governance

### 4. Backup & Disaster Recovery

```bash
# Export runbook for backup
az automation runbook export \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --output-folder ./backups

# Export Logic App definition
az logic workflow show \
  --name bearish-bot-orchestrator \
  --resource-group TradeBot > logic-app-backup.json
```

---

## 📝 Parameter Reference

### Runbook Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `durationMinutes` | int | ✅ | - | Trading session duration (1-85) |
| `resourceGroup` | string | ❌ | TradeBot | Azure resource group |
| `vmName` | string | ❌ | BearishAlphaBot-VM-01 | VM name |
| `imageTag` | string | ❌ | vm-vmboot-9 | Docker image tag |
| `keyVaultName` | string | ❌ | bearish-kv | Key Vault name |
| `kvSecretNames` | string | ❌ | BINGX-KEY,... | Comma-separated secrets |
| `idempotencyToken` | string | ❌ | "" | Unique execution token |
| `maxRetries` | int | ❌ | 3 | Max retry attempts |
| `targetEnv` | string | ❌ | prod | BingX routing env: `vst` uses `open-api-vst`, `prod` uses production API |

### Logic App Trigger Schema

```json
{
  "durationMinutes": 60,           // Required: 1-85
  "targetEnv": "prod",             // Optional: "vst" | "prod" (default: "prod")
  "imageTag": "vm-vmboot-9",       // Optional
  "keyVaultName": "bearish-kv",    // Optional
  "kvSecretNames": "BINGX-KEY,..." // Optional
}
```

Example payloads:

```json
{ "durationMinutes": 10, "targetEnv": "vst" }
```

```json
{ "durationMinutes": 10, "targetEnv": "prod" }
```

---

## 🔗 Related Documentation

- [Azure Automation Runbooks](https://learn.microsoft.com/azure/automation/automation-runbook-execution)
- [Logic Apps](https://learn.microsoft.com/azure/logic-apps/)
- [Managed Identity](https://learn.microsoft.com/entra/identity/managed-identities-azure-resources/)
- [Azure Key Vault](https://learn.microsoft.com/azure/key-vault/)
- [VM Run Command](https://learn.microsoft.com/azure/virtual-machines/run-command-overview)

---

## 📞 Support & Feedback

For issues related to this automation pipeline, please refer to:
- **GitHub Issue**: #434
- **Repository**: SefaGH/bearish-alpha-bot
- **VM IP**: 20.73.171.66 (SSH access available)

---

## ✅ Deployment Checklist

- [ ] Azure Automation Account created
- [ ] Runbook deployed and published
- [ ] Managed Identity enabled and permissions assigned
- [ ] Key Vault secrets configured
- [ ] Logic App created with HTTP trigger
- [ ] API connections configured (Automation + Office 365)
- [ ] Test execution completed successfully
- [ ] Monitoring and alerts configured
- [ ] Documentation updated with HTTP trigger URL
- [ ] Team notified of new deployment

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-28  
**Author**: Bearish Alpha Bot Team
