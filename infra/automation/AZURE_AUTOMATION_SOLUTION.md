# Azure Automation Solution for Bearish Alpha Bot

**Issue:** [#434 - Dayanıklı, Güvenli ve İzlenebilir Trade Bot Mobil Çalıştırma Pipeline'ı](https://github.com/SefaGH/bearish-alpha-bot/issues/434)

**Status:** ✅ Production Ready

**Deployment Date:** November 29, 2025

---

## 🎯 Overview

This solution provides a production-grade, serverless automation pipeline for remotely executing the Bearish Alpha Bot on Azure VMs. The system eliminates the need for persistent SSH connections, provides comprehensive monitoring, and ensures secure, reliable bot execution.

### Key Features

- ✅ **Zero-Touch Execution**: Start trading sessions via HTTP API
- ✅ **Automatic VM Management**: Start, execute, and stop VMs automatically
- ✅ **Secure Secret Management**: Azure Key Vault integration for credentials
- ✅ **Concurrency Protection**: Prevents multiple simultaneous bot instances
- ✅ **Comprehensive Logging**: Structured logs with timestamps and severity
- ✅ **Email Notifications**: Real-time alerts via SendGrid
- ✅ **Retry Logic**: Built-in resilience with configurable retry attempts
- ✅ **Cost Optimization**: VMs only run during trading sessions

---

## 🏗️ Architecture

```
┌─────────────────┐
│   HTTP Trigger  │ (Optional Logic App or Direct API Call)
│   via Logic App │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Azure Automation Runbook          │
│   Start-BearishBot-Enhanced.ps1     │
│                                     │
│   • Parameter validation            │
│   • Concurrency check               │
│   • VM power management             │
│   • Key Vault secret retrieval      │
│   • Docker container execution      │
│   • Structured logging              │
└────────┬────────────────────────────┘
         │
         ├──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
    │   VM    │   │  Key    │   │  Managed │   │ SendGrid │
    │ Control │   │  Vault  │   │ Identity │   │  Email   │
    └─────────┘   └─────────┘   └──────────┘   └──────────┘
         │
         ▼
    ┌──────────────────────────────┐
    │  BearishAlphaBot-VM-01       │
    │  Docker Container Execution  │
    │  • Pull latest image         │
    │  • Inject secrets via env    │
    │  • Execute trading session   │
    │  • Write logs to volumes     │
    └──────────────────────────────┘
```

---

## 📋 Components

### 1. Azure Automation Runbook

**File:** `Start-BearishBot-Enhanced.ps1`

**Purpose:** Orchestrates the entire execution lifecycle

**Key Functions:**
- `Write-StructuredLog`: Consistent logging format
- `Invoke-WithRetry`: Resilient operation execution
- `Test-ConcurrencyLock`: Prevents parallel executions

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `durationMinutes` | Integer | Yes | - | Trading session duration (1-85 min) |
| `resourceGroup` | String | No | `TradeBot` | Azure resource group |
| `vmName` | String | No | `BearishAlphaBot-VM-01` | Target VM name |
| `imageTag` | String | No | `vm-vmboot-9` | Docker image tag |
| `keyVaultName` | String | No | `bearish-kv` | Key Vault for secrets |
| `kvSecretNames` | String | No | `BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN` | Secrets to inject |
| `idempotencyToken` | String | No | Auto-generated | Unique execution ID |

### 2. Logic App Workflow (Optional)

**File:** `logic-app-workflow-sendgrid.json`

**Purpose:** HTTP-triggered orchestration with email notifications

**Features:**
- HTTP trigger with JSON schema validation
- Concurrency check before runbook invocation
- Wait for job completion (up to 2 hours)
- SendGrid email notifications (success/failure)
- Structured JSON response

**Deployment:** Use Azure Portal to import the workflow JSON

### 3. Deployment Scripts

#### `Deploy-Simple.ps1`
Deploys the runbook to Azure Automation Account.

```powershell
.\Deploy-Simple.ps1 -ResourceGroup "TradeBot" -AutomationAccountName "tradebot-automation"
```

#### `Deploy-LogicApp.ps1`
Deploys the Logic App workflow (requires manual configuration).

```powershell
.\Deploy-LogicApp.ps1 -ResourceGroup "TradeBot" -LogicAppName "bearish-bot-orchestrator"
```

---

## 🚀 Setup Guide

### Prerequisites

1. **Azure Subscription** with contributor access
2. **Azure VM** with Docker installed (BearishAlphaBot-VM-01)
3. **Azure Container Registry** (bearishalphabot.azurecr.io)
4. **Azure Key Vault** with bot secrets (bearish-kv)
5. **SendGrid API Key** (optional, for email notifications)

### Step-by-Step Deployment

#### 1. Create Azure Automation Account

```bash
az automation account create \
  --name "tradebot-automation" \
  --resource-group "TradeBot" \
  --location "eastus" \
  --sku "Basic"
```

#### 2. Deploy Runbook

```powershell
cd infra/automation
.\Deploy-Simple.ps1 -ResourceGroup "TradeBot" -AutomationAccountName "tradebot-automation"
```

**Output:**
```
✓ Existing runbook deleted
✓ Runbook created
✓ Content uploaded
✓ Runbook published
```

#### 3. Enable Managed Identity

**Option A: Azure Portal**
1. Go to Automation Account → Identity
2. Enable "System assigned" identity
3. Copy the Principal (Object) ID

**Option B: Azure CLI**
```bash
az automation account update \
  --name "tradebot-automation" \
  --resource-group "TradeBot" \
  --identity '[{"type":"SystemAssigned"}]'
```

#### 4. Assign Permissions

**VM Contributor Role:**
```bash
az role assignment create \
  --assignee <PRINCIPAL_ID> \
  --role "Virtual Machine Contributor" \
  --scope "/subscriptions/<SUB_ID>/resourceGroups/TradeBot/providers/Microsoft.Compute/virtualMachines/BearishAlphaBot-VM-01"
```

**Key Vault Access:**
```bash
az keyvault set-policy \
  --name bearish-kv \
  --object-id <PRINCIPAL_ID> \
  --secret-permissions get list
```

#### 5. Test Execution

```bash
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=5 imageTag=vm-vmboot-9
```

#### 6. Monitor Job

```bash
# Get job status
az automation job show \
  --job-name <JOB_ID> \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --query '{status:status, startTime:startTime, endTime:endTime}'

# Get job output
az automation job output \
  --job-name <JOB_ID> \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot
```

---

## 🔧 Usage

### Direct API Invocation

```bash
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=60 imageTag=vm-vmboot-9
```

### Via Logic App (After Deployment)

```powershell
$endpoint = "<LOGIC_APP_CALLBACK_URL>"

Invoke-RestMethod -Method POST -Uri $endpoint `
  -ContentType "application/json" `
  -Body (@{
      durationMinutes = 60
      imageTag = "vm-vmboot-9"
      keyVaultName = "bearish-kv"
      kvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN"
  } | ConvertTo-Json)
```

### Via PowerShell (Programmatic)

```powershell
Connect-AzAccount

$params = @{
    durationMinutes = 30
    resourceGroup = "TradeBot"
    vmName = "BearishAlphaBot-VM-01"
    imageTag = "vm-vmboot-9"
    keyVaultName = "bearish-kv"
    kvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN"
}

Start-AzAutomationRunbook `
    -AutomationAccountName "tradebot-automation" `
    -Name "Start-BearishBot-Enhanced" `
    -ResourceGroupName "TradeBot" `
    -Parameters $params
```

---

## 📊 Monitoring & Logging

### Structured Logging Format

All logs follow a consistent JSON-like structure:

```json
{
  "timestamp": "2025-11-29T00:06:14Z",
  "level": "INFO",
  "message": "Step 1: Validating parameters...",
  "context": {
    "durationMinutes": 60,
    "imageTag": "vm-vmboot-9",
    "jobId": "44fc1756-ad9b-4620-a66d-2c7b19c716cc"
  }
}
```

### Log Levels

- **INFO**: Normal operational messages
- **SUCCESS**: Successful operation completion
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures

### Viewing Logs

**Azure Portal:**
1. Navigate to Automation Account → Jobs
2. Select the job
3. View "All Logs" or "Output" stream

**Azure CLI:**
```bash
az automation job stream list \
  --job-name <JOB_ID> \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot
```

---

## 🔒 Security

### Secrets Management

- **Key Vault Integration**: All sensitive credentials stored in Azure Key Vault
- **Managed Identity**: No stored passwords or API keys in runbook
- **Least Privilege**: Minimal required permissions assigned
- **Temporary Files**: Secrets written to VM are shredded after use

### Network Security

- **No Public Endpoints**: VM not directly accessible from internet
- **Azure Backbone**: All communication via Azure internal network
- **No SSH Required**: Eliminates SSH key management risks

### Audit Trail

- **Job History**: All executions logged with unique job IDs
- **Parameter Logging**: Input parameters recorded for audit
- **Concurrency Tracking**: Prevents unauthorized parallel executions

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "az command not recognized"

**Problem:** Azure CLI not available in Azure Automation sandbox.

**Solution:** Use PowerShell cmdlets (`Invoke-AzVMRunCommand`) instead of `az` CLI.

**Fixed in:** Current version uses `Invoke-AzVMRunCommand` with base64 encoding.

#### 2. PowerShell Parse Errors

**Problem:** Bash script embedded in PowerShell causing parse errors.

**Solution:** Base64-encode bash script and pass as parameter to wrapper script.

**Implementation:**
```powershell
$base64Script = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($bashScript))
$wrapper = @"
#!/bin/bash
echo "\$1" | base64 --decode | bash
"@
Invoke-AzVMRunCommand -ScriptString $wrapper -Parameter @{arg1 = $base64Script}
```

#### 3. Managed Identity Permissions

**Problem:** Runbook fails with "Forbidden" or "Unauthorized" errors.

**Solution:** Verify role assignments and Key Vault access policies.

```bash
# Check role assignments
az role assignment list --assignee <PRINCIPAL_ID>

# Check Key Vault policies
az keyvault show --name bearish-kv --query properties.accessPolicies
```

#### 4. Concurrent Execution Blocked

**Problem:** Job fails immediately with "Concurrent execution detected".

**Solution:** Wait for existing job to complete or manually release lock.

```bash
# Check running jobs
az automation job list \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --query "[?properties.status=='Running']"

# Stop stuck job if needed
az automation job stop \
  --job-name <JOB_ID> \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot
```

#### 5. VM Doesn't Start

**Problem:** VM remains in "PowerState/stopped" or "PowerState/deallocated" state.

**Solution:** Check VM status and permissions.

```bash
# Check VM state
az vm get-instance-view \
  --name BearishAlphaBot-VM-01 \
  --resource-group TradeBot \
  --query instanceView.statuses

# Manually start if needed
az vm start \
  --name BearishAlphaBot-VM-01 \
  --resource-group TradeBot
```

---

## 💰 Cost Optimization

### Resource Costs

| Resource | Type | Cost (Estimated) |
|----------|------|------------------|
| Azure Automation | Basic SKU | ~$1/month (500 min free) |
| Logic App | Consumption | ~$0.01 per execution |
| VM (B2s) | Compute | ~$0.05/hour (only when running) |
| Key Vault | Standard | ~$0.03/10k operations |
| **Total** | **Per 1-hour session** | **~$0.10** |

### Cost Reduction Tips

1. **Minimize Session Duration**: Use precise `durationMinutes` values
2. **Scheduled Executions**: Avoid unnecessary test runs
3. **Deallocate VMs**: Runbook automatically deallocates after execution
4. **Spot Instances**: Consider using spot VMs for non-critical trading
5. **Monitoring**: Set up Azure Cost Alerts for unexpected charges

---

## 📈 Performance

### Execution Timeline

**1-minute trading session:**
- VM Start: ~30 seconds
- Container Pull: ~10 seconds
- Trading Execution: 60 seconds
- Container Cleanup: ~5 seconds
- VM Deallocate: ~15 seconds
- **Total: ~2 minutes**

**10-minute trading session:**
- Overhead: ~60 seconds (startup + shutdown)
- Trading: 600 seconds
- **Total: ~11 minutes**

### Optimization

- **Image Caching**: Pre-pull Docker images on VM for faster startup
- **Persistent VMs**: Keep VM running for multiple short sessions
- **Batch Executions**: Group multiple trading sessions if applicable

---

## 🔄 Maintenance

### Regular Tasks

#### Weekly
- Review job execution logs for failures
- Monitor Azure Automation job history
- Check Key Vault secret expiration

#### Monthly
- Update Docker image tags (`imageTag` parameter)
- Review and rotate Key Vault secrets
- Audit role assignments and permissions

#### Quarterly
- Update runbook with latest best practices
- Review and optimize VM sizing
- Test disaster recovery procedures

### Updating the Runbook

```powershell
# 1. Modify Start-BearishBot-Enhanced.ps1 locally
# 2. Redeploy
.\Deploy-Simple.ps1 -ResourceGroup "TradeBot" -AutomationAccountName "tradebot-automation"

# 3. Test with short duration
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=1 imageTag=vm-vmboot-9
```

---

## 🧪 Testing

### Test Checklist

- [ ] 1-minute test execution completes successfully
- [ ] 10-minute test executes full trading session
- [ ] Concurrent execution is properly blocked
- [ ] Key Vault secrets are successfully retrieved
- [ ] VM starts, executes, and deallocates correctly
- [ ] Container logs are written to volumes
- [ ] Email notifications are received (if configured)
- [ ] Job output contains structured logs

### Validation Commands

```bash
# Start test
JOB_ID=$(az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=5 imageTag=vm-vmboot-9 \
  --query name -o tsv)

# Monitor progress
watch -n 10 "az automation job show --job-name $JOB_ID --automation-account-name tradebot-automation --resource-group TradeBot --query '{status:status,startTime:startTime}'"

# Verify completion
az automation job show \
  --job-name $JOB_ID \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --query '{status:status,exception:exception}'
```

---

## 📚 References

### Documentation
- [Azure Automation Documentation](https://learn.microsoft.com/azure/automation/)
- [Azure Logic Apps Documentation](https://learn.microsoft.com/azure/logic-apps/)
- [Azure Key Vault Documentation](https://learn.microsoft.com/azure/key-vault/)
- [SendGrid Email API](https://docs.sendgrid.com/api-reference/mail-send/mail-send)

### Related Files
- `Start-BearishBot-Enhanced.ps1` - Main runbook
- `Deploy-Simple.ps1` - Deployment script
- `Deploy-LogicApp.ps1` - Logic App deployment
- `logic-app-workflow-sendgrid.json` - Logic App definition

### Project Documentation
- [AZURE_VM_DEPLOYMENT_SUCCESS.md](../../AZURE_VM_DEPLOYMENT_SUCCESS.md)
- [AZURE_DEPLOYMENT_GUIDE.md](../../AZURE_DEPLOYMENT_GUIDE.md)
- [GitHub Issue #434](https://github.com/SefaGH/bearish-alpha-bot/issues/434)

---

## ✅ Production Checklist

Before going live, ensure:

- [x] Runbook deployed and tested
- [x] Managed Identity enabled and configured
- [x] VM Contributor role assigned
- [x] Key Vault access policy configured
- [x] 1-minute test passed
- [x] 10-minute test passed
- [ ] Logic App deployed (optional)
- [ ] SendGrid email configured (optional)
- [ ] Monitoring alerts configured
- [ ] Cost alerts configured
- [ ] Documentation reviewed by team

---

## 🎉 Conclusion

This Azure Automation solution provides a robust, secure, and cost-effective way to execute the Bearish Alpha Bot remotely. The system eliminates manual intervention, provides comprehensive logging, and ensures reliable execution with built-in retry logic and concurrency protection.

**Status:** ✅ Production Ready (Tested: November 29, 2025)

**Next Steps:**
1. Configure Logic App for HTTP-triggered execution
2. Set up SendGrid email notifications
3. Schedule regular executions via Azure Automation schedules
4. Monitor job execution and optimize parameters

---

**Maintained by:** Sefa Asar  
**Last Updated:** November 29, 2025  
**Version:** 1.0.0
