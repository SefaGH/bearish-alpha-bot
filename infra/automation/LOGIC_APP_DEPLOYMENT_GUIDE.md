# Logic App Deployment Guide - Azure Portal

**Status:** Manual deployment required due to Azure CLI limitations  
**File:** `logic-app-workflow-sendgrid.json` (✅ Updated with forceRestart parameter)  
**Date:** November 29, 2025

---

## 📋 Prerequisites

- Azure subscription with Contributor access to TradeBot resource group
- SendGrid account with API key (for email notifications)
- Logic App Designer access in Azure Portal

---

## 🚀 Deployment Steps

### Step 1: Create Logic App in Azure Portal

1. Navigate to **Azure Portal** → https://portal.azure.com
2. Go to **Resource Groups** → **TradeBot**
3. Click **+ Create** → Search for **Logic App**
4. Click **Create**

**Configuration:**
```
Resource Group: TradeBot
Logic App Name: bearish-bot-orchestrator
Region: West Europe (same as resource group)
Type: Consumption
Enable Log Analytics: Yes (recommended)
```

5. Click **Review + Create** → **Create**
6. Wait for deployment (1-2 minutes)

---

### Step 2: Import Workflow Definition

1. Go to **Logic App** → **bearish-bot-orchestrator**
2. Click **Logic app code view** (left menu)
3. Open file: `infra/automation/logic-app-workflow-sendgrid.json`
4. **Copy the entire JSON content**
5. **Paste** into the code view in Azure Portal
6. Click **Save** (top toolbar)

**Verification:**
- ✅ HTTP trigger appears at the top
- ✅ Multiple actions visible (Check_Concurrent, Validate_Parameters, etc.)
- ✅ No syntax errors shown

---

### Step 3: Configure Parameters

1. Click **Logic App Designer** (switch from code view)
2. Click **Parameters** tab (top menu)
3. Add parameter:

**Parameter: sendgrid_api_key**
```
Name: sendgrid_api_key
Type: SecureString
Value: <YOUR_SENDGRID_API_KEY>
```

**How to get SendGrid API key:**
- Log in to SendGrid: https://app.sendgrid.com
- Navigate to **Settings** → **API Keys**
- Click **Create API Key**
- Name: `bearish-bot-notifications`
- Permissions: **Full Access** (or Mail Send only)
- Copy the API key (save it securely - shown only once)

---

### Step 4: Configure Managed Identity (Required for Runbook Access)

1. Go to **Logic App** → **Identity** (left menu)
2. **System assigned** tab
3. Set **Status** to **On**
4. Click **Save**
5. Copy the **Object (principal) ID**

**Grant Automation Permissions:**
```powershell
# Run in Azure Cloud Shell or local PowerShell with Az module
$principalId = "<OBJECT_ID_FROM_STEP_5>"

# Grant automation job operator role
New-AzRoleAssignment `
    -ObjectId $principalId `
    -RoleDefinitionName "Automation Job Operator" `
    -Scope "/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/TradeBot/providers/Microsoft.Automation/automationAccounts/tradebot-automation"
```

**Alternative (Azure Portal):**
1. Navigate to **Automation Account** → **tradebot-automation**
2. Click **Access control (IAM)** → **+ Add** → **Add role assignment**
3. Role: **Automation Job Operator**
4. Assign access to: **Managed identity**
5. Select: **Logic App** → **bearish-bot-orchestrator**
6. Click **Save**

---

### Step 5: Update Notification Email

1. Go to **Logic App Designer**
2. Scroll to **Send_Email_Success** action (near bottom)
3. Update **To** field:
   ```
   your-email@example.com
   ```
4. Update **From** field (must be verified in SendGrid):
   ```
   noreply@yourdomain.com
   ```
5. Repeat for **Send_Email_Failure** action
6. Click **Save** (top toolbar)

---

### Step 6: Get HTTP Trigger URL

1. Go to **Logic App Designer**
2. Click **HTTP trigger** (first action)
3. Click **Get callback URL** (top-right of trigger)
4. Copy the URL (format: `https://prod-XX.westeurope.logic.azure.com:443/workflows/...`)
5. **Save this URL securely** - needed for HTTP requests

**Security Note:**
- This URL contains a SAS token for authentication
- Anyone with this URL can trigger the Logic App
- Consider using Azure AD authentication for production

---

## 🧪 Testing

### Test 1: Normal Start (forceRestart=false)

**PowerShell:**
```powershell
$url = "<YOUR_CALLBACK_URL>"

Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-11",
        "forceRestart": false
    }'
```

**Expected Behavior:**
- ✅ Logic App runs successfully
- ✅ Runbook checks container status
- ✅ Aborts if bot is already running (unless forceRestart=true)
- ✅ Email notification sent on completion

---

### Test 2: Force Restart (forceRestart=true)

**PowerShell:**
```powershell
$url = "<YOUR_CALLBACK_URL>"

Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-11",
        "forceRestart": true
    }'
```

**Expected Behavior:**
- ✅ Logic App runs successfully
- ✅ Runbook stops existing container (30s timeout)
- ✅ Starts new container with vm-vmboot-11
- ✅ Trades for 5 minutes
- ✅ Email notification sent

---

### Test 3: Using Test Script

**From repository:**
```powershell
cd C:\Users\sefaa\bearish-alpha-bot\infra\automation

# Test normal start
.\Test-LogicApp.ps1 -CallbackUrl "<YOUR_URL>" -DurationMinutes 5 -ForceRestart $false

# Test force restart
.\Test-LogicApp.ps1 -CallbackUrl "<YOUR_URL>" -DurationMinutes 5 -ForceRestart $true
```

---

## 📊 Monitoring

### View Logic App Runs
1. Azure Portal → Logic App → **Runs history**
2. Click any run to see:
   - Trigger inputs
   - Action outputs
   - Execution duration
   - Success/failure status

### View Runbook Jobs
1. Azure Portal → Automation Account → **tradebot-automation**
2. Click **Jobs** (left menu)
3. Click any job to see:
   - Runbook parameters
   - Execution logs
   - Output streams
   - Errors (if any)

### View Email Notifications
- Check your email inbox for:
  - **Subject:** `✅ Bearish Bot Completed - <Job ID>` (success)
  - **Subject:** `❌ Bearish Bot Failed - <Job ID>` (failure)
- Email contains:
  - Duration
  - Image tag
  - Job ID
  - Status
  - Start/end times

---

## 🔧 Troubleshooting

### Issue: "Unauthorized" error when triggering Logic App

**Cause:** Managed Identity doesn't have Automation Job Operator role

**Solution:**
```powershell
# Get Logic App managed identity
$identity = az logic workflow show --name bearish-bot-orchestrator --resource-group TradeBot --query identity.principalId -o tsv

# Grant role
az role assignment create \
    --assignee $identity \
    --role "Automation Job Operator" \
    --scope "/subscriptions/<SUB_ID>/resourceGroups/TradeBot/providers/Microsoft.Automation/automationAccounts/tradebot-automation"
```

---

### Issue: Email not sent (SendGrid error)

**Possible causes:**
1. SendGrid API key invalid/expired
2. Sender email not verified in SendGrid
3. SendGrid account suspended

**Solution:**
1. Verify API key in SendGrid portal
2. Add sender email to **Sender Authentication** in SendGrid
3. Check SendGrid account status
4. Update `sendgrid_api_key` parameter in Logic App

---

### Issue: Runbook fails with "Container already running"

**Cause:** Bot is already running and `forceRestart=false`

**Solution:**
- This is expected behavior (prevents duplicate executions)
- To force restart: Set `forceRestart=true` in request body
- Or wait for current session to complete

---

### Issue: Logic App shows "definition validation failed"

**Cause:** Workflow JSON syntax error or missing parameters

**Solution:**
1. Validate JSON syntax: https://jsonlint.com
2. Compare with original: `infra/automation/logic-app-workflow-sendgrid.json`
3. Check all parameters are defined
4. Ensure managed identity is enabled

---

## 📝 Important Notes

### forceRestart Parameter
- **Default:** `false` (prevents duplicate executions)
- **true:** Stops existing container + starts new one
- **Use cases:**
  - Deploying new image version
  - Recovering from stuck container
  - Emergency restart

### Image Tag Versioning
- **Current:** `vm-vmboot-11` (13.5 GB)
- **Backup:** `vm-vmboot-10`
- **Latest features:** Status check, circuit breaker, optimized WS
- **Update process:**
  1. Build new image: `vm-vmboot-12`
  2. Push to ACR
  3. Use `forceRestart=true` + new imageTag

### Cost Considerations
- **Logic App:** ~$0.001 per execution
- **Runbook:** Included in Automation Account
- **SendGrid:** Free tier: 100 emails/day
- **Total:** <$5/month for typical usage

---

## ✅ Deployment Checklist

Before marking as complete:

- [ ] Logic App created in Azure Portal
- [ ] Workflow definition imported (with forceRestart parameter)
- [ ] Managed Identity enabled
- [ ] Automation Job Operator role assigned
- [ ] SendGrid API key configured
- [ ] Notification email updated
- [ ] HTTP callback URL obtained and saved
- [ ] Test 1 passed (forceRestart=false)
- [ ] Test 2 passed (forceRestart=true)
- [ ] Email notifications received
- [ ] Runs history shows successful executions
- [ ] Documentation updated with callback URL

---

## 🔗 References

- **Workflow File:** `infra/automation/logic-app-workflow-sendgrid.json`
- **Test Script:** `infra/automation/Test-LogicApp.ps1`
- **Runbook:** `infra/automation/Start-BearishBot-Enhanced.ps1`
- **Documentation:** 
  - `infra/automation/AZURE_AUTOMATION_SOLUTION.md`
  - `infra/automation/AZURE_MOBILE_APP_GUIDE.md`
  - `LOGIC_APP_SYNC_SUMMARY.md`

---

**Last Updated:** November 29, 2025  
**Status:** Ready for deployment ✅  
**Version:** 1.1.0 (with forceRestart parameter)
