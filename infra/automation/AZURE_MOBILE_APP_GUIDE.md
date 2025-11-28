# 📱 Azure Mobile App - Complete Usage Guide

**Last Updated**: November 29, 2025  
**Feature**: Smart Status Detection with Force Restart

---

## 🎯 Overview

This guide shows you how to start and manage the Bearish Alpha Bot directly from your iPhone using the Azure mobile app. The runbook now includes **smart status detection** that prevents duplicate executions.

---

## 📲 Prerequisites

### 1. Install Azure Mobile App
- Open **App Store** on iPhone
- Search: "Microsoft Azure"
- Install and sign in with your Azure account

### 2. Verify Access
You need these permissions:
- **Reader** on Resource Group
- **Automation Operator** on Automation Account

---

## 🚀 Step-by-Step: Starting the Bot

### Step 1: Open Azure App
1. Launch **Azure** app on your iPhone
2. Sign in if prompted

### Step 2: Navigate to Automation Account
1. Tap **☰** (hamburger menu, top-left)
2. Tap **All resources**
3. Scroll or search for: `tradebot-automation`
4. Tap on `tradebot-automation`

### Step 3: Find the Runbook
1. Scroll down to **Runbooks** section
2. Tap on `Start-BearishBot-Enhanced`
3. You'll see runbook details

### Step 4: Start the Runbook

#### ⚠️ IMPORTANT: No Direct "Start" Button

The Azure mobile app **does NOT show a "Start" button** for runbooks. Instead, you need to use one of these methods:

---

## 📱 Method 1: Azure Portal via Mobile Browser (Recommended)

### Step-by-Step:

1. **Open Safari** on your iPhone
2. Navigate to: `https://portal.azure.com`
3. Sign in with your Azure credentials
4. Tap **≡** (menu) → **All resources**
5. Search and tap: `tradebot-automation`
6. Tap **Runbooks** (left sidebar)
7. Tap `Start-BearishBot-Enhanced`
8. **Tap "Start" button** at the top
9. Enter parameters:
   ```
   durationMinutes: 60
   imageTag: vm-vmboot-9
   forceRestart: ☐ (leave unchecked)
   ```
10. Tap **OK**
11. Job will start immediately

### Expected Result:

**If bot is NOT running:**
```
Status: Running
Message: "Step 2: Checking if bot container is already running..."
         "Container status: NOT_FOUND"
         "✅ No existing container found. Proceeding with fresh start."
```

**If bot IS running:**
```
Status: Failed
Message: "❌ Bot is already RUNNING. Aborting to prevent duplicate execution."
         "Container started at: 2025-11-29T10:30:00Z"
         "Container uptime: Up 25 minutes"
         "Recent container logs: [logs shown here]"
         "To force restart, run with -forceRestart parameter"
```

---

## 📱 Method 2: iOS Shortcuts (Best User Experience)

### Setup (One-Time):

1. **Install Shortcuts app** (pre-installed on iOS 13+)
2. Open **Shortcuts** app
3. Tap **+** (Create Shortcut)
4. Follow actions below

### Shortcut Actions:

#### Action 1: Ask for Input
- **Type**: Number
- **Question**: "Trading duration (minutes)?"
- **Default Answer**: 60

#### Action 2: Text (Dictionary)
```json
{
  "durationMinutes": [Provided Input],
  "imageTag": "vm-vmboot-9",
  "keyVaultName": "bearish-kv"
}
```

#### Action 3: Run Script Over SSH

> **Note**: This requires SSH access to the VM. Alternatively, use Azure CLI via Shortcuts (requires Azure CLI shortcut setup).

**For Azure CLI approach:**

1. Install **a-Shell** app from App Store (Azure CLI for iOS)
2. Configure Azure credentials
3. Use script:
```bash
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=[Provided Input] imageTag=vm-vmboot-9
```

#### Action 4: Show Notification
- **Title**: "🐻 Trading Bot Started"
- **Body**: "Duration: [Provided Input] minutes"

### Usage:
1. Tap Shortcut icon
2. Enter duration
3. Wait for notification
4. Check Azure Portal for job status

---

## 📱 Method 3: Azure CLI via Termius (Advanced)

### Setup:

1. **Install Termius** app (SSH client)
2. **Install Azure CLI on VM** (if not already)
3. Create SSH saved session to VM

### Usage:

1. Open **Termius** app
2. Connect to: `azureuser@20.73.171.66`
3. Run command:
```bash
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters durationMinutes=60 imageTag=vm-vmboot-9
```

4. To force restart:
```bash
az automation runbook start \
  --name Start-BearishBot-Enhanced \
  --automation-account-name tradebot-automation \
  --resource-group TradeBot \
  --parameters '{"durationMinutes":60,"imageTag":"vm-vmboot-9","forceRestart":true}'
```

---

## 🔍 Monitoring Job Status

### Via Azure Mobile App:

1. Open **Azure** app
2. Navigate to: `tradebot-automation`
3. Tap **Jobs** (under Runbooks section)
4. Tap on the latest job
5. View:
   - **Status**: Queued → Running → Completed/Failed
   - **Output**: Live logs stream
   - **Start Time**: Job start timestamp
   - **Duration**: Elapsed time

### Via Azure Portal (Mobile Browser):

1. Open Safari → `portal.azure.com`
2. Navigate to: `tradebot-automation` → **Jobs**
3. Tap latest job → **Output** tab
4. Refresh to see live logs

### Via Telegram:

- Bot sends notifications automatically
- No additional setup needed
- Receive trade updates in real-time

---

## 🎛️ Parameters Explained

### Required Parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `durationMinutes` | Integer | Trading session length (1-85) | `60` |

### Optional Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imageTag` | String | `vm-vmboot-9` | Docker image tag |
| `keyVaultName` | String | `bearish-kv` | Key Vault name |
| `forceRestart` | Boolean | `false` | Force restart if bot running |

### Force Restart Usage:

**When to use `forceRestart=true`:**
- Bot crashed but container still running
- Need to apply urgent config changes
- Testing/debugging scenarios

**⚠️ Warning**: Force restart will **stop active trading** and restart fresh.

---

## 📊 Execution Scenarios

### Scenario 1: Normal Start (Bot Not Running)

**Parameters:**
```json
{
  "durationMinutes": 60,
  "imageTag": "vm-vmboot-9"
}
```

**Expected Output:**
```
[2025-11-29 10:00:00] [INFO] Step 1: Checking VM status...
[2025-11-29 10:00:05] [INFO] Current VM state: VM running
[2025-11-29 10:00:05] [INFO] Step 2: Checking if bot container is already running...
[2025-11-29 10:00:10] [INFO] Container status: NOT_FOUND
[2025-11-29 10:00:10] [INFO] ✅ No existing container found. Proceeding with fresh start.
[2025-11-29 10:00:15] [INFO] Step 3: Checking for concurrent executions...
[2025-11-29 10:00:20] [SUCCESS] ✅ Bot started successfully
```

**Duration**: ~11 minutes (includes VM overhead)  
**Result**: ✅ **COMPLETED**

---

### Scenario 2: Duplicate Prevention (Bot Already Running)

**Parameters:**
```json
{
  "durationMinutes": 60,
  "imageTag": "vm-vmboot-9",
  "forceRestart": false
}
```

**Expected Output:**
```
[2025-11-29 10:00:00] [INFO] Step 1: Checking VM status...
[2025-11-29 10:00:05] [INFO] Current VM state: VM running
[2025-11-29 10:00:05] [INFO] Step 2: Checking if bot container is already running...
[2025-11-29 10:00:10] [INFO] Container status: RUNNING
[2025-11-29 10:00:10] [WARNING] ❌ Bot is already RUNNING. Aborting to prevent duplicate execution.
[2025-11-29 10:00:10] [INFO] Container started at: 2025-11-29T09:30:00Z
[2025-11-29 10:00:10] [INFO] Container uptime: Up 30 minutes
[2025-11-29 10:00:10] [INFO] Recent container logs:
[2025-11-29 10:00:10] [INFO]   [INFO] Trading loop active
[2025-11-29 10:00:10] [INFO]   [INFO] Heartbeat - is_running=True
[2025-11-29 10:00:10] [INFO] To force restart, run with -forceRestart parameter
[2025-11-29 10:00:10] [ERROR] Bot already running. Use -forceRestart to override.
```

**Duration**: ~30 seconds (early abort)  
**Result**: ❌ **FAILED** (expected behavior)

---

### Scenario 3: Force Restart (Bot Already Running)

**Parameters:**
```json
{
  "durationMinutes": 60,
  "imageTag": "vm-vmboot-9",
  "forceRestart": true
}
```

**Expected Output:**
```
[2025-11-29 10:00:00] [INFO] Step 1: Checking VM status...
[2025-11-29 10:00:05] [INFO] Current VM state: VM running
[2025-11-29 10:00:05] [INFO] Step 2: Checking if bot container is already running...
[2025-11-29 10:00:10] [INFO] Container status: RUNNING
[2025-11-29 10:00:10] [WARNING] ⚠️ Bot is RUNNING but forceRestart=true. Stopping existing container...
[2025-11-29 10:00:15] [INFO] Stopping container: bearish-bot
[2025-11-29 10:00:45] [SUCCESS] ✅ Existing container stopped. Proceeding with fresh start.
[2025-11-29 10:00:50] [INFO] Step 3: Checking for concurrent executions...
[2025-11-29 10:01:00] [SUCCESS] ✅ Bot started successfully
```

**Duration**: ~11 minutes + 30 seconds (stop timeout)  
**Result**: ✅ **COMPLETED**

---

## 🔔 Notifications

### Push Notifications (Azure Mobile App):

**Enable in Azure App:**
1. Tap **Profile** (bottom-right avatar)
2. Tap **Settings** → **Notifications**
3. Enable:
   - ✅ **Automation job status changes**
   - ✅ **Runbook execution failures**

**You'll receive:**
- 🔵 Job started notification
- ✅ Job completed notification
- ❌ Job failed notification (with reason)

### Email Notifications (SendGrid):

**Setup required:**
1. Deploy Logic App (optional)
2. Configure SendGrid API key
3. Set email addresses in workflow

**You'll receive:**
- ✅ Success emails with execution summary
- ❌ Failure emails with error details

### Telegram Notifications:

**Already configured:**
- Bot sends trade notifications automatically
- Real-time updates during execution
- No additional setup needed

---

## 🆘 Troubleshooting

### Issue: "No Start Button in Azure App"

**Cause**: Azure mobile app doesn't support starting runbooks directly.

**Solution**:
- Use **Azure Portal via Safari** (Method 1)
- Or create **iOS Shortcut** (Method 2)
- Or use **SSH/Azure CLI** (Method 3)

---

### Issue: Job Fails with "Bot already running"

**Cause**: Container status check detected running bot.

**Solution**:
1. **Check if bot should be running:**
   ```bash
   ssh azureuser@20.73.171.66
   sudo docker ps --filter "name=bearish-bot"
   ```

2. **If bot is stuck/crashed:**
   - Use `forceRestart=true` parameter
   - Or manually stop: `sudo docker stop bearish-bot`

3. **If bot is trading normally:**
   - Wait for session to complete
   - Do NOT force restart during active trades

---

### Issue: Job Stays "Queued" for Long Time

**Cause**: VM taking long to start or Azure Agent not ready.

**Solution**:
1. Check VM status in Azure app
2. If VM is "Stopped", manually start it:
   - Navigate to VM in Azure app
   - Tap **Start** button
3. Wait 2-3 minutes, then retry runbook

---

### Issue: "Permission Denied" Error

**Cause**: Managed Identity missing required permissions.

**Solution**:
```bash
# On PC, run:
az role assignment create \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --role "Virtual Machine Contributor" \
  --scope "/subscriptions/<SUB_ID>/resourceGroups/TradeBot"

az keyvault set-policy \
  --name bearish-kv \
  --object-id <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --secret-permissions get list
```

---

### Issue: Container Logs Show Errors

**Check logs in Azure app:**
1. Navigate to job → **Output** tab
2. Look for error messages
3. Common issues:
   - **"BINGX-KEY not found"**: Key Vault access issue
   - **"Image not found"**: Wrong `imageTag` parameter
   - **"Connection refused"**: BingX API issue

**Solution**:
- Verify Key Vault secrets exist
- Check `imageTag` parameter spelling
- Test BingX API connectivity

---

## 💰 Cost Considerations

### Per Execution:
- **Automation Job**: ~$0.002/minute
- **VM Compute**: ~$0.10/hour (auto-deallocates after)
- **Logic App** (if used): ~$0.000025/action

### Monthly Cost (1 hour/day):
- **Automation**: ~$3.60/month
- **VM**: ~$3.00/month
- **Total**: **~$6.60/month**

### Cost Optimization Tips:
- Use shorter durations for testing
- VM auto-deallocates (don't pay when idle)
- Monitor job history to avoid duplicate runs

---

## ✅ Quick Reference

### Fast Commands:

```bash
# Check if bot is running (SSH to VM)
sudo docker ps --filter "name=bearish-bot"

# Stop bot manually
sudo docker stop bearish-bot && sudo docker rm bearish-bot

# View bot logs
sudo docker logs bearish-bot --tail 50

# Check VM status
az vm get-instance-view -g TradeBot -n BearishAlphaBot-VM-01 --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" -o tsv
```

### Parameter Templates:

**Quick Test (5 minutes):**
```json
{"durationMinutes": 5, "imageTag": "vm-vmboot-9"}
```

**Standard Session (1 hour):**
```json
{"durationMinutes": 60, "imageTag": "vm-vmboot-9"}
```

**Force Restart:**
```json
{"durationMinutes": 60, "imageTag": "vm-vmboot-9", "forceRestart": true}
```

---

## 🎯 Best Practices

### 1. **Always Test with Short Duration First**
```json
{"durationMinutes": 5, "imageTag": "vm-vmboot-9"}
```

### 2. **Monitor First Execution**
- Watch job logs in Azure app
- Check Telegram for trade notifications
- Verify VM deallocates after completion

### 3. **Use Force Restart Sparingly**
- Only use when bot is truly stuck
- Check logs first to confirm bot is inactive
- Avoid force restart during active trades

### 4. **Check Status Before Starting**
```bash
# SSH to VM
ssh azureuser@20.73.171.66
sudo docker ps --filter "name=bearish-bot"
```

### 5. **Enable Notifications**
- Azure mobile app push notifications
- Telegram bot notifications
- Monitor both for complete visibility

---

## 📞 Support

### Documentation:
- **Complete Guide**: `AZURE_AUTOMATION_SOLUTION.md`
- **Implementation Summary**: `ISSUE_434_IMPLEMENTATION_SUMMARY.md`
- **Troubleshooting**: See above section

### Logs:
- **Azure Portal**: Job Output tab
- **VM Logs**: `ssh azureuser@20.73.171.66; sudo docker logs bearish-bot`
- **Telegram**: Real-time notifications

### GitHub:
- **Report Issues**: GitHub Issue #434
- **Feature Requests**: Create new issue with label `enhancement`

---

## 🎉 Summary

**You can now manage your trading bot from iPhone!**

✅ **Safe**: Prevents duplicate executions  
✅ **Smart**: Detects container status automatically  
✅ **Flexible**: Force restart option when needed  
✅ **Transparent**: Shows logs and status details  
✅ **Cost-Efficient**: Auto-deallocates VM after execution  

**Recommended Method**: Azure Portal via Safari (easiest, most reliable)

---

**Last Updated**: November 29, 2025  
**Version**: 1.1.0 (Status Check Feature)  
**Tested On**: iPhone iOS 17.x, Azure App v4.x
