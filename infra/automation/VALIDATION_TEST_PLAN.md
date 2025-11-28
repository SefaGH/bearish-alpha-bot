# End-to-End Validation Test Plan

**Objective:** Validate Logic App + Runbook + forceRestart functionality  
**Date:** November 29, 2025  
**Duration:** ~30 minutes

---

## 🎯 Test Scenarios

### Scenario 1: Normal Start - Bot Not Running ✅
**Purpose:** Verify normal execution flow when no container exists

**Prerequisites:**
- No container running on VM
- Logic App deployed with forceRestart parameter
- SendGrid configured

**Test Steps:**
```powershell
# Step 1: Ensure no container is running
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps -a"

# Step 2: Trigger Logic App
$url = "<YOUR_CALLBACK_URL>"
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-11",
        "forceRestart": false
    }'
```

**Expected Results:**
- ✅ Logic App runs successfully (200 OK)
- ✅ Runbook job created in Automation Account
- ✅ VM starts (if stopped)
- ✅ Container status check: NOT_FOUND or STOPPED
- ✅ Container starts with vm-vmboot-11
- ✅ Trading executes for 5 minutes
- ✅ Container stops automatically
- ✅ VM deallocates after cleanup
- ✅ Email notification sent (success)

**Validation Commands:**
```powershell
# Check Logic App run
az logic workflow show --name bearish-bot-orchestrator --resource-group TradeBot

# Check runbook job (get job ID from Logic App response)
az automation job show --name <JOB_ID> --automation-account-name tradebot-automation --resource-group TradeBot

# Check VM status
az vm show --name BearishAlphaBot-VM-01 --resource-group TradeBot --query powerState -o tsv
```

---

### Scenario 2: Normal Start - Bot Already Running ⚠️
**Purpose:** Verify duplicate prevention (status check aborts execution)

**Prerequisites:**
- Container already running on VM
- forceRestart=false (default)

**Test Steps:**
```powershell
# Step 1: Start a long-running session (10 minutes)
az automation runbook start \
    --name Start-BearishBot-Enhanced \
    --automation-account-name tradebot-automation \
    --resource-group TradeBot \
    --parameters durationMinutes=10 imageTag=vm-vmboot-11

# Step 2: Wait 1 minute for container to start

# Step 3: Try to start another session (should abort)
$url = "<YOUR_CALLBACK_URL>"
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-11",
        "forceRestart": false
    }'
```

**Expected Results:**
- ✅ Logic App runs successfully (200 OK)
- ✅ Runbook job created
- ✅ Container status check: RUNNING
- ✅ Runbook logs: "Container is already running (started at: ..., uptime: ...)"
- ✅ Execution aborts gracefully (no error)
- ✅ Original container continues running
- ✅ Email notification sent (status: aborted/info)
- ❌ No duplicate trading session created

**Validation Commands:**
```powershell
# Check container status
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}'"

# Should show only 1 container running
```

---

### Scenario 3: Force Restart - Bot Already Running 🔄
**Purpose:** Verify forceRestart parameter stops existing container and starts new one

**Prerequisites:**
- Container already running on VM
- forceRestart=true

**Test Steps:**
```powershell
# Step 1: Start a long-running session
az automation runbook start \
    --name Start-BearishBot-Enhanced \
    --automation-account-name tradebot-automation \
    --resource-group TradeBot \
    --parameters durationMinutes=20 imageTag=vm-vmboot-11

# Step 2: Wait 2 minutes for container to start and begin trading

# Step 3: Note container start time
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}'"

# Step 4: Force restart with new session
$url = "<YOUR_CALLBACK_URL>"
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-11",
        "forceRestart": true
    }'

# Step 5: Check container start time again (should be newer)
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}'"
```

**Expected Results:**
- ✅ Logic App runs successfully (200 OK)
- ✅ Runbook job created
- ✅ Container status check: RUNNING
- ✅ Runbook logs: "Force restart enabled, stopping existing container..."
- ✅ Existing container stopped (30s graceful timeout)
- ✅ Old container removed
- ✅ New container started with vm-vmboot-11
- ✅ New session runs for 5 minutes
- ✅ Container start time updated (newer timestamp)
- ✅ Email notification sent (success)
- ❌ No overlap between old and new sessions

**Validation:**
- Container CreatedAt timestamp should be newer
- Only 1 container should be running (old one removed)

---

### Scenario 4: Image Update with Force Restart 🆕
**Purpose:** Verify new image deployment workflow

**Prerequisites:**
- New image available (e.g., vm-vmboot-12)
- Container running with old image (vm-vmboot-11)

**Test Steps:**
```powershell
# Step 1: Verify current image
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps --format 'table {{.Image}}\t{{.Names}}\t{{.Status}}'"

# Step 2: Deploy new image with force restart
$url = "<YOUR_CALLBACK_URL>"
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-12",
        "forceRestart": true
    }'

# Step 3: Verify new image is running
az vm run-command invoke \
    --resource-group TradeBot \
    --name BearishAlphaBot-VM-01 \
    --command-id RunShellScript \
    --scripts "docker ps --format 'table {{.Image}}\t{{.Names}}\t{{.Status}}'"
```

**Expected Results:**
- ✅ Old container (vm-vmboot-11) stopped
- ✅ New image (vm-vmboot-12) pulled from ACR
- ✅ New container started with vm-vmboot-12
- ✅ Trading executes with new code
- ✅ No errors during image pull
- ✅ Email notification sent

---

### Scenario 5: Parameter Validation ✔️
**Purpose:** Verify parameter validation and error handling

**Test Steps:**
```powershell
# Test 1: Missing required parameter (durationMinutes)
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "imageTag": "vm-vmboot-11"
    }'

# Expected: 400 Bad Request or validation error

# Test 2: Invalid duration (> 85 minutes)
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 100,
        "imageTag": "vm-vmboot-11"
    }'

# Expected: Validation error or clamped to 85

# Test 3: Invalid imageTag (non-existent)
Invoke-RestMethod -Method POST -Uri $url `
    -ContentType "application/json" `
    -Body '{
        "durationMinutes": 5,
        "imageTag": "vm-vmboot-999"
    }'

# Expected: Docker pull error, graceful failure, email notification
```

**Expected Results:**
- ✅ Missing parameters rejected
- ✅ Invalid values handled gracefully
- ✅ Error notifications sent
- ✅ No partial executions

---

## 📊 Validation Checklist

After completing all scenarios, verify:

### Logic App
- [ ] HTTP trigger accepts forceRestart parameter
- [ ] Parameter validation works correctly
- [ ] Managed Identity has correct permissions
- [ ] Runs history shows all test executions
- [ ] Error handling works (failed tests logged)

### Runbook
- [ ] Container status check works (RUNNING/STOPPED/NOT_FOUND)
- [ ] forceRestart=false aborts when container is running
- [ ] forceRestart=true stops and restarts container
- [ ] 30-second graceful stop timeout works
- [ ] Image pull works for all valid tags
- [ ] VM lifecycle management works (start/stop/deallocate)

### Email Notifications
- [ ] Success emails received (all passed tests)
- [ ] Failure emails received (invalid parameter tests)
- [ ] Emails contain correct job details (ID, duration, image)
- [ ] Email format is readable and actionable

### Container Management
- [ ] No duplicate containers created
- [ ] Old containers properly removed
- [ ] New containers start successfully
- [ ] Container logs accessible
- [ ] Resource cleanup works

---

## 🐛 Common Issues

### Issue: Container not stopping with forceRestart=true
**Symptom:** Old container still running after force restart  
**Check:** Runbook logs for stop timeout errors  
**Fix:** Increase timeout in `Stop-ExistingContainer` function

### Issue: Email not sent
**Symptom:** No email received after test  
**Check:** 
- SendGrid API key valid
- Sender email verified in SendGrid
- Logic App runs history for error details
**Fix:** Update SendGrid configuration

### Issue: "Unauthorized" when triggering runbook
**Symptom:** Logic App fails with 403 error  
**Check:** Managed Identity has "Automation Job Operator" role  
**Fix:** Re-assign role via Azure Portal or CLI

---

## 📈 Performance Metrics

Record these metrics for each test:

| Metric | Target | Actual |
|--------|--------|--------|
| Logic App response time | < 2s | ___ |
| Runbook start time | < 30s | ___ |
| VM start time (if stopped) | < 60s | ___ |
| Container start time | < 45s | ___ |
| Container stop time (force) | < 30s | ___ |
| Total execution time (5 min test) | ~6-7 min | ___ |
| Email delivery time | < 60s | ___ |

---

## ✅ Test Results Template

```
Date: _______________
Tester: _______________
Environment: Azure Production (TradeBot)

Scenario 1: Normal Start - Bot Not Running
Status: ☐ PASS  ☐ FAIL
Notes: _________________________________

Scenario 2: Normal Start - Bot Already Running
Status: ☐ PASS  ☐ FAIL
Notes: _________________________________

Scenario 3: Force Restart - Bot Already Running
Status: ☐ PASS  ☐ FAIL
Notes: _________________________________

Scenario 4: Image Update with Force Restart
Status: ☐ PASS  ☐ FAIL
Notes: _________________________________

Scenario 5: Parameter Validation
Status: ☐ PASS  ☐ FAIL
Notes: _________________________________

Overall Status: ☐ ALL PASS  ☐ SOME FAIL
Production Ready: ☐ YES  ☐ NO (reason: _________)
```

---

**Document Version:** 1.0  
**Last Updated:** November 29, 2025  
**Related Docs:** LOGIC_APP_DEPLOYMENT_GUIDE.md, AZURE_AUTOMATION_SOLUTION.md
