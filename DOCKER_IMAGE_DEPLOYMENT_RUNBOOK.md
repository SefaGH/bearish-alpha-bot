# Docker Image Deployment Runbook - All Phases (Strategy Variables)

> NOTE (Reporting Sidecars Retired)
>
> In the current production environment, the reporting sidecars (log-parser / fluent-bit) are retired.
> This runbook should be followed for the main `bearish-bot` container lifecycle only.

**Date**: December 8, 2025  
**Image**: `bearishalphabot.azurecr.io/bearish-bot:strategy-config-v1`  
**Status**: Building... (ETA 10-15 minutes)

---

## 📋 Pre-Deployment Checklist

- [x] New image with All 3 Phases (38 strategy variables)
- [x] Extended migration script (`migrate_config_to_appconfig_v2.py`)
- [x] Azure App Configuration packages in requirements.txt
- [x] VM disk space verified (7.9GB available)
- [x] Old image size: 13.8GB (will be removed)
- [ ] Image build completed ← WAITING
- [ ] VM disk cleaned
- [ ] New image pulled
- [ ] Containers restarted

---

## 🚀 Deployment Steps (Execute in Order)

### Phase 0: Wait for Image Build

**Status Check Command**:
```powershell
az acr repository show-tags --name bearishalphabot --repository bearish-bot
```

**Expected Output**:
```
[
  "strategy-config-v1",    <- NEW IMAGE
  "vm-vmboot-12",
  "vm-vmboot-11",
  ...
]
```

---

### Phase 1: Clean Disk Space on VM

**Command**:
```powershell
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "echo 'Before cleanup:' ; df -h / ; echo '' ; echo 'Removing old image...' ; docker rmi bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-12 ; echo '' ; echo 'Docker cleanup...' ; docker system prune -f --volumes ; echo '' ; echo 'After cleanup:' ; df -h /" `
  --query "value[0].message" `
  --output tsv
```

**Expected Output**:
```
Before cleanup:
/dev/root        29G   22G  7.9G  73%

Removing old image...
Deleted: sha256:...

Docker cleanup...
Deleted 5 images
Reclaimed 13.8GB

After cleanup:
/dev/root        29G   8.2G  20.8G  27%
```

---

### Phase 2: Pull New Image

**Command**:
```powershell
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "docker pull bearishalphabot.azurecr.io/bearish-bot:strategy-config-v1 && docker images | grep strategy-config-v1" `
  --query "value[0].message" `
  --output tsv
```

**Expected Output**:
```
Pulling from bearish-bot
[...]
Status: Downloaded newer image for bearishalphabot.azurecr.io/bearish-bot:strategy-config-v1

bearishalphabot.azurecr.io/bearish-bot   strategy-config-v1   <image-id>   13.8GB
```

---

### Phase 3: Update Docker Compose & Restart

**Command**:
```powershell
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "cd /mnt/bearish && docker-compose down && docker-compose up -d && sleep 5 && docker ps" `
  --query "value[0].message" `
  --output tsv
```

**Expected Output**:
```
Stopping bearish-bot ... done
Removing bearish-bot ... done

Creating bearish-bot ... done
Creating log-parser ... (retired / not expected)
Creating fluent-bit ... (retired / not expected)

CONTAINER ID   IMAGE                                                   STATUS
abc123def      bearishalphabot.azurecr.io/bearish-bot:strategy-c...   Up 2 seconds
```

---

### Phase 4: Verify New Configuration Loads

**Command**:
```powershell
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "docker logs bearish-bot 2>&1 | grep -i 'app config\|strategy\|config\|load' | tail -30" `
  --query "value[0].message" `
  --output tsv
```

**Expected Output** (should show):
```
[CONFIG] Loading from Azure App Configuration...
[OK] Connected to App Configuration
[OK] Loaded 64 settings (26 base + 38 strategy)
[OK] Loaded strategy variables (Phase 1, 2, 3)
[STRATEGY] OB enabled, STR enabled, ML regime confidence loaded
```

---

### Phase 5: Health Check

**Command**:
```powershell
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "curl -s http://localhost:8000/health | jq . || echo 'No health endpoint'" `
  --query "value[0].message" `
  --output tsv
```

**Expected Output**:
```
{
  "status": "healthy",
  "uptime": "45 seconds",
  "config_loaded": true,
  "strategy_vars": 38
}
```

---

## 🔄 Rollback Plan

If issues occur after deployment:

### Quick Rollback (< 5 minutes)

```powershell
# Go back to last working image
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "cd /mnt/bearish && sed -i 's/:strategy-config-v1/:vm-vmboot-12/g' docker-compose.yml && docker-compose up -d" `
  --query "value[0].message" `
  --output tsv
```

### Full Rollback

```powershell
# If old image still exists in registry
az vm run-command invoke `
  --resource-group "TRADEBOT" `
  --name "BearishAlphaBot-VM-01" `
  --command-id RunShellScript `
  --scripts "docker pull bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-12 && cd /mnt/bearish && docker-compose up -d" `
  --query "value[0].message" `
  --output tsv
```

---

## ⚠️ Important Notes

1. **Disk Space**: Old image will free 13.8GB
2. **Downtime**: ~30 seconds (containers restart)
3. **Data**: No data loss (logs/data volumes persist)
4. **Config Loading**: Graceful fallback to ENV vars if App Config unavailable
5. **Strategy Variables**: Phase 1, 2, 3 all included (38 total)

---

## 🎯 What Changed in New Image

### Added to Script
- ✅ `scripts/migrate_config_to_appconfig_v2.py` (800+ lines)
- ✅ 38 strategy variables (8+13+17)
- ✅ 3-phase deployment system
- ✅ Dry-run mode for testing

### Added to Application
- ✅ Azure App Configuration support in `src/config/live_trading_config.py`
- ✅ Strategy variable loading
- ✅ Graceful fallback handling

### Added to Dependencies
- ✅ `azure-appconfiguration>=1.4.0`
- ✅ `azure-appconfiguration-provider>=2.3.1`

---

## 📊 Monitoring After Deployment

### Check Config Loading
```powershell
# Monitor logs for config operations
az vm run-command invoke --resource-group TRADEBOT --name BearishAlphaBot-VM-01 `
  --command-id RunShellScript `
  --scripts "docker logs -f bearish-bot 2>&1 | grep -i 'config\|strategy\|load'" `
  --query "value[0].message"
```

### Check Strategy Variables
```powershell
# Verify all 38 variables loaded
az vm run-command invoke --resource-group TRADEBOT --name BearishAlphaBot-VM-01 `
  --command-id RunShellScript `
  --scripts "docker exec bearish-bot python -c \"from src.config import LiveTradingConfiguration; c = LiveTradingConfiguration.load(); print(f'Total settings: {len(c.config)}')\"" `
  --query "value[0].message"
```

### Test Strategy Toggle (Phase 1)
```powershell
# Change strategy setting in App Config, verify bot picks it up
az appconfig kv set --name appcs-bearish-bot `
  --key "BearishAlphaBot/STRATEGY_OB_ENABLED" `
  --value "false" `
  --label production

# Wait 30 seconds, then check logs
az vm run-command invoke --resource-group TRADEBOT --name BearishAlphaBot-VM-01 `
  --command-id RunShellScript `
  --scripts "docker logs bearish-bot 2>&1 | grep -i 'strategy.*false\|ob.*disabled'"
```

---

## ✅ Success Criteria

- [ ] Image built and tagged as `strategy-config-v1`
- [ ] Old image (13.8GB) removed from VM
- [ ] New image pulled successfully
- [ ] Containers running (bearish-bot; reporting sidecars retired)
- [ ] Logs show config loading from App Configuration
- [ ] 64 settings loaded (26 base + 38 strategy)
- [ ] No errors in first 5 minutes of logs
- [ ] Strategy toggles work via App Config changes

---

**Status**: READY TO DEPLOY (Waiting for image build)  
**Risk Level**: LOW (Old image backed up, full rollback available)  
**Estimated Deployment Time**: 10 minutes (including disk cleanup + pull)

