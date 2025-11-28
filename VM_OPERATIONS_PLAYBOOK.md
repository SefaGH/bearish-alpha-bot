# 📘 Bearish Alpha Bot - VM Operations Playbook

This guide covers common operational tasks for managing the Bearish Alpha Bot on the Azure VM (`BearishAlphaBot-VM-01`).

## 1. 🚀 Deployment & Updates

### Update & Restart Bot (One-Step)
To pull the latest code/image and restart the bot on the VM:

```powershell
# From your local machine
az vm run-command invoke -g TradeBot -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "python3 /home/azureuser/scripts/vm_run_session.py --duration 1200"
```
*Note: This uses the helper script inside the VM to handle Docker operations.*

## 2. 🔍 Monitoring & Logs

### 📥 Download Full Log File (Recommended)
Since `az vm run-command` truncates output, use this script to download the **latest** full log file to your local machine:

```powershell
.\scripts\fetch_latest_log.ps1
```
*   **Prerequisite:** SSH Key must be in `~/.ssh/id_rsa` (or configured in SSH agent).
*   **Output:** Logs are saved to `logs/downloaded/`.

### View Live Logs (Tail)
To see the last 50 lines of the running bot:

```powershell
az vm run-command invoke -g TradeBot -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "docker logs --tail 50 bearish-bot"
```

### Check Container Status
```powershell
az vm run-command invoke -g TradeBot -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "docker ps"
```

## 3. 🛠️ Troubleshooting

### "BingX API Error" during Shutdown
If you see errors like `bingx GET ...` during shutdown:
1.  This is usually a network glitch when the container is stopping.
2.  Verify if positions were actually closed by checking the next run's startup logs or using the BingX app/site.
3.  The bot attempts to close positions *before* killing the connection, but network race conditions can occur.

### VM Connection Issues
If `fetch_latest_log.ps1` fails:
1.  Check your VPN/Internet connection.
2.  Verify VM IP: `az vm show -d -g TradeBot -n BearishAlphaBot-VM-01 --query publicIps -o tsv`
3.  Ensure your IP is allowed in the VM's Network Security Group (NSG) for port 22 (SSH).

## 4. 📂 File Locations (On VM)

*   **Logs:** `/mnt/bearish/logs/` (Mapped to container `/app/logs`)
*   **Data:** `/mnt/bearish/data/` (Mapped to container `/app/data`)
*   **Scripts:** `/home/azureuser/scripts/`
