# Deployment Summary - Reporting Automation

## Overview
A new Docker image (`vm-vmboot-7`) has been built and pushed to ACR. This image contains the latest changes for:
1.  **Reporting Automation**: The bot now automatically triggers the Azure Function to generate and email a PDF report upon shutdown.
2.  **Logger Enhancements**: The logger now exposes the active log filename for the reporting module.
3.  **Dependencies**: `aiohttp` is used for the trigger request.

## Deployment Steps (VM)
To deploy this new version on the VM, execute the following commands:

```bash
# 1. Connect to VM
ssh azureuser@<VM_IP>

# 2. Stop and Remove Old Container
sudo docker stop bearish-bot || true
sudo docker rm bearish-bot || true

# 3. Pull New Image
sudo docker pull bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-7

# 4. Run New Container
sudo docker run -d \
   --name bearish-bot \
   --env-file /home/azureuser/bearish-bot.env \
   -v /mnt/bearish/logs:/app/logs \
   -v /mnt/bearish/data:/app/data \
   bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-7
```

## Verification
1.  **Check Logs**: `sudo docker logs -f bearish-bot`
2.  **Wait for Shutdown**: Let the bot run for its configured duration (or stop it manually).
3.  **Check Email**: Verify that the report email is received.
