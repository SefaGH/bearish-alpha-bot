# Update Bot on VM
# This script is meant to be run on the VM

# 1. Stop and remove existing container
sudo docker stop bearish-bot || true
sudo docker rm bearish-bot || true

# 2. Pull the new image
sudo docker pull bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-9

# 3. Run the new container
# Note: We mount logs and data to persist them on the host
sudo docker run -d \
   --name bearish-bot \
   --restart unless-stopped \
   --env-file /home/azureuser/bearish-bot.env \
   -v /mnt/bearish/logs:/app/logs \
   -v /mnt/bearish/data:/app/data \
   bearishalphabot.azurecr.io/bearish-bot:vm-vmboot-9

# 4. Verify it's running
sudo docker ps
sudo docker logs bearish-bot --tail 50
