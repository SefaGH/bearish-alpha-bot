# Cleanup Disk Space on VM
# This script is meant to be run on the VM

echo "Disk usage before cleanup:"
df -h

# 1. Prune Docker system (images, containers, networks)
echo "Pruning Docker system..."
sudo docker system prune -a -f --volumes

# 2. Check for large log files in /var/log and truncate them if needed (be careful)
# For now, just listing them
echo "Large files in /var/log:"
sudo find /var/log -type f -size +100M

# 3. Check /mnt/bearish/logs size
echo "Size of /mnt/bearish/logs:"
du -sh /mnt/bearish/logs

# 4. Check /mnt/bearish/data size
echo "Size of /mnt/bearish/data:"
du -sh /mnt/bearish/data

echo "Disk usage after cleanup:"
df -h
