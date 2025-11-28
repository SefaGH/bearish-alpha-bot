param (
    [string]$ResourceGroup = "TradeBot",
    [string]$VMName = "BearishAlphaBot-VM-01",
    [string]$RemoteUser = "azureuser",
    [string]$LocalDest = ".\logs\downloaded"
)

$ErrorActionPreference = "Stop"

# Create local dir
if (-not (Test-Path $LocalDest)) {
    New-Item -ItemType Directory -Force -Path $LocalDest | Out-Null
}

Write-Host "🔍 Getting VM IP..." -ForegroundColor Cyan
$ip = az vm show -d -g $ResourceGroup -n $VMName --query publicIps -o tsv

if (-not $ip) {
    Write-Error "Could not find VM IP."
    exit 1
}

Write-Host "🔍 Finding latest log file on VM (via SSH)..." -ForegroundColor Cyan
# Use SSH to find the filename to avoid Azure Agent locks
$findCmd = "ls -t /mnt/bearish/logs/live_trading_*.log | head -n 1"
$remotePath = ssh -o StrictHostKeyChecking=no "${RemoteUser}@${ip}" $findCmd

if (-not $remotePath) {
    Write-Error "No log file found on VM."
    exit 1
}

# Clean up output (trim whitespace)
$remotePath = $remotePath.Trim()

Write-Host "⬇️ Downloading $remotePath from $ip..." -ForegroundColor Cyan
try {
    # Use SCP to download
    # Note: Assumes your SSH key is in %USERPROFILE%\.ssh\id_rsa or loaded in ssh-agent
    scp -o StrictHostKeyChecking=no "${RemoteUser}@${ip}:${remotePath}" $LocalDest
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Log downloaded to: $LocalDest" -ForegroundColor Green
        Get-ChildItem "$LocalDest\*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    } else {
        throw "SCP command returned exit code $LASTEXITCODE"
    }
}
catch {
    Write-Error "SCP failed. Ensure you have SSH access to the VM and your private key is in %USERPROFILE%\.ssh. Error: $_"
}
