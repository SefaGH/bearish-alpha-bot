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

Write-Host "🔍 Finding latest log file on VM..." -ForegroundColor Cyan
# We use run-command to find the filename to avoid complex SSH parsing
$cmd = "ls -t /mnt/bearish/logs/live_trading_*.log | head -n 1"
$res = az vm run-command invoke -g $ResourceGroup -n $VMName --command-id RunShellScript --scripts $cmd --query 'value[0].message' -o tsv

# Clean up output (sometimes contains extra newlines/stdout markers)
# We take the last non-empty line which is usually the file path
# Force array to avoid "char" indexing on single string result
$lines = @($res -split "`n" | Where-Object { $_ -match "live_trading" })

if ($lines.Count -eq 0) {
    Write-Error "No log file found on VM."
    exit 1
}

# Take the first match
$remotePath = $lines[0].Trim()

if (-not $remotePath) {
    Write-Error "No log file path resolved."
    exit 1
}

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
