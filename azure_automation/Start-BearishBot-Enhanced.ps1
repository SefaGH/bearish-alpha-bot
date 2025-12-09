<#
.SYNOPSIS
    Azure VM üzerinde Docker Trade Bot'unu ASENKRON (Fire-and-Forget) modda başlatır.

.DESCRIPTION
    Bu Runbook:
    1. VM'e bağlanır ve disk temizliği yapar.
    2. Botu arka planda (Detached) başlatır.
    3. Botun ayağa kalktığını (Health Check) doğrular.
    4. Bot bitmeden hemen BAŞARIYLA sonlanır.
    
    NOT: Botun çalışıp çalışmadığının takibi Logic App (Polling Loop) tarafından yapılır.

.PARAMETER ResourceGroup
    VM'in bulunduğu kaynak grubu (Varsayılan: TradeBot)
.PARAMETER VMName
    Botun çalışacağı sanal makine adı (Varsayılan: BearishAlphaBot-VM-01)
.PARAMETER ImageTag
    Docker imajının tag'i (örn: vm-vmboot-12)
.PARAMETER DurationMinutes
    Botun kaç dakika çalışacağı
.PARAMETER IdempotencyToken
    Job ID (Takip için)
.PARAMETER ForceRestart
    Mevcut container varsa zorla durdurup yeniden başlat
#>

param(
    [Parameter(Mandatory=$false)]
    [string] $ResourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$false)]
    [string] $VMName = "BearishAlphaBot-VM-01",
    
    [Parameter(Mandatory=$false)]
    [string] $ImageTag = "",
    
    [Parameter(Mandatory=$false)]
    [int] $DurationMinutes = 60,
    
    [Parameter(Mandatory=$false)]
    [string] $IdempotencyToken = "",
    
    [Parameter(Mandatory=$false)]
    [bool] $ForceRestart = $false
)

$ErrorActionPreference = "Stop"

Write-Output "╔════════════════════════════════════════════════════════╗"
Write-Output "║   BEARISH BOT - ASYNC STARTUP (FIRE & FORGET)          ║"
Write-Output "╚════════════════════════════════════════════════════════╝"
Write-Output "Mode: Asynchronous (Logic App will monitor the running bot)"
Write-Output "Parameters:"
Write-Output "  VM: $VMName ($ResourceGroup)"
Write-Output "  Image: $ImageTag"
Write-Output "  Duration: $DurationMinutes min"
Write-Output ""

try {
    # 1. AUTHENTICATION
    Write-Output "[1/6] Authenticating..."
    Connect-AzAccount -Identity | Out-Null
    
    # 1b. FETCH IMAGE TAG FROM APP CONFIGURATION (if not provided)
    if ([string]::IsNullOrEmpty($ImageTag)) {
        Write-Output "[1b/6] Fetching image tag from Azure App Configuration..."
        try {
            $appConfigName = "appcs-bearish-bot"
            $keyName = "DOCKER_IMAGE_TAG"
            $label = "production"
            
            $imageTagValue = Get-AzAppConfigurationKeyValue `
                -Endpoint "https://$appConfigName.azconfig.io" `
                -Key $keyName `
                -Label $label `
                -ErrorAction Stop
            
            $ImageTag = $imageTagValue.Value
            Write-Output "      ✅ Image tag from App Config: $ImageTag"
        }
        catch {
            $ImageTag = "appconfig-rest-api-v2"  # Production fallback
            Write-Output "      ⚠️ App Config fetch failed, using fallback: $ImageTag"
            Write-Output "      Error: $($_.Exception.Message)"
        }
    }
    else {
        Write-Output "      ℹ️ Using provided image tag: $ImageTag"
    }
    
    # 2. VM STATUS CHECK
    Write-Output "[2/6] Checking VM status..."
    $vmStatus = Get-AzVM -ResourceGroupName $ResourceGroup -Name $VMName -Status
    $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like "PowerState/*" }).DisplayStatus
    
    if ($powerState -ne "VM running") {
        throw "VM is not running ($powerState). Logic App should start VM first."
    }
    
    # 3. PREPARE SCRIPT
    Write-Output "[3/6] Preparing startup script..."
    
    $tradingDurationSeconds = if ($DurationMinutes -eq 0) { "" } else { $DurationMinutes * 60 }
    
    $startupScript = @"
#!/bin/bash
set -e

echo "=== BOT INITIALIZATION ==="
echo "Date: `$(date)"

# --- ADIM 1: TEMİZLİK (DISK FULL ÖNLEMİ) ---
echo "1. Cleaning Docker system..."
# Kullanılmayan her şeyi sil (Volume'lar dahil)
docker system prune -af --volumes || true

# --- ADIM 2: ESKİ OTURUMU TEMİZLE ---
echo "2. Checking for existing containers..."
if docker ps -a --format '{{.Names}}' | grep -q "^bearish-bot$"; then
    echo "   Stopping and removing existing 'bearish-bot' container..."
    sudo docker stop bearish-bot 2>/dev/null || true
    sudo docker rm bearish-bot 2>/dev/null || true
fi

# --- ADIM 3: ENV AYARLARI ---
if [ -n "$tradingDurationSeconds" ]; then
    echo "3. Updating duration in env file..."
    sudo sed -i "s/^#\\? *TRADING_DURATION=.*/TRADING_DURATION=$tradingDurationSeconds/" /home/azureuser/bearish-bot.env
fi

# --- ADIM 3b: AZURE APP CONFIGURATION ENV VARS ---
echo "3b. Ensuring Azure App Configuration environment variables..."
ENV_FILE="/home/azureuser/bearish-bot.env"

# Ensure AZURE_APPCONFIG_ENDPOINT is set
if ! grep -q "^AZURE_APPCONFIG_ENDPOINT=" "$ENV_FILE"; then
    echo "   Adding AZURE_APPCONFIG_ENDPOINT..."
    echo "AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io" | sudo tee -a "$ENV_FILE" > /dev/null
fi

# Ensure AZURE_APPCONFIG_LABEL is set
if ! grep -q "^AZURE_APPCONFIG_LABEL=" "$ENV_FILE"; then
    echo "   Adding AZURE_APPCONFIG_LABEL..."
    echo "AZURE_APPCONFIG_LABEL=production" | sudo tee -a "$ENV_FILE" > /dev/null
fi

echo "   ✓ App Configuration environment variables configured"

# --- ADIM 4: BOTU BAŞLAT (PYTHON WRAPPER) ---
echo "4. Launching bot container..."
cd /home/azureuser

# vm_run_session.py zaten container'ı --detach (arka plan) modunda başlatıyor.
# --image parametresini geçiriyoruz.
sudo python3 vm_run_session.py --image "bearishalphabot.azurecr.io/bearish-bot:$ImageTag" --name "bearish-bot"

# --- ADIM 5: SAĞLIK KONTROLÜ (HEALTH CHECK) ---
echo "5. Verifying startup health (10s wait)..."
sleep 10

# Container 'running' durumunda mı?
if docker ps --filter "name=^bearish-bot$" --filter "status=running" | grep -q "bearish-bot"; then
    echo "✅ SUCCESS: Bot container is UP and RUNNING."
    echo "   Container ID: `$(docker ps --filter "name=^bearish-bot$" --format "{{.ID}}")`"
    exit 0
else
    echo "❌ CRITICAL FAILURE: Bot container died immediately!"
    echo "=== RECENT LOGS ==="
    docker logs --tail 20 bearish-bot
    exit 1
fi
"@

    # 4. EXECUTE ON VM
    Write-Output "[5/6] Sending command to VM..."
    
    $invokeParams = @{
        ResourceGroupName = $ResourceGroup
        VMName = $VMName
        CommandId = 'RunShellScript'
        ScriptString = $startupScript
    }
    
    $result = Invoke-AzVMRunCommand @invokeParams
    
    # 5. ANALYZE OUTPUT
    Write-Output "[6/6] Analyzing result..."
    
    $output = ""
    if ($result.Value -and $result.Value[0].Message) { $output = $result.Value[0].Message }
    elseif ($result.Value[1] -and $result.Value[1].Message) { $output = $result.Value[1].Message }
    
    Write-Output ""
    Write-Output "--- VM OUTPUT ---"
    Write-Output $output
    Write-Output "-----------------"
    Write-Output ""

    # 6. FINAL VERDICT
    if ($output -match "SUCCESS: Bot container is UP and RUNNING") {
        Write-Output "╔════════════════════════════════════════════════════════╗"
        Write-Output "║             ✅ RUNBOOK COMPLETED SUCCESSFULLY          ║"
        Write-Output "╚════════════════════════════════════════════════════════╝"
        Write-Output "Bot is running in background."
        Write-Output "Logic App will now start polling the container status."
    } else {
        Write-Error "Bot failed to start properly. See VM Output above for logs."
    }

} catch {
    Write-Error "Runbook execution failed: $($_.Exception.Message)"
    throw $_
}