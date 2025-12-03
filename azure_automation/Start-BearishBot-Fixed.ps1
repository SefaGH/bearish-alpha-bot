<#
.SYNOPSIS
    Azure VM üzerinde Docker Trade Bot'unu başlatır.

.DESCRIPTION
    Bu Runbook, Azure Automation'da güvenli ve stabil şekilde bot başlatır:
    1. Managed Identity ile kimlik doğrulama
    2. VM'e bağlanıp mevcut vm_run_session.py kullanarak bot başlatma
    3. Asenkron çalışma (Runbook hemen döner)
    4. Log ve hata yönetimi

.PARAMETER ResourceGroup
    VM'in bulunduğu kaynak grubu
.PARAMETER VMName
    Botun çalışacağı sanal makine adı
.PARAMETER ImageTag
    Docker imajının tag'i (örn: vm-vmboot-12, latest)
.PARAMETER DurationMinutes
    Botun kaç dakika çalışacağı (0 = sınırsız)
.PARAMETER IdempotencyToken
    Job ID olarak kullanılacak benzersiz token (Logic App tarafından set edilir)
.PARAMETER ForceRestart
    Bot zaten çalışıyorsa yeniden başlat
.PARAMETER KeyVaultName
    Azure Key Vault adı (ileride kullanılmak üzere)
.PARAMETER KvSecretNames
    Key Vault'tan çekilecek secret'lar (ileride kullanılmak üzere)
.PARAMETER StorageAccountName
    Log upload için storage account (Logic App kullanır, runbook sadece pass-through)
.PARAMETER StorageContainerName
    Log container adı (Logic App kullanır, runbook sadece pass-through)
#>

param(
    [Parameter(Mandatory=$true)]
    [string] $ResourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$true)]
    [string] $VMName = "BearishAlphaBot-VM-01",
    
    [Parameter(Mandatory=$false)]
    [string] $ImageTag = "vm-vmboot-12",
    
    [Parameter(Mandatory=$false)]
    [int] $DurationMinutes = 60,
    
    [Parameter(Mandatory=$false)]
    [string] $IdempotencyToken = "",
    
    [Parameter(Mandatory=$false)]
    [bool] $ForceRestart = $false,
    
    [Parameter(Mandatory=$false)]
    [string] $KeyVaultName = "bearish-kv",
    
    [Parameter(Mandatory=$false)]
    [string] $KvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN",
    
    [Parameter(Mandatory=$false)]
    [string] $StorageAccountName = "bearishstorage",
    
    [Parameter(Mandatory=$false)]
    [string] $StorageContainerName = "raw-logs"
)

$ErrorActionPreference = "Stop"

Write-Output "╔════════════════════════════════════════════════════════╗"
Write-Output "║       BEARISH ALPHA BOT - AZURE AUTOMATION START       ║"
Write-Output "╚════════════════════════════════════════════════════════╝"
Write-Output ""
Write-Output "Parameters:"
Write-Output "  Resource Group: $ResourceGroup"
Write-Output "  VM Name: $VMName"
Write-Output "  Image Tag: $ImageTag"
Write-Output "  Duration: $DurationMinutes minutes"
if ($IdempotencyToken) {
    Write-Output "  Job ID (Idempotency Token): $IdempotencyToken"
}
if ($ForceRestart) {
    Write-Output "  Force Restart: TRUE (will kill existing bot)"
}
Write-Output ""
Write-Output "NOTE: This runbook ONLY starts the bot on VM."
Write-Output "      Log upload is handled by Logic App after job completion."
Write-Output ""

try {
    # ---------------------------------------------------
    # 1. AUTHENTICATION
    # ---------------------------------------------------
    Write-Output "[1/5] Authenticating to Azure with Managed Identity..."
    Connect-AzAccount -Identity | Out-Null
    Write-Output "      ✅ Authenticated successfully"
    Write-Output ""

    # ---------------------------------------------------
    # 2. VM STATUS CHECK
    # ---------------------------------------------------
    Write-Output "[2/5] Checking VM status..."
    $vmStatus = Get-AzVM -ResourceGroupName $ResourceGroup -Name $VMName -Status
    $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like "PowerState/*" }).DisplayStatus
    
    Write-Output "      VM PowerState: $powerState"
    
    if ($powerState -ne "VM running") {
        throw "VM is not running. Current state: $powerState. Please start the VM first."
    }
    Write-Output "      ✅ VM is running"
    
    # ---------------------------------------------------
    # 2.5. CHECK EXISTING BOT (if ForceRestart=false)
    # ---------------------------------------------------
    if (-not $ForceRestart) {
        Write-Output ""
        Write-Output "[2.5/5] Checking for existing bot container..."
        
        $checkScript = @"
#!/bin/bash
# Check if container is RUNNING (not just exists)
if docker ps --filter 'name=^bearish-bot$' --filter 'status=running' --format '{{.Names}}' | grep -q 'bearish-bot'; then
    echo "STATUS:RUNNING"
    exit 0
else
    echo "STATUS:NOT_RUNNING"
    exit 0
fi
"@
        
        $checkResult = Invoke-AzVMRunCommand -ResourceGroupName $ResourceGroup -VMName $VMName -CommandId 'RunShellScript' -ScriptString $checkScript
        $botStatus = ""
        if ($checkResult.Value -and $checkResult.Value[0].Message) {
            $botStatus = $checkResult.Value[0].Message
            Write-Output "      Debug: Raw output = '$botStatus'"
        }
        
        if ($botStatus -match "STATUS:RUNNING") {
            throw "Bot is already running on VM! Set ForceRestart=true to override, or wait for current session to complete."
        }
        
        Write-Output "      ✅ No existing bot detected"
    } else {
        Write-Output ""
        Write-Output "[2.5/5] ForceRestart=TRUE → Will stop any existing bot"
    }
    Write-Output ""

    # ---------------------------------------------------
    # 3. PREPARE STARTUP SCRIPT
    # ---------------------------------------------------
    Write-Output "[3/5] Preparing bot startup script..."
    
    # Trading duration hesaplama
    $tradingDurationSeconds = if ($DurationMinutes -eq 0) { "" } else { $DurationMinutes * 60 }
    
    # Basit ve güvenli script - mevcut vm_run_session.py kullanıyor
    $startupScript = @"
#!/bin/bash
set -e

echo "=== BEARISH BOT STARTUP - AZURE AUTOMATION ==="
echo "Timestamp: `$(date '+%Y-%m-%d %H:%M:%S')"
echo "Image Tag: $ImageTag"
echo "Duration: $DurationMinutes minutes ($tradingDurationSeconds seconds)"
echo ""

if [ ! -f /home/azureuser/vm_run_session.py ]; then
    echo "ERROR: vm_run_session.py not found!"
    exit 1
fi

# TRADING_DURATION ayarla (eğer belirtilmişse)
if [ -n "$tradingDurationSeconds" ]; then
    echo "Setting TRADING_DURATION=$tradingDurationSeconds in bearish-bot.env"
    sudo sed -i "s/^#\\? *TRADING_DURATION=.*/TRADING_DURATION=$tradingDurationSeconds/" /home/azureuser/bearish-bot.env
else
    echo "Duration not specified, using env file default"
fi

# Bot'u başlat
echo ""
echo "Starting bot with vm_run_session.py..."
echo "Using image: bearishalphabot.azurecr.io/bearish-bot:$ImageTag"
cd /home/azureuser
sudo python3 vm_run_session.py --image "bearishalphabot.azurecr.io/bearish-bot:$ImageTag"

echo ""
echo "=== BOT STARTUP COMPLETED ==="
echo "Note: Bot is now running. Check 'docker logs bearish-bot' for output."
"@

    Write-Output "      ✅ Script prepared"
    Write-Output ""

    # ---------------------------------------------------
    # 4. SEND COMMAND TO VM
    # ---------------------------------------------------
    Write-Output "[4/5] Sending startup command to VM..."
    Write-Output "      (This may take 30-60 seconds...)"
    
    $invokeParams = @{
        ResourceGroupName = $ResourceGroup
        VMName = $VMName
        CommandId = 'RunShellScript'
        ScriptString = $startupScript
    }
    
    $result = Invoke-AzVMRunCommand @invokeParams
    
    Write-Output ""
    Write-Output "[5/5] Processing VM response..."
    
    # ---------------------------------------------------
    # 5. ANALYZE RESULT
    # ---------------------------------------------------
    $output = ""
    if ($result.Value -and $result.Value[0].Message) {
        $output = $result.Value[0].Message
    } elseif ($result.Value[1] -and $result.Value[1].Message) {
        $output = $result.Value[1].Message
    }
    
    Write-Output ""
    Write-Output "╔════════════════════════════════════════════════════════╗"
    Write-Output "║                    VM OUTPUT                           ║"
    Write-Output "╚════════════════════════════════════════════════════════╝"
    Write-Output $output
    Write-Output ""
    
    # Başarı kontrolü
    if ($output -match "BOT STARTUP COMPLETED" -or $output -match "Starting bot") {
        Write-Output "╔════════════════════════════════════════════════════════╗"
        Write-Output "║                   ✅ SUCCESS!                          ║"
        Write-Output "╚════════════════════════════════════════════════════════╝"
        Write-Output ""
        Write-Output "The Bearish Alpha Bot has been started successfully!"
        Write-Output ""
        Write-Output "📊 Session Info:"
        Write-Output "   - VM: $VMName"
        Write-Output "   - Resource Group: $ResourceGroup"
        Write-Output "   - Duration: $DurationMinutes minutes"
        Write-Output "   - Expected End: $((Get-Date).AddMinutes($DurationMinutes).ToString('yyyy-MM-dd HH:mm:ss'))"
        Write-Output ""
        Write-Output "📝 Monitoring:"
        Write-Output "   - Bot logs: SSH → docker logs bearish-bot"
        Write-Output "   - Report: Will be auto-generated after bot stops"
        Write-Output "   - Email: SendGrid notification will be sent by Logic App"
        Write-Output ""
        Write-Output "⚙️  Next Steps (Automated by Logic App):"
        Write-Output "   1. Wait for bot to complete ($DurationMinutes min)"
        Write-Output "   2. Logic App uploads logs to raw-logs container"
        Write-Output "   3. Event Grid triggers report generation"
        Write-Output "   4. SendGrid sends completion email"
    } else {
        Write-Warning "⚠️ Command executed but success confirmation not found in output"
        Write-Warning "Please check VM manually to verify bot status"
    }

} catch {
    Write-Output ""
    Write-Output "╔════════════════════════════════════════════════════════╗"
    Write-Output "║                   ❌ ERROR!                            ║"
    Write-Output "╚════════════════════════════════════════════════════════╝"
    Write-Output ""
    Write-Error "Failed to start bot: $($_.Exception.Message)"
    Write-Output ""
    Write-Output "Troubleshooting:"
    Write-Output "  1. Verify VM is running in Azure Portal"
    Write-Output "  2. Check vm_run_session.py exists on VM"
    Write-Output "  3. Verify Managed Identity has VM Contributor role"
    Write-Output "  4. Check VM's Docker service is running"
    Write-Output ""
    throw $_
}
