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
    Docker imajının tag'i (örn: vm-vmboot-11, latest)
.PARAMETER DurationMinutes
    Botun kaç dakika çalışacağı (0 = sınırsız)
#>

param(
    [Parameter(Mandatory=$true)]
    [string] $ResourceGroup = "TRADEBOT",
    
    [Parameter(Mandatory=$true)]
    [string] $VMName = "BearishAlphaBot-VM-01",
    
    [Parameter(Mandatory=$false)]
    [string] $ImageTag = "vm-vmboot-11",
    
    [Parameter(Mandatory=$false)]
    [int] $DurationMinutes = 20
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

# Script'in varlığını kontrol et
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

# Image tag'i güncelle (eğer varsayılandan farklıysa)
if [ "$ImageTag" != "vm-vmboot-11" ]; then
    echo "Updating image tag to: $ImageTag"
    # vm_run_session.py içinde IMAGE değişkenini değiştir
    sudo sed -i "s|vm-vmboot-[0-9]*|$ImageTag|g" /home/azureuser/vm_run_session.py
fi

# Bot'u başlat (mevcut script kullanılıyor)
echo ""
echo "Starting bot with vm_run_session.py..."
cd /home/azureuser
sudo python3 vm_run_session.py

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
        Write-Output "📊 Monitoring:"
        Write-Output "   - VM: $VMName"
        Write-Output "   - Resource Group: $ResourceGroup"
        Write-Output "   - Duration: $DurationMinutes minutes"
        Write-Output ""
        Write-Output "📝 To check bot status:"
        Write-Output "   SSH: ssh azureuser@[VM-IP] 'docker ps'"
        Write-Output "   Logs: ssh azureuser@[VM-IP] 'docker logs bearish-bot'"
        Write-Output ""
        Write-Output "⏱️ Bot will automatically stop after $DurationMinutes minutes"
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
