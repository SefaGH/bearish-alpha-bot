param(
    [Parameter(Mandatory=$true)] [string] $resourceGroup,
    [Parameter(Mandatory=$true)] [string] $vmName
)

$ErrorActionPreference = "Stop"

Write-Output "--- BAĞLANTI TESTİ BAŞLIYOR ---"
Write-Output "Resource Group: $resourceGroup"
Write-Output "VM Name: $vmName"

try {
    # 1. Kimlik Doğrulama
    Write-Output "`n1. Azure'a bağlanılıyor..."
    Connect-AzAccount -Identity | Out-Null
    Write-Output "   ✅ Bağlantı OK."

    # 2. VM Durumunu Kontrol Et (İyileştirilmiş)
    Write-Output "`n2. VM Durumu sorgulanıyor..."
    $vmStatus = Get-AzVM -ResourceGroupName $resourceGroup -Name $vmName -Status
    
    # PowerState'i güvenli şekilde bul (index'e güvenme)
    $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like "PowerState/*" }).DisplayStatus
    Write-Output "   -> VM PowerState: $powerState"

    if ($powerState -ne "VM running") {
        Write-Error "!!! HATA: VM çalışmıyor. Şu anki durum: $powerState"
        Write-Output "Lütfen Azure Portal'dan VM'i başlatın ve tekrar deneyin."
        return
    }

    # 3. Basit Komut Gönderimi
    Write-Output "`n3. VM'e test komutu gönderiliyor..."
    
    # Inline script (dosya oluşturmaya gerek yok)
    $scriptContent = @'
echo "=== VM BAĞLANTI TESTİ ==="
echo "Hostname: $(hostname)"
echo "User: $(id -un)"
echo "Uptime: $(uptime -p)"
echo "Docker Status: $(systemctl is-active docker 2>/dev/null || echo 'N/A')"
echo "=== TEST BAŞARILI ==="
'@
    
    Write-Output "   -> Komut gönderiliyor (RunShellScript)..."
    
    $invokeParams = @{
        ResourceGroupName = $resourceGroup
        VMName = $vmName
        CommandId = 'RunShellScript'
        ScriptString = $scriptContent
    }
    
    $result = Invoke-AzVMRunCommand @invokeParams

    # 4. Sonuçları Ekrana Bas (İyileştirilmiş)
    Write-Output "`n--- VM CEVABI ---"
    
    if ($result.Value -and $result.Value[0].Message) {
        Write-Output $result.Value[0].Message
    } elseif ($result.Value[1] -and $result.Value[1].Message) {
        # Bazen stdout Value[1]'de olur
        Write-Output $result.Value[1].Message
    } else {
        Write-Warning "VM komutu kabul etti ama boş cevap döndü."
        Write-Output "`nHam Sonuç:"
        Write-Output ($result | ConvertTo-Json -Depth 3)
    }

    Write-Output "`n--- ✅ TEST BAŞARIYLA TAMAMLANDI ---"

} catch {
    Write-Error "`n!!! BEKLENMEYEN HATA OLUŞTU !!!"
    Write-Error "Hata: $($_.Exception.Message)"
    Write-Error "Satır: $($_.InvocationInfo.ScriptLineNumber)"
    throw
}
