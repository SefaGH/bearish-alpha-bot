param(
    [Parameter(Mandatory=$true)] [string] $resourceGroup,
    [Parameter(Mandatory=$true)] [string] $vmName
)

$ErrorActionPreference = "Stop"

Write-Output "--- BAĞLANTI TESTİ BAŞLIYOR ---"

try {
    # 1. Kimlik Doğrulama
    Write-Output "1. Azure'a bağlanılıyor..."
    Connect-AzAccount -Identity | Out-Null
    Write-Output "   -> Bağlantı OK."

    # 2. VM Durumunu Kontrol Et (KRİTİK ADIM)
    Write-Output "2. VM Durumu sorgulanıyor..."
    $vmStatus = Get-AzVM -ResourceGroupName $resourceGroup -Name $vmName -Status
    
    # VM'in o anki durumunu (Running/Deallocated) bul
    $displayStatus = $vmStatus.Statuses[1].DisplayStatus
    Write-Output "   -> VM Şu An: $displayStatus"

    if ($displayStatus -ne "VM running") {
        Write-Error "!!! HATA: VM çalışmıyor (Running değil). Komut gönderilemez."
        Write-Output "Lütfen Azure Portal'dan VM'i başlatın ve tekrar deneyin."
        return
    }

    # 3. Basit Komut Gönderimi (Docker yok, sadece Linux çekirdek testi)
    Write-Output "3. VM'e 'id' ve 'hostname' komutu gönderiliyor..."
    
    # Çok basit bir script, hata ihtimali %0
    $scriptCmd = "echo 'MERHABA AZURE' && hostname && id"
    
    $tempPath = ".\test_connection.sh"
    $scriptCmd | Out-File -FilePath $tempPath -Encoding utf8

    $result = Invoke-AzVMRunCommand -ResourceGroupName $resourceGroup -VMName $vmName -CommandId 'RunShellScript' -ScriptPath $tempPath

    # 4. Sonuçları Ekrana Bas
    Write-Output "--- VM CEVABI AŞAĞIDADIR ---"
    
    if ($result.Value[0].Message) {
        Write-Output $result.Value[0].Message
    } else {
        Write-Warning "VM komutu kabul etti ama boş cevap döndü."
        Write-Output "Ham Sonuç Obj: "
        Write-Output $result
    }

    Write-Output "--- TEST BİTTİ ---"

} catch {
    Write-Error "BEKLENMEYEN HATA OLUŞTU!"
    Write-Error $_.Exception.Message
}