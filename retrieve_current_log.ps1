# Script to download full log file from Azure VM
$remotePath = "/mnt/bearish/logs/live_trading_20251127_173903_737628.log"
$localPath = "C:\Users\sefaa\bearish-alpha-bot\live_trading_20251127_173903_737628_FULL.log"
$parts = @()
$chunkSize = 30
$totalLines = 2494 # We know this from wc -l

Write-Host "Target Remote File: $remotePath"
Write-Host "Target Local File: $localPath"
Write-Host "Total Lines: $totalLines"
Write-Host "Chunk Size: $chunkSize"

for ($i = 1; $i -le $totalLines; $i += $chunkSize) {
    $end = $i + $chunkSize - 1
    if ($end -gt $totalLines) { $end = $totalLines }
    
    Write-Host "Downloading lines $i to $end..."
    
    # Use sed to get specific line range
    $cmd = "sed -n '${i},${end}p' $remotePath"
    
    $chunk = az vm run-command invoke -g TradeBot -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts $cmd --query "value[0].message" -o tsv
    
    # Filter out Azure wrapper text if present (though tsv usually handles it well, sometimes headers appear)
    $cleanChunk = $chunk -split "`n" | Where-Object { $_ -notmatch "Enable succeeded|stdout|stderr" }
    
    $parts += $cleanChunk
    
    # Small delay to be nice to the API
    Start-Sleep -Milliseconds 500
}

# Write complete file
$parts | Out-File -FilePath $localPath -Encoding UTF8
Write-Host "Complete log saved to: $localPath"
Write-Host "Total lines downloaded: $($parts.Count)"

