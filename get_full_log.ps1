# Script to download full log file from Azure VM
$parts = @()

# Get first 1000 lines
Write-Host "Downloading part 1..."
$part1 = az vm run-command invoke -g TRADEBOT -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "head -n 1000 /tmp/live_trading_20251126_132213_091659.log" --query "value[0].message" -o tsv
$parts += ($part1 -split "`n" | Where-Object { $_ -notmatch "Enable succeeded|stdout|stderr" -and $_.Trim() -ne "" })

# Get lines 1001-2000  
Write-Host "Downloading part 2..."
$part2 = az vm run-command invoke -g TRADEBOT -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "sed -n '1001,2000p' /tmp/live_trading_20251126_132213_091659.log" --query "value[0].message" -o tsv
$parts += ($part2 -split "`n" | Where-Object { $_ -notmatch "Enable succeeded|stdout|stderr" -and $_.Trim() -ne "" })

# Get remaining lines (2001+)
Write-Host "Downloading part 3..."
$part3 = az vm run-command invoke -g TRADEBOT -n BearishAlphaBot-VM-01 --command-id RunShellScript --scripts "tail -n +2001 /tmp/live_trading_20251126_132213_091659.log" --query "value[0].message" -o tsv
$parts += ($part3 -split "`n" | Where-Object { $_ -notmatch "Enable succeeded|stdout|stderr" -and $_.Trim() -ne "" })

# Write complete file
$outputPath = "C:\Users\sefaa\bearish-alpha-bot\live_trading_20251126_132213_091659_FULL.log"
$parts | Out-File -FilePath $outputPath -Encoding UTF8
Write-Host "Complete log saved to: $outputPath"
Write-Host "Total lines: $($parts.Count)"