#!/usr/bin/env pwsh
# Act ile workflow test scripti
# Kullanim: .\run_act_test.ps1 [duration_in_seconds]
# Ornek: .\run_act_test.ps1 600

param(
    [int]$Duration = 600
)

Write-Host "🚀 Starting Act workflow test..." -ForegroundColor Cyan
Write-Host "   Mode: paper" -ForegroundColor Yellow
Write-Host "   Duration: $Duration seconds" -ForegroundColor Yellow
Write-Host "   ML: enabled" -ForegroundColor Yellow
Write-Host "   Debug: enabled" -ForegroundColor Yellow
Write-Host ""

act workflow_dispatch `
  -W .github/workflows/live_trading_launcher.yml `
  -j live-trading `
  --env-file .env `
  --var-file .vars `
  --container-architecture linux/amd64 `
  -P ubuntu-latest=catthehacker/ubuntu:act-latest `
  --input mode=paper `
  --input duration=$Duration `
  --input enable_ml=true `
  --input dry_run=false `
  --input skip_preflight=false `
  --input infinite=false `
  --input auto_restart=false `
  --input debug_mode=true
