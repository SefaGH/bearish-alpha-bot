#!/usr/bin/env pwsh
# Build custom Docker image for Act with pre-installed dependencies

Write-Host "🔨 Building custom Docker image for Act..." -ForegroundColor Cyan
Write-Host "   This will take 5-10 minutes but only needs to be done once" -ForegroundColor Yellow
Write-Host ""

docker build -f Dockerfile.act -t bearish-alpha-bot:act-cached .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
    Write-Host "   Image name: bearish-alpha-bot:act-cached" -ForegroundColor Green
    Write-Host ""
    Write-Host "Now update run_act_test.ps1 to use this image." -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}
