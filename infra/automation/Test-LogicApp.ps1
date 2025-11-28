<#
.SYNOPSIS
    Test Logic App HTTP trigger with forceRestart parameter

.DESCRIPTION
    Tests the Logic App workflow with both forceRestart values

.PARAMETER CallbackUrl
    Logic App HTTP trigger callback URL

.PARAMETER DurationMinutes
    Trading session duration (1-85 minutes)

.PARAMETER ImageTag
    Docker image tag (default: vm-vmboot-11)

.PARAMETER ForceRestart
    Force restart even if bot is running (default: false)

.EXAMPLE
    .\Test-LogicApp.ps1 -CallbackUrl "https://..." -DurationMinutes 5
    
.EXAMPLE
    .\Test-LogicApp.ps1 -CallbackUrl "https://..." -DurationMinutes 5 -ForceRestart $true
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$CallbackUrl,
    
    [Parameter(Mandatory=$true)]
    [int]$DurationMinutes,
    
    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "vm-vmboot-11",
    
    [Parameter(Mandatory=$false)]
    [bool]$ForceRestart = $false
)

$ErrorActionPreference = "Stop"

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "║          🧪 Logic App HTTP Trigger Test                        ║" -ForegroundColor Cyan
Write-Host "║                                                                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Build request body
$body = @{
    durationMinutes = $DurationMinutes
    imageTag = $ImageTag
    forceRestart = $ForceRestart
    keyVaultName = "bearish-kv"
    kvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN"
} | ConvertTo-Json

Write-Host "📋 REQUEST DETAILS:" -ForegroundColor Yellow
Write-Host "   Duration: $DurationMinutes minutes" -ForegroundColor White
Write-Host "   Image Tag: $ImageTag" -ForegroundColor White
Write-Host "   Force Restart: $ForceRestart" -ForegroundColor White
Write-Host "`n📝 Request Body:" -ForegroundColor Yellow
Write-Host $body -ForegroundColor Gray

Write-Host "`n🚀 Sending HTTP request..." -ForegroundColor Yellow

try {
    $response = Invoke-RestMethod -Method POST -Uri $CallbackUrl `
        -ContentType "application/json" `
        -Body $body `
        -ErrorAction Stop
    
    Write-Host "   ✓ Request sent successfully" -ForegroundColor Green
    
    Write-Host "`n📊 RESPONSE:" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10 | Write-Host -ForegroundColor White
    
    Write-Host "`n✅ Test completed successfully!`n" -ForegroundColor Green
    
} catch {
    Write-Host "   ✗ Request failed" -ForegroundColor Red
    Write-Host "`n❌ ERROR DETAILS:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "`nResponse Body:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor Gray
    }
    
    exit 1
}

Write-Host "💡 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "   1. Check Azure Portal > Logic App > Runs history" -ForegroundColor White
Write-Host "   2. Check Azure Portal > Automation Account > Jobs" -ForegroundColor White
Write-Host "   3. Monitor runbook execution logs" -ForegroundColor White
Write-Host "`n"
