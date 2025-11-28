<#
.SYNOPSIS
    Simplified deployment script for Bearish Alpha Bot runbook
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$true)]
    [string]$AutomationAccountName = "tradebot-automation",
    
    [Parameter(Mandatory=$false)]
    [string]$RunbookName = "Start-BearishBot-Enhanced",
    
    [Parameter(Mandatory=$false)]
    [string]$RunbookPath = "$PSScriptRoot\Start-BearishBot-Enhanced.ps1"
)

$ErrorActionPreference = "Stop"

Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    Bearish Bot Runbook - Simplified Deployment    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Step 1: Delete existing runbook if present
Write-Host "Step 1: Checking for existing runbook..." -ForegroundColor Yellow

$existing = az automation runbook show `
    --name $RunbookName `
    --automation-account-name $AutomationAccountName `
    --resource-group $ResourceGroup `
    --query name `
    -o tsv 2>$null

if ($existing) {
    Write-Host "  Deleting existing runbook..." -ForegroundColor Gray
    az automation runbook delete `
        --name $RunbookName `
        --automation-account-name $AutomationAccountName `
        --resource-group $ResourceGroup `
        --yes `
        --output none
    Write-Host "✓ Existing runbook deleted" -ForegroundColor Green
}

# Step 2: Create runbook
Write-Host "`nStep 2: Creating runbook..." -ForegroundColor Yellow

az automation runbook create `
    --resource-group $ResourceGroup `
    --automation-account-name $AutomationAccountName `
    --name $RunbookName `
    --type PowerShell `
    --location eastus `
    --output none

Write-Host "✓ Runbook created" -ForegroundColor Green

# Step 3: Upload content
Write-Host "`nStep 3: Uploading runbook content..." -ForegroundColor Yellow

$scriptContent = Get-Content -Path $RunbookPath -Raw
$tempFile = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tempFile -Value $scriptContent -Encoding UTF8

az automation runbook replace-content `
    --resource-group $ResourceGroup `
    --automation-account-name $AutomationAccountName `
    --name $RunbookName `
    --content "@$tempFile" `
    --output none

Remove-Item -Path $tempFile -Force

Write-Host "✓ Content uploaded" -ForegroundColor Green

# Step 4: Publish
Write-Host "`nStep 4: Publishing runbook..." -ForegroundColor Yellow

az automation runbook publish `
    --resource-group $ResourceGroup `
    --automation-account-name $AutomationAccountName `
    --name $RunbookName `
    --output none

Write-Host "✓ Runbook published" -ForegroundColor Green

# Success
Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           Runbook Deployed Successfully!           ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "⚠ IMPORTANT: Manual Steps Required ⚠`n" -ForegroundColor Yellow
Write-Host "1. Enable Managed Identity:" -ForegroundColor White
Write-Host "   • Go to Azure Portal → Automation Account → Identity" -ForegroundColor Gray
Write-Host "   • Enable 'System assigned' identity" -ForegroundColor Gray
Write-Host "   • Copy the Principal (Object) ID`n" -ForegroundColor Gray

Write-Host "2. Assign VM Contributor Role:" -ForegroundColor White
Write-Host "   az role assignment create \" -ForegroundColor Gray
Write-Host "     --assignee <PRINCIPAL_ID> \" -ForegroundColor Gray
Write-Host "     --role 'Virtual Machine Contributor' \" -ForegroundColor Gray
Write-Host "     --scope /subscriptions/<SUB_ID>/resourceGroups/TradeBot/providers/Microsoft.Compute/virtualMachines/BearishAlphaBot-VM-01`n" -ForegroundColor Gray

Write-Host "3. Configure Key Vault Access:" -ForegroundColor White
Write-Host "   az keyvault set-policy \" -ForegroundColor Gray
Write-Host "     --name bearish-kv \" -ForegroundColor Gray
Write-Host "     --object-id <PRINCIPAL_ID> \" -ForegroundColor Gray
Write-Host "     --secret-permissions get list`n" -ForegroundColor Gray

Write-Host "4. Test the runbook:" -ForegroundColor White
Write-Host "   az automation runbook start \" -ForegroundColor Gray
Write-Host "     --name $RunbookName \" -ForegroundColor Gray
Write-Host "     --automation-account-name $AutomationAccountName \" -ForegroundColor Gray
Write-Host "     --resource-group $ResourceGroup \" -ForegroundColor Gray
Write-Host "     --parameters durationMinutes=5`n" -ForegroundColor Gray
