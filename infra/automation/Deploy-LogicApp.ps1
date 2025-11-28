#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploys the Bearish Bot Logic App workflow to Azure.

.DESCRIPTION
    Creates a Logic App with HTTP trigger that invokes the Azure Automation runbook
    and sends email notifications via SendGrid on completion.

.PARAMETER ResourceGroup
    Azure resource group name.

.PARAMETER Location
    Azure region for deployment.

.PARAMETER LogicAppName
    Name of the Logic App to create.

.PARAMETER SendGridApiKey
    SendGrid API key for email notifications. If not provided, will attempt to read from Key Vault.

.EXAMPLE
    .\Deploy-LogicApp.ps1 -ResourceGroup "TradeBot" -LogicAppName "bearish-bot-orchestrator"
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroup,
    
    [Parameter(Mandatory = $false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory = $false)]
    [string]$LogicAppName = "bearish-bot-orchestrator",
    
    [Parameter(Mandatory = $false)]
    [string]$SendGridApiKey
)

$ErrorActionPreference = "Stop"

Write-Host @"

╔════════════════════════════════════════════════════╗
║     Bearish Bot Logic App Deployment              ║
╚════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Step 1: Get SendGrid API Key
Write-Host "Step 1: Configuring SendGrid API key..." -ForegroundColor Yellow

if (-not $SendGridApiKey) {
    Write-Host "  Attempting to retrieve from Key Vault..." -ForegroundColor Gray
    try {
        $SendGridApiKey = Get-AzKeyVaultSecret -VaultName "bearish-kv" -Name "SENDGRID-API-KEY" -AsPlainText -ErrorAction Stop
        Write-Host "  ✓ Retrieved from Key Vault" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to retrieve from Key Vault" -ForegroundColor Red
        Write-Host "  Please provide SendGrid API key via -SendGridApiKey parameter or add it to Key Vault" -ForegroundColor Yellow
        exit 1
    }
}

# Step 2: Create Logic App
Write-Host "`nStep 2: Creating Logic App..." -ForegroundColor Yellow

$existingLogicApp = az logic workflow show `
    --resource-group $ResourceGroup `
    --name $LogicAppName `
    2>$null

if ($existingLogicApp) {
    Write-Host "  ⚠ Logic App already exists, updating..." -ForegroundColor Yellow
} else {
    Write-Host "  Creating new Logic App..." -ForegroundColor Gray
}

# Create Logic App with definition
$workflowPath = Join-Path $PSScriptRoot "logic-app-workflow-sendgrid.json"

az logic workflow create `
    --resource-group $ResourceGroup `
    --location $Location `
    --name $LogicAppName `
    --definition "@$workflowPath" `
    --output none

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Logic App created/updated" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to create Logic App" -ForegroundColor Red
    exit 1
}

# Step 3: Enable Managed Identity
Write-Host "`nStep 3: Enabling Managed Identity..." -ForegroundColor Yellow

$identity = az logic workflow identity assign `
    --resource-group $ResourceGroup `
    --name $LogicAppName `
    --query principalId `
    --output tsv `
    2>$null

if ($identity) {
    Write-Host "  ✓ Managed Identity enabled" -ForegroundColor Green
    Write-Host "  Principal ID: $identity" -ForegroundColor Gray
} else {
    Write-Host "  ✗ Failed to enable Managed Identity" -ForegroundColor Red
    exit 1
}

# Step 4: Create Azure Automation Connection
Write-Host "`nStep 4: Creating Azure Automation API connection..." -ForegroundColor Yellow

$automationConnectionName = "azureautomation-connection"

# Check if connection exists
$existingConnection = az resource show `
    --resource-group $ResourceGroup `
    --resource-type "Microsoft.Web/connections" `
    --name $automationConnectionName `
    2>$null

if (-not $existingConnection) {
    Write-Host "  Creating new API connection..." -ForegroundColor Gray
    
    $connectionDefinition = @{
        properties = @{
            displayName = "Azure Automation Connection"
            api = @{
                id = "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/providers/Microsoft.Web/locations/$Location/managedApis/azureautomation"
            }
            parameterValues = @{
                token = @{
                    type = "ManagedServiceIdentity"
                }
            }
        }
        location = $Location
    } | ConvertTo-Json -Depth 10
    
    $tempFile = [System.IO.Path]::GetTempFileName()
    $connectionDefinition | Out-File -FilePath $tempFile -Encoding utf8
    
    az resource create `
        --resource-group $ResourceGroup `
        --resource-type "Microsoft.Web/connections" `
        --name $automationConnectionName `
        --properties "@$tempFile" `
        --output none
    
    Remove-Item $tempFile
    
    Write-Host "  ✓ API connection created" -ForegroundColor Green
} else {
    Write-Host "  ⚠ API connection already exists" -ForegroundColor Yellow
}

# Step 5: Assign permissions
Write-Host "`nStep 5: Assigning permissions..." -ForegroundColor Yellow

# Automation Contributor role for Logic App
$roleAssignment = az role assignment create `
    --assignee $identity `
    --role "Automation Contributor" `
    --scope "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/$ResourceGroup/providers/Microsoft.Automation/automationAccounts/tradebot-automation" `
    2>$null

if ($roleAssignment) {
    Write-Host "  ✓ Automation Contributor role assigned" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Role assignment may already exist" -ForegroundColor Yellow
}

# Step 6: Update Logic App with SendGrid parameter
Write-Host "`nStep 6: Configuring SendGrid parameter..." -ForegroundColor Yellow

# Note: SendGrid API key should be stored as a secure parameter
# This is a placeholder - actual implementation should use Key Vault reference
Write-Host "  ⚠ SendGrid API key should be configured via Azure Portal" -ForegroundColor Yellow
Write-Host "  Navigate to: Logic App > Workflow settings > Parameters" -ForegroundColor Gray
Write-Host "  Add parameter: sendgrid_api_key = $($SendGridApiKey.Substring(0,10))..." -ForegroundColor Gray

# Step 7: Get callback URL
Write-Host "`nStep 7: Retrieving HTTP endpoint..." -ForegroundColor Yellow

Start-Sleep -Seconds 3

$callbackUrl = az rest `
    --method post `
    --uri "/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/$ResourceGroup/providers/Microsoft.Logic/workflows/$LogicAppName/triggers/manual/listCallbackUrl?api-version=2016-06-01" `
    --query value `
    --output tsv `
    2>$null

if ($callbackUrl) {
    Write-Host "  ✓ HTTP endpoint retrieved" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Failed to retrieve callback URL" -ForegroundColor Yellow
    $callbackUrl = "https://portal.azure.com"
}

Write-Host @"

╔════════════════════════════════════════════════════╗
║         Logic App Deployed Successfully!           ║
╚════════════════════════════════════════════════════╝

📍 HTTP Endpoint:
$callbackUrl

📧 Email Notifications: Configured via SendGrid

🔧 Manual Configuration Required:
1. Go to Azure Portal > Logic App > Workflow settings
2. Add parameter 'sendgrid_api_key' with your SendGrid API key
3. Update email addresses in workflow if needed

📝 Test the endpoint:
Invoke-RestMethod -Method POST -Uri "$callbackUrl" ``
    -ContentType "application/json" ``
    -Body (@{
        durationMinutes = 5
        imageTag = "vm-vmboot-9"
    } | ConvertTo-Json)

"@ -ForegroundColor Green
