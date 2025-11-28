<#
.SYNOPSIS
    Deploys the Bearish Alpha Bot runbook to Azure Automation Account
    
.DESCRIPTION
    This script:
    - Creates or updates the Azure Automation runbook
    - Configures Managed Identity and permissions
    - Sets up Key Vault access
    - Validates deployment
    
.PARAMETER ResourceGroup
    Resource group containing the Automation Account
    
.PARAMETER AutomationAccountName
    Name of the Automation Account
    
.PARAMETER RunbookName
    Name for the runbook (default: Start-BearishBot-Enhanced)
    
.PARAMETER RunbookPath
    Path to the runbook script file
    
.PARAMETER Location
    Azure region (default: eastus)
    
.PARAMETER SubscriptionId
    Azure subscription ID (optional, uses current context if not specified)
    
.EXAMPLE
    .\Deploy-AutomationRunbook.ps1 -ResourceGroup "TradeBot" -AutomationAccountName "tradebot-automation"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup,
    
    [Parameter(Mandatory=$true)]
    [string]$AutomationAccountName,
    
    [Parameter(Mandatory=$false)]
    [string]$RunbookName = "Start-BearishBot-Enhanced",
    
    [Parameter(Mandatory=$false)]
    [string]$RunbookPath = "$PSScriptRoot\Start-BearishBot-Enhanced.ps1",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId = "",
    
    [Parameter(Mandatory=$false)]
    [string]$KeyVaultName = "bearish-kv",
    
    [Parameter(Mandatory=$false)]
    [string]$VMResourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$false)]
    [string]$VMName = "BearishAlphaBot-VM-01"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Info {
    param([string]$Message)
    Write-Host "  $Message" -ForegroundColor Gray
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-ErrorMessage {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

try {
    Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  Bearish Alpha Bot Automation Runbook Deployment  ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan
    
    # Validate runbook file exists
    if (-not (Test-Path $RunbookPath)) {
        throw "Runbook file not found: $RunbookPath"
    }
    Write-Success "Runbook file found: $RunbookPath"
    
    # Set subscription context
    Write-Step "Setting Azure Subscription Context"
    
    if ($SubscriptionId) {
        az account set --subscription $SubscriptionId
        Write-Success "Subscription set to: $SubscriptionId"
    } else {
        $currentSub = az account show --query id -o tsv
        Write-Info "Using current subscription: $currentSub"
    }
    
    # Check if Automation Account exists
    Write-Step "Checking Automation Account"
    
    $automationAccountExists = az automation account show `
        --name $AutomationAccountName `
        --resource-group $ResourceGroup `
        --query name `
        --output tsv 2>$null
    
    if (-not $automationAccountExists) {
        Write-Info "Automation Account not found. Creating..."
        
        az automation account create `
            --name $AutomationAccountName `
            --resource-group $ResourceGroup `
            --location $Location `
            --sku Basic `
            --output none
        
        Write-Success "Automation Account created: $AutomationAccountName"
    } else {
        Write-Success "Automation Account exists: $AutomationAccountName"
    }
    
    # Enable Managed Identity
    Write-Step "Configuring Managed Identity"
    
    Write-Info "Enabling system-assigned managed identity..."
    
    # Use az resource to enable identity
    $subscriptionId = az account show --query id -o tsv
    $resourceId = "/subscriptions/$subscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.Automation/automationAccounts/$AutomationAccountName"
    
    try {
        # Enable identity using az resource
        az resource update `
            --ids $resourceId `
            --set identity.type=SystemAssigned `
            --output none 2>&1 | Out-Null
        
        Write-Success "Managed Identity enabled"
        
        # Wait for identity to propagate
        Start-Sleep -Seconds 10
        
        # Get principal ID
        $accountInfo = az automation account show `
            --name $AutomationAccountName `
            --resource-group $ResourceGroup `
            --output json 2>&1 | ConvertFrom-Json
        
        $principalId = $accountInfo.identity.principalId
        
        if (-not $principalId) {
            Write-Warning "Could not automatically retrieve Principal ID"
            Write-Warning "Please enable Managed Identity manually in Azure Portal:"
            Write-Warning "1. Go to Automation Account → Identity"
            Write-Warning "2. Enable System assigned identity"
            Write-Warning "3. Copy the Principal ID and continue with manual permission setup"
            throw "Manual identity configuration required"
        }
        
    } catch {
        Write-Warning "Automatic identity configuration failed: $($_.Exception.Message)"
        Write-Warning "You can enable it manually in Azure Portal and re-run this script"
        throw
    }
    
    Write-Success "Managed Identity enabled"
    Write-Info "Principal ID: $principalId"
    
    # Delete existing runbook if present
    Write-Step "Managing Runbook"
    
    $runbookExists = az automation runbook show `
        --name $RunbookName `
        --automation-account-name $AutomationAccountName `
        --resource-group $ResourceGroup `
        --query name `
        --output tsv 2>$null
    
    if ($runbookExists) {
        Write-Info "Existing runbook found. Deleting..."
        
        az automation runbook delete `
            --name $RunbookName `
            --automation-account-name $AutomationAccountName `
            --resource-group $ResourceGroup `
            --yes `
            --output none
        
        Write-Success "Existing runbook deleted"
    }
    
    # Create new runbook
    Write-Info "Creating runbook: $RunbookName"
    
    az automation runbook create `
        --resource-group $ResourceGroup `
        --automation-account-name $AutomationAccountName `
        --name $RunbookName `
        --type PowerShell `
        --location $Location `
        --output none
    
    Write-Success "Runbook created"
    
    # Upload runbook content
    Write-Info "Uploading runbook content..."
    
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
    
    Write-Success "Runbook content uploaded"
    
    # Publish runbook
    Write-Info "Publishing runbook..."
    
    az automation runbook publish `
        --resource-group $ResourceGroup `
        --automation-account-name $AutomationAccountName `
        --name $RunbookName `
        --output none
    
    Write-Success "Runbook published"
    
    # Configure permissions
    Write-Step "Configuring Permissions"
    
    # Get subscription ID
    $subscriptionId = az account show --query id -o tsv
    
    # VM Contributor role
    Write-Info "Assigning Virtual Machine Contributor role..."
    
    $vmResourceId = "/subscriptions/$subscriptionId/resourceGroups/$VMResourceGroup/providers/Microsoft.Compute/virtualMachines/$VMName"
    
    az role assignment create `
        --assignee-object-id $principalId `
        --assignee-principal-type ServicePrincipal `
        --role "Virtual Machine Contributor" `
        --scope $vmResourceId `
        --output none 2>$null
    
    Write-Success "VM Contributor role assigned"
    
    # Key Vault access
    Write-Info "Configuring Key Vault access..."
    
    $kvExists = az keyvault show `
        --name $KeyVaultName `
        --query name `
        --output tsv 2>$null
    
    if ($kvExists) {
        az keyvault set-policy `
            --name $KeyVaultName `
            --object-id $principalId `
            --secret-permissions get list `
            --output none
        
        Write-Success "Key Vault access configured"
    } else {
        Write-Warning "Key Vault '$KeyVaultName' not found. Skipping Key Vault configuration."
    }
    
    # Test runbook
    Write-Step "Validating Deployment"
    
    $runbookInfo = az automation runbook show `
        --name $RunbookName `
        --automation-account-name $AutomationAccountName `
        --resource-group $ResourceGroup `
        --output json | ConvertFrom-Json
    
    Write-Info "Runbook Name: $($runbookInfo.name)"
    Write-Info "Runbook Type: $($runbookInfo.runbookType)"
    Write-Info "State: $($runbookInfo.state)"
    Write-Info "Location: $($runbookInfo.location)"
    
    # Summary
    Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║           Deployment Completed Successfully       ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════╝`n" -ForegroundColor Green
    
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "1. Test the runbook with:" -ForegroundColor White
    Write-Host "   az automation runbook start \" -ForegroundColor Gray
    Write-Host "     --name $RunbookName \" -ForegroundColor Gray
    Write-Host "     --automation-account-name $AutomationAccountName \" -ForegroundColor Gray
    Write-Host "     --resource-group $ResourceGroup \" -ForegroundColor Gray
    Write-Host "     --parameters durationMinutes=10" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Create Logic App for HTTP-triggered execution" -ForegroundColor White
    Write-Host "3. Configure monitoring and alerts" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "`n╔════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "║              Deployment Failed                     ║" -ForegroundColor Red
    Write-Host "╚════════════════════════════════════════════════════╝`n" -ForegroundColor Red
    
    Write-ErrorMessage $_.Exception.Message
    
    if ($_.ScriptStackTrace) {
        Write-Host "`nStack Trace:" -ForegroundColor Yellow
        Write-Host $_.ScriptStackTrace -ForegroundColor Gray
    }
    
    exit 1
}
