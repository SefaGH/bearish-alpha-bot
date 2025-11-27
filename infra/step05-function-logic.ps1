# Step 05 - Azure Function App and Logic App scaffolding
# Usage: pwsh ./infra/step05-function-logic.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$functionStorage = "bearishfuncstore"
$functionApp = "bearish-reporting-func"
$functionPlan = "bearish-func-plan"
$logicAppName = "bearish-report-orchestrator"
$logicAppTemplate = Join-Path $PSScriptRoot ".." "docs" "reporting" "logic-app" "logicapp-template.json"
$functionEndpoint = "https://$functionApp.azurewebsites.net/api/run-report"

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

Write-Host "Creating storage account for Function App ($functionStorage)..."
$storageProbe = az storage account show `
    --name $functionStorage `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $storageProbe) {
    az storage account create `
        --name $functionStorage `
        --resource-group $opsResourceGroup `
        --location $location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --https-only true `
        --min-tls-version TLS1_2 `
        --allow-blob-public-access false `
        --output none
} else {
    Write-Host "Storage account already exists. Skipping creation."
}

Write-Host "Creating Function App plan ($functionPlan)..."
$planExists = az functionapp plan show `
    --name $functionPlan `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $planExists) {
    az functionapp plan create `
        --name $functionPlan `
        --resource-group $opsResourceGroup `
        --location $location `
        --sku EP1 `
        --is-linux `
        --min-instances 1 `
        --max-burst 3 `
        --output none
} else {
    Write-Host "Function plan already exists. Skipping creation."
}

Write-Host "Creating Function App ($functionApp)..."
$functionExists = az functionapp show `
    --name $functionApp `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $functionExists) {
    az functionapp create `
        --name $functionApp `
        --resource-group $opsResourceGroup `
        --plan $functionPlan `
        --runtime python `
        --runtime-version 3.11 `
        --functions-version 4 `
        --storage-account $functionStorage `
        --os-type Linux `
        --assign-identity `
        --output none
} else {
    Write-Host "Function App already exists. Ensuring managed identity is enabled..."
    az functionapp identity assign `
        --name $functionApp `
        --resource-group $opsResourceGroup `
        --output none
}

$functionIdentity = az functionapp identity show `
    --name $functionApp `
    --resource-group $opsResourceGroup `
    --query "{principalId:principalId, tenantId:tenantId}" `
    --output json | ConvertFrom-Json

Write-Host "Configuring Function App settings..."
$appSettingsMap = @{
    LOG_ANALYTICS_WORKSPACE_ID = "@Microsoft.KeyVault(SecretUri=https://bearish-kv.vault.azure.net/secrets/log-analytics-workspace-id/)"
    LOG_ANALYTICS_WORKSPACE_URL = "https://api.loganalytics.io/v1"
    ADX_CLUSTER_URI = "https://bearish-adx.westeurope.kusto.windows.net"
    ADX_DATABASE = "bearishdb"
    REPORTS_STORAGE_ACCOUNT = "bearishstorage"
    REPORTS_CONTAINER = "reports"
}
$appSettingsFile = [System.IO.Path]::GetTempFileName()
$appSettingsMap | ConvertTo-Json | Set-Content -Path $appSettingsFile -Encoding utf8

az functionapp config appsettings set `
    --name $functionApp `
    --resource-group $opsResourceGroup `
    --settings @$appSettingsFile `
    --output none

Remove-Item -Path $appSettingsFile -ErrorAction SilentlyContinue

Write-Host "Assigning Function App MSI roles..."
$workspaceResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.OperationalInsights/workspaces/bearish-logs"
$storageResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.Storage/storageAccounts/bearishstorage"

function Ensure-RoleAssignment {
    param (
        [string]$PrincipalId,
        [string]$RoleName,
        [string]$Scope
    )

    $assignments = az role assignment list `
        --assignee-object-id $PrincipalId `
        --scope $Scope `
        --output json | ConvertFrom-Json

    if ($assignments | Where-Object { $_.roleDefinitionName -eq $RoleName }) {
        Write-Host "Role $RoleName already assigned at scope $Scope."
    } else {
        az role assignment create `
            --assignee-object-id $PrincipalId `
            --role $RoleName `
            --scope $Scope `
            --output none
    }
}

Ensure-RoleAssignment -PrincipalId $functionIdentity.principalId -RoleName "Log Analytics Reader" -Scope $workspaceResourceId
Ensure-RoleAssignment -PrincipalId $functionIdentity.principalId -RoleName "Storage Blob Data Contributor" -Scope $storageResourceId

Write-Host "Granting Key Vault access to Function MSI..."
az keyvault set-policy `
    --name bearish-kv `
    --object-id $functionIdentity.principalId `
    --secret-permissions get list `
    --output none

Write-Host "Deploying Logic App definition ($logicAppName)..."
if (-not (Test-Path $logicAppTemplate)) {
    throw "Logic App template not found: $logicAppTemplate"
}
$logicDefinition = Get-Content -Raw -Path $logicAppTemplate | ConvertFrom-Json
$logicDefinition.parameters.functionEndpoint.defaultValue = $functionEndpoint
$logicPayload = @{ definition = $logicDefinition }
$tempLogicFile = [System.IO.Path]::GetTempFileName()
$logicPayload | ConvertTo-Json -Depth 20 | Set-Content -Path $tempLogicFile -Encoding utf8

$logicExists = az logic workflow show `
    --name $logicAppName `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $logicExists) {
    az logic workflow create `
        --name $logicAppName `
        --resource-group $opsResourceGroup `
        --definition @$tempLogicFile `
        --output none
} else {
    Write-Host "Logic App already exists. Updating definition..."
    az logic workflow update `
        --name $logicAppName `
        --resource-group $opsResourceGroup `
        --definition @$tempLogicFile `
        --output none
}

Remove-Item -Path $tempLogicFile -ErrorAction SilentlyContinue

Write-Host "Step 05 complete. Publish Function code and update Key Vault secrets before invoking."