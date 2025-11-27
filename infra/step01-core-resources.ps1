# Step 01 - Core Azure resource scaffolding for reporting pipeline
# Usage: pwsh ./infra/step01-core-resources.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$workspaceName = "bearish-logs"
$storageAccount = "bearishstorage"

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

Write-Host "Registering required resource providers (OperationalInsights, Storage)..."
az provider register --namespace Microsoft.OperationalInsights --output none
az provider register --namespace Microsoft.Storage --output none

Write-Host "Creating resource group $opsResourceGroup in $location (idempotent)..."
az group create --name $opsResourceGroup --location $location --output none

Write-Host "Creating Log Analytics workspace $workspaceName..."
az monitor log-analytics workspace create `
    --resource-group $opsResourceGroup `
    --workspace-name $workspaceName `
    --location $location `
    --sku PerGB2018 `
    --retention-time 30 `
    --output none

Write-Host "(Optional) Enable Container Insights via VM Insights after provider registration completes."

Write-Host "Creating Storage account $storageAccount..."
az storage account create `
    --name $storageAccount `
    --resource-group $opsResourceGroup `
    --location $location `
    --sku Standard_LRS `
    --kind StorageV2 `
    --https-only true `
    --min-tls-version TLS1_2 `
    --allow-blob-public-access false `
    --output none

Write-Host "Creating blob containers (reports, raw-logs, parsed-events)..."
$containers = @("reports", "raw-logs", "parsed-events")
foreach ($container in $containers) {
    az storage container create `
        --name $container `
        --account-name $storageAccount `
        --auth-mode login `
        --public-access off `
        --resource-group $opsResourceGroup `
        --output none
}

Write-Host "Step 01 complete. Review before executing."