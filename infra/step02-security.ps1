# Step 02 - Security: Managed Identity and Key Vault scaffolding
# Usage: pwsh ./infra/step02-security.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$identityName = "bearish-reporter-id"
$keyVaultName = "bearish-kv"
$vmResourceGroup = "TradeBot"
$vmName = "BearishAlphaBot-VM-01"

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

Write-Host "Creating user-assigned managed identity $identityName..."
az identity create --name $identityName --resource-group $opsResourceGroup --location $location --output none

$identity = az identity show --name $identityName --resource-group $opsResourceGroup --query "{principalId:principalId, clientId:clientId, id:id}" --output json | ConvertFrom-Json

Write-Host "Assigning identity to VM $vmName..."
az vm identity assign --name $vmName --resource-group $vmResourceGroup --identities $identity.id --output none

Write-Host "Creating Key Vault $keyVaultName..."
az keyvault create `
    --name $keyVaultName `
    --resource-group $opsResourceGroup `
    --location $location `
    --enable-rbac-authorization false `
    --retention-days 90 `
    --sku standard `
    --output none

Write-Host "Granting Key Vault access to identity (secrets get/list)..."
az keyvault set-policy `
    --name $keyVaultName `
    --object-id $identity.principalId `
    --secret-permissions get list `
    --output none

Write-Host "Granting current Azure CLI identity permissions to manage secrets..."
$currentObjectId = az ad signed-in-user show --query id --output tsv
if (-not $currentObjectId) {
    throw "Unable to resolve the current Azure CLI identity Object ID. Ensure you are logged in with 'az login'."
}

az keyvault set-policy `
    --name $keyVaultName `
    --object-id $currentObjectId `
    --secret-permissions get list set delete purge `
    --output none

Write-Host "Step 02 complete. Populate secrets next (workspace key, SendGrid, etc.)."