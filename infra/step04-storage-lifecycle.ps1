# Step 04 - Storage lifecycle and monitoring alerts scaffolding
# Usage: pwsh ./infra/step04-storage-lifecycle.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$storageAccount = "bearishstorage"
$workspaceName = "bearish-logs"
$lifecyclePolicyFile = Join-Path $PSScriptRoot "storage-lifecycle.json"
$actionGroupName = "bearish-alerts"
$actionGroupEmail = "alerts@example.com"  # TODO: replace with real email or remove
$vmResourceId = "/subscriptions/$subscriptionId/resourceGroups/TradeBot/providers/Microsoft.Compute/virtualMachines/BearishAlphaBot-VM-01"
$workspaceResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.OperationalInsights/workspaces/$workspaceName"

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

Write-Host "Applying storage lifecycle policy..."
if (-not (Test-Path $lifecyclePolicyFile)) {
    throw "Lifecycle policy file missing: $lifecyclePolicyFile"
}
az storage account management-policy create `
    --account-name $storageAccount `
    --resource-group $opsResourceGroup `
    --policy @$lifecyclePolicyFile `
    --output none

Write-Host "Creating action group $actionGroupName (edit recipients before running)..."
az monitor action-group create `
    --name $actionGroupName `
    --resource-group $opsResourceGroup `
    --short-name Bearish `
    --action email Admin $actionGroupEmail `
    --output none
$actionGroupResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.Insights/actionGroups/$actionGroupName"

Write-Host "Creating container stop alert (Container Insights)..."
az monitor metrics alert create `
    --name "bearish-container-stop" `
    --resource-group $opsResourceGroup `
    --scopes $vmResourceId `
    --condition "avg Percentage CPU > 95" `
    --description "CPU usage spiked on bearish-bot VM" `
    --window-size 5m `
    --evaluation-frequency 5m `
    --severity 2 `
    --action $actionGroupResourceId `
    --output none

Write-Host "Creating shutdown event alert (Log Analytics scheduled query)..."
$shutdownCondition = "count 'ShutdownEvents' > 0 resource id _ResourceId at least 1 violations out of 1 aggregated points"
$shutdownQuery = "ShutdownEvents=BearishEvents_CL | where event_type_s == 'shutdown'"

az monitor scheduled-query create `
    --name "bearish-shutdown-alert" `
    --resource-group $opsResourceGroup `
    --scopes $workspaceResourceId `
    --condition $shutdownCondition `
    --condition-query $shutdownQuery `
    --description "Bot shutdown detected" `
    --window-size 5m `
    --evaluation-frequency 5m `
    --severity 2 `
    --action-groups $actionGroupResourceId `
    --skip-query-validation true `
    --output none

Write-Host "Step 04 complete. Adjust action group recipients before execution."