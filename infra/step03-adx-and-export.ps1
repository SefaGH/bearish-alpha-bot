# Step 03 - Azure Data Explorer and Log Analytics export scaffolding
# Usage: pwsh ./infra/step03-adx-and-export.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$kustoCluster = "bearish-adx"
$kustoDatabase = "bearishdb"
$workspaceName = "bearish-logs"
$storageAccount = "bearishstorage"

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

Write-Host "Ensuring kusto CLI extension is installed..."
az extension add --name kusto --upgrade --only-show-errors | Out-Null

Write-Host "Registering required resource providers (Kusto, Insights)..."
az provider register --namespace Microsoft.Kusto --only-show-errors | Out-Null
az provider register --namespace Microsoft.Insights --only-show-errors | Out-Null

Write-Host "Ensuring data explorer cluster $kustoCluster exists..."
$clusterProbe = az kusto cluster show `
    --name $kustoCluster `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $clusterProbe) {
    Write-Host "Creating Azure Data Explorer cluster $kustoCluster..."
    az kusto cluster create `
        --name $kustoCluster `
        --resource-group $opsResourceGroup `
        --location $location `
        --sku name=Standard_D11_v2 tier=Standard capacity=2 `
        --enable-double-encryption false `
        --output none

    Write-Host "Waiting for cluster provisioning (this can take several minutes)..."
    az kusto cluster wait `
        --name $kustoCluster `
        --resource-group $opsResourceGroup `
        --created `
        --interval 60 `
        --timeout 3600 | Out-Null
} else {
    Write-Host "Cluster already exists. Skipping creation."
}

Write-Host "Ensuring ADX database $kustoDatabase exists..."
$databaseProbe = az kusto database show `
    --cluster-name $kustoCluster `
    --resource-group $opsResourceGroup `
    --name $kustoDatabase `
    --query name `
    --output tsv 2>$null

if (-not $databaseProbe) {
    Write-Host "Creating ADX database $kustoDatabase..."
    az kusto database create `
        --cluster-name $kustoCluster `
        --resource-group $opsResourceGroup `
        --database-name $kustoDatabase `
        --read-write-database location=$location soft-delete-period=P90D hot-cache-period=P7D `
        --output none
} else {
    Write-Host "Database already exists. Skipping creation."
}

Write-Host "Deploying ADX table schema (bearish_events)..."
$schemaFile = Join-Path $PSScriptRoot "schema-bearish-events.kql"
if (-not (Test-Path $schemaFile)) {
    throw "Schema file not found: $schemaFile"
}
$schemaRaw = Get-Content -Raw -Path $schemaFile
$schemaBlocks = $schemaRaw -split "(\r?\n){2,}"
$schemaNormalized = @()
foreach ($block in $schemaBlocks) {
    $trimmed = $block.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        continue
    }
    $singleLine = ($trimmed -replace "\r?\n", " ") -replace "\s{2,}", " "
    $schemaNormalized += $singleLine.Trim()
}
$schemaContent = ($schemaNormalized -join "`n").Trim()
$schemaHash = (Get-FileHash -InputStream ([System.IO.MemoryStream][System.Text.Encoding]::UTF8.GetBytes($schemaContent)) -Algorithm SHA256).Hash
az kusto script create `
    --name create-bearish-events `
    --cluster-name $kustoCluster `
    --database-name $kustoDatabase `
    --resource-group $opsResourceGroup `
    --script-content $schemaContent `
    --continue-on-errors true `
    --force-update-tag $schemaHash `
    --output none

Write-Host "Configuring Log Analytics diagnostic settings..."
$workspaceResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.OperationalInsights/workspaces/$workspaceName"
$storageAccountId = az storage account show `
    --name $storageAccount `
    --resource-group $opsResourceGroup `
    --query id `
    --output tsv

if (-not $storageAccountId) {
    throw "Unable to resolve storage account ID for $storageAccount."
}

$availableLogCategoriesRaw = az monitor diagnostic-settings categories list `
    --resource $workspaceResourceId `
    --query "value[?categoryType=='Logs'].name" `
    --output tsv
$availableLogCategories = $availableLogCategoriesRaw -split "(`r`n|`n|`r)" | Where-Object { $_ }

if (-not $availableLogCategories -or $availableLogCategories.Count -eq 0) {
    throw "Failed to retrieve diagnostic categories for workspace $workspaceName."
}


$selectedLogs = @()
foreach ($category in $availableLogCategories) {
    $selectedLogs += @{ category = $category; enabled = $true }
}

if ($selectedLogs.Count -eq 0) {
    $availableJoined = ($availableLogCategories -join ", ")
    throw "None of the desired diagnostic log categories are available. Available: $availableJoined"
}

$logsJson = ($selectedLogs | ConvertTo-Json -Depth 3 -Compress)
$diagName = "bearish-logs-to-storage"

Write-Host "Configuring diagnostic setting $diagName to stream workspace logs to blob storage..."
az monitor diagnostic-settings create `
    --name $diagName `
    --resource $workspaceResourceId `
    --storage-account $storageAccountId `
    --logs $logsJson `
    --metrics '[{"category":"AllMetrics","enabled":true}]' `
    --output none

Write-Host "Step 03 complete. Verify ADX ingestion scripts and blob sink." 