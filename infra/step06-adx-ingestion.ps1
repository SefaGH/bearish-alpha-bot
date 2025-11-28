# Step 06 - Azure Data Explorer ingestion wiring (Storage -> Event Grid -> Event Hub -> ADX)
# Usage: pwsh ./infra/step06-adx-ingestion.ps1

$ErrorActionPreference = "Stop"

$subscriptionId = "74ab10ba-c96d-449e-97cb-ee4f9c0de714"
$location = "westeurope"
$opsResourceGroup = "tradebot-ops"
$storageAccount = "bearishstorage"
$kustoCluster = "bearish-adx"
$kustoDatabase = "bearishdb"
$eventHubNamespace = "bearishreportingehns"
$eventHubName = "parsed-events"
$eventHubConsumerGroup = "adx"
$eventGridSubscription = "bearish-storage-to-eh"
$dataConnectionName = "bearish-parsed-events-eg"

function Ensure-ProviderRegistration {
    param([string]$Namespace)

    $state = az provider show `
        --namespace $Namespace `
        --query registrationState `
        --output tsv 2>$null

    if ($state -ne "Registered") {
        Write-Host "Registering resource provider $Namespace..."
        az provider register `
            --namespace $Namespace `
            --output none

        for ($attempt = 1; $attempt -le 12; $attempt++) {
            Start-Sleep -Seconds 5
            $state = az provider show `
                --namespace $Namespace `
                --query registrationState `
                --output tsv 2>$null
            if ($state -eq "Registered") {
                break
            }
        }

        if ($state -ne "Registered") {
            throw "Provider $Namespace failed to register."
        }
    } else {
        Write-Host "Provider $Namespace already registered."
    }
}

function Get-KustoAccessToken {
    az account get-access-token `
        --resource https://kusto.kusto.windows.net `
        --query accessToken `
        --output tsv
}

function Invoke-KustoManagementCommand {
    param(
        [string]$ClusterUri,
        [string]$Database,
        [string]$Command,
        [string]$AccessToken
    )

    $body = @{ db = $Database; csl = $Command } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$ClusterUri/v1/rest/mgmt" -Headers @{
        Authorization = "Bearer $AccessToken"
        Accept = "application/json"
        "Content-Type" = "application/json"
    } -Body $body
}

function Ensure-KustoIngestionMapping {
    param(
        [string]$ClusterUri,
        [string]$Database,
        [string]$Table,
        [string]$MappingName,
        [string]$MappingJson
    )

    Write-Host "Ensuring ingestion mapping '$MappingName' exists on $Table..."
    $accessToken = Get-KustoAccessToken
    $showResult = Invoke-KustoManagementCommand -ClusterUri $ClusterUri -Database $Database -Command ".show table $Table ingestion mappings" -AccessToken $accessToken
    $mappingExists = $false
    $rows = $showResult.Tables[0].Rows
    if ($rows) {
        foreach ($row in $rows) {
            if ($row[0] -eq $MappingName) {
                $mappingExists = $true
                break
            }
        }
    }

    if (-not $mappingExists) {
        $command = ".create-or-alter table $Table ingestion json mapping '$MappingName' '$MappingJson'"
        Invoke-KustoManagementCommand -ClusterUri $ClusterUri -Database $Database -Command $command -AccessToken $accessToken | Out-Null
    } else {
        Write-Host "Ingestion mapping '$MappingName' already present."
    }
}

Write-Host "Setting subscription context..."
az account set --subscription $subscriptionId

$clusterUri = "https://$kustoCluster.$location.kusto.windows.net"

Ensure-ProviderRegistration -Namespace "Microsoft.EventHub"
Ensure-ProviderRegistration -Namespace "Microsoft.EventGrid"

$mappingJson = '[{"column":"run_id","path":"$.run_id","datatype":"string"},{"column":"timestamp_utc","path":"$.timestamp_utc","datatype":"datetime"},{"column":"event_type","path":"$.event_type","datatype":"string"},{"column":"logger","path":"$.logger","datatype":"string"},{"column":"level","path":"$.level","datatype":"string"},{"column":"message","path":"$.message","datatype":"string"},{"column":"symbol","path":"$.symbol","datatype":"string"},{"column":"entry_price","path":"$.entry_price","datatype":"real"},{"column":"exit_price","path":"$.exit_price","datatype":"real"},{"column":"pnl_usd","path":"$.pnl_usd","datatype":"real"},{"column":"holding_time_s","path":"$.holding_time_s","datatype":"long"},{"column":"strategy","path":"$.strategy","datatype":"string"},{"column":"ml_confidence","path":"$.ml_confidence","datatype":"real"},{"column":"rl_confidence","path":"$.rl_confidence","datatype":"real"},{"column":"signal_score","path":"$.signal_score","datatype":"real"},{"column":"extra","path":"$.extra","datatype":"dynamic"}]'
Ensure-KustoIngestionMapping -ClusterUri $clusterUri -Database $kustoDatabase -Table "bearish_events" -MappingName "bearish_events_json_mapping" -MappingJson $mappingJson

$storageResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.Storage/storageAccounts/$storageAccount"
$clusterResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.Kusto/clusters/$kustoCluster"
$eventHubNamespaceResourceId = "/subscriptions/$subscriptionId/resourceGroups/$opsResourceGroup/providers/Microsoft.EventHub/namespaces/$eventHubNamespace"
$eventHubResourceId = "$eventHubNamespaceResourceId/eventhubs/$eventHubName"
$eventGridSubscriptionResourceId = "$storageResourceId/providers/Microsoft.EventGrid/eventSubscriptions/$eventGridSubscription"

Write-Host "Ensuring ADX cluster has a system-assigned managed identity..."
$clusterIdentity = az kusto cluster show `
    --name $kustoCluster `
    --resource-group $opsResourceGroup `
    --query identity `
    --output json | ConvertFrom-Json

if (-not $clusterIdentity -or -not $clusterIdentity.principalId) {
    Write-Host "Enabling managed identity on cluster $kustoCluster..."
    az kusto cluster update `
        --name $kustoCluster `
        --resource-group $opsResourceGroup `
        --type SystemAssigned `
        --output none

    for ($attempt = 1; $attempt -le 6; $attempt++) {
        Start-Sleep -Seconds 5
        $clusterIdentity = az kusto cluster show `
            --name $kustoCluster `
            --resource-group $opsResourceGroup `
            --query identity `
            --output json | ConvertFrom-Json

        if ($clusterIdentity -and $clusterIdentity.principalId) {
            break
        }
    }
}

if (-not $clusterIdentity.principalId) {
    throw "Failed to determine cluster managed identity principal ID."
}

Write-Host "Ensuring cluster identity has Storage Blob Data Reader on $storageAccount..."
$roleAssignments = az role assignment list `
    --assignee-object-id $clusterIdentity.principalId `
    --scope $storageResourceId `
    --output json | ConvertFrom-Json

if (-not ($roleAssignments | Where-Object { $_.roleDefinitionName -eq "Storage Blob Data Reader" })) {
    az role assignment create `
        --assignee-object-id $clusterIdentity.principalId `
        --assignee-principal-type ServicePrincipal `
        --role "Storage Blob Data Reader" `
        --scope $storageResourceId `
        --output none
} else {
    Write-Host "Storage Blob Data Reader role already assigned."
}

Write-Host "Ensuring Event Hubs namespace $eventHubNamespace exists..."
$namespaceExists = az eventhubs namespace show `
    --name $eventHubNamespace `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $namespaceExists) {
    az eventhubs namespace create `
        --name $eventHubNamespace `
        --resource-group $opsResourceGroup `
        --location $location `
        --sku Standard `
        --capacity 1 `
        --output none
} else {
    Write-Host "Event Hubs namespace already exists. Skipping creation."
}

Write-Host "Ensuring Event Hub $eventHubName exists..."
$eventHubExists = az eventhubs eventhub show `
    --name $eventHubName `
    --namespace-name $eventHubNamespace `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $eventHubExists) {
    az eventhubs eventhub create `
        --name $eventHubName `
        --namespace-name $eventHubNamespace `
        --resource-group $opsResourceGroup `
        --cleanup-policy Delete `
        --retention-time 24 `
        --partition-count 2 `
        --output none
} else {
    Write-Host "Event Hub already exists."
}

Write-Host "Ensuring consumer group '$eventHubConsumerGroup' exists on Event Hub..."
$consumerGroupExists = az eventhubs eventhub consumer-group show `
    --name $eventHubConsumerGroup `
    --eventhub-name $eventHubName `
    --namespace-name $eventHubNamespace `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $consumerGroupExists) {
    az eventhubs eventhub consumer-group create `
        --name $eventHubConsumerGroup `
        --eventhub-name $eventHubName `
        --namespace-name $eventHubNamespace `
        --resource-group $opsResourceGroup `
        --output none
} else {
    Write-Host "Consumer group already exists."
}

Write-Host "Ensuring Event Grid subscription $eventGridSubscription routes Storage events to Event Hub..."
$eventGridExists = az eventgrid event-subscription show `
    --name $eventGridSubscription `
    --source-resource-id $storageResourceId `
    --query name `
    --output tsv 2>$null

if (-not $eventGridExists) {
    az eventgrid event-subscription create `
        --name $eventGridSubscription `
        --source-resource-id $storageResourceId `
        --endpoint-type eventhub `
        --endpoint $eventHubResourceId `
        --included-event-types Microsoft.Storage.BlobCreated `
        --subject-begins-with "/blobServices/default/containers/parsed-events/" `
        --output none
} else {
    Write-Host "Event Grid subscription already exists."
}

Write-Host "Ensuring ADX Event Grid data connection $dataConnectionName exists..."
$dataConnectionExists = az kusto data-connection show `
    --cluster-name $kustoCluster `
    --database-name $kustoDatabase `
    --name $dataConnectionName `
    --resource-group $opsResourceGroup `
    --query name `
    --output tsv 2>$null

if (-not $dataConnectionExists) {
    az kusto data-connection event-grid create `
        --cluster-name $kustoCluster `
        --database-name $kustoDatabase `
        --name $dataConnectionName `
        --resource-group $opsResourceGroup `
        --location $location `
        --storage-account-resource-id $storageResourceId `
        --event-grid-resource-id $eventGridSubscriptionResourceId `
        --event-hub-resource-id $eventHubResourceId `
        --consumer-group $eventHubConsumerGroup `
        --data-format JSON `
        --blob-storage-event-type Microsoft.Storage.BlobCreated `
        --table-name bearish_events `
        --mapping-rule-name bearish_events_json_mapping `
        --managed-identity-resource-id $clusterResourceId `
        --output none
} else {
    Write-Host "Data connection already exists."
}

Write-Host "Step 06 complete. Upload a sample NDJSON file to 'parsed-events/' to verify ingestion."