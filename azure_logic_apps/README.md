# Bearish Bot Stop Logic Apps

This directory contains the ARM templates and parameter files for deploying the "Stop API" and "Stop Core" Logic Apps.

## Deployment

1.  **Login to Azure:**
    ```bash
    az login
    az account set --subscription 74ab10ba-c96d-449e-97cb-ee4f9c0de714
    ```

2.  **Deploy Stop Core:**
    ```bash
    az deployment group create \
      --resource-group TradeBot \
      --name deploy-stop-core \
      --template-file stop-core.template.json \
      --parameters @stop-core.parameters.json
    ```

3.  **Deploy Stop API:**
    ```bash
    az deployment group create \
      --resource-group TradeBot \
      --name deploy-stop-api \
      --template-file stop-api.template.json \
      --parameters @stop-api.parameters.json
    ```

## RBAC Configuration

After deployment, you must assign the `Virtual Machine Contributor` role to the `bearish-bot-stop-core` Logic App's managed identity so it can execute commands on the VM.

```bash
# Get principalId of Stop Core logic app
principalId=$(az resource show \
  -g TradeBot \
  -n bearish-bot-stop-core \
  --resource-type "Microsoft.Logic/workflows" \
  --query "identity.principalId" -o tsv)

# Get VM ID
vmId=$(az vm show -g TradeBot -n BearishAlphaBot-VM-01 --query id -o tsv)

# Assign Role
az role assignment create \
  --assignee-object-id "$principalId" \
  --assignee-principal-type ServicePrincipal \
  --role "Virtual Machine Contributor" \
  --scope "$vmId"
```

## Usage

### Get Callback URL

To get the URL for the iPhone Shortcut:

```bash
az rest --method post --url "https://management.azure.com/subscriptions/74ab10ba-c96d-449e-97cb-ee4f9c0de714/resourceGroups/TradeBot/providers/Microsoft.Logic/workflows/bearish-bot-stop-api/triggers/manual/listCallbackUrl?api-version=2016-06-01"
```

### iPhone Shortcut

*   **URL:** (From above)
*   **Method:** POST
*   **Headers:** Content-Type: application/json
*   **Body:**
    ```json
    {"timeoutSeconds":90,"reason":"iphone_shortcut"}
    ```

## Troubleshooting

*   Check Logic App Run History in Azure Portal.
*   Verify Managed Identity permissions if "Check_Container_Status" fails with 403.
*   Ensure `acsemail` connection is valid.
