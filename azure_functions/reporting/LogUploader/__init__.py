import logging
import json
import os
import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.compute.models import RunCommandInput

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Upload trading bot logs from Azure VM to blob storage using Azure SDK.
    Uses managed identity for authentication (no az CLI required).
    """
    logging.info('LogUploader function triggered - PRODUCTION VERSION (Azure SDK)')
    
    vm_name = req.params.get('vmName', 'BearishAlphaBot-VM-01')
    resource_group = req.params.get('resourceGroup', 'TradeBot')
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID', '74ab10ba-c96d-449e-97cb-ee4f9c0de714')
    
    # Bash script to upload log using managed identity + Azure Storage REST API
    # This runs ON THE VM, so VM's managed identity is used
    bash_script = """
set -e

LOG_DIR="/mnt/bearish/logs"
STORAGE_ACCOUNT="bearishstorage"
CONTAINER="raw-logs"

# Find latest log file
LATEST_LOG=$(ls -t $LOG_DIR/live_trading_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo '{"status":"error","message":"No log files found"}'
    exit 1
fi

FILENAME=$(basename "$LATEST_LOG")

# Get managed identity token for Storage
TOKEN=$(curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://storage.azure.com/" | jq -r .access_token)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
    echo '{"status":"error","message":"Failed to get managed identity token"}'
    exit 1
fi

# Upload using Storage REST API
BLOB_URL="https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER}/${FILENAME}"
CONTENT_LENGTH=$(stat -c%s "$LATEST_LOG")

curl -X PUT "$BLOB_URL" \
    -H "Authorization: Bearer $TOKEN" \
    -H "x-ms-blob-type: BlockBlob" \
    -H "x-ms-version: 2021-08-06" \
    -H "Content-Length: $CONTENT_LENGTH" \
    --data-binary "@$LATEST_LOG" \
    -w "%{http_code}" -o /dev/null -s > /tmp/upload_status.txt

HTTP_CODE=$(cat /tmp/upload_status.txt)

if [ "$HTTP_CODE" = "201" ]; then
    echo '{"status":"success","message":"Log uploaded","file":"'$FILENAME'","size":'$CONTENT_LENGTH'}'
else
    echo '{"status":"error","message":"Upload failed with HTTP '$HTTP_CODE'"}'
    exit 1
fi
"""
    
    try:
        # Authenticate using function app's managed identity
        credential = DefaultAzureCredential()
        compute_client = ComputeManagementClient(credential, subscription_id)
        
        logging.info(f"Executing RunCommand on VM: {vm_name} in RG: {resource_group}")
        
        # Create RunCommand input
        run_command_input = RunCommandInput(
            command_id='RunShellScript',
            script=[bash_script]
        )
        
        # Execute command (async operation)
        poller = compute_client.virtual_machines.begin_run_command(
            resource_group_name=resource_group,
            vm_name=vm_name,
            parameters=run_command_input
        )
        
        # Wait for completion (timeout: 120 seconds)
        result = poller.result(timeout=120)
        
        # Parse VM output
        if result.value and len(result.value) > 0:
            vm_output = result.value[0].message
            logging.info(f"VM command raw output: {vm_output}")
            
            try:
                # Extract JSON from output (skip "Enable succeeded:" prefix and [stdout]/[stderr])
                # Format: "Enable succeeded: \n[stdout]\n{JSON}\n\n[stderr]\n"
                if '[stdout]' in vm_output:
                    stdout_section = vm_output.split('[stdout]')[1].split('[stderr]')[0].strip()
                    result_data = json.loads(stdout_section)
                else:
                    # Fallback: try parsing entire output
                    result_data = json.loads(vm_output)
                
                logging.info(f"Parsed result: {result_data}")
                
                if result_data.get('status') == 'success':
                    return func.HttpResponse(
                        json.dumps(result_data),
                        status_code=200,
                        mimetype="application/json"
                    )
                else:
                    return func.HttpResponse(
                        json.dumps(result_data),
                        status_code=500,
                        mimetype="application/json"
                    )
                    
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse VM output as JSON: {e}")
                return func.HttpResponse(
                    json.dumps({
                        "status": "error",
                        "message": f"Failed to parse VM output: {str(e)}",
                        "raw_output": vm_output[:500]
                    }),
                    status_code=500,
                    mimetype="application/json"
                )
        else:
            logging.warning("VM RunCommand returned no output")
            return func.HttpResponse(
                json.dumps({
                    "status": "error",
                    "message": "VM command returned no output",
                    "vm": vm_name
                }),
                status_code=500,
                mimetype="application/json"
            )
            
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error executing VM RunCommand: {error_msg}")
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "message": f"RunCommand failed: {error_msg}",
                "vm": vm_name,
                "resource_group": resource_group
            }),
            status_code=500,
            mimetype="application/json"
        )


