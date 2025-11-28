<#
.SYNOPSIS
    Azure Automation Runbook for secure and resilient Bearish Alpha Bot execution on Azure VM
    
.DESCRIPTION
    Comprehensive runbook that:
    - Manages VM lifecycle (start/deallocate)
    - Implements concurrency control with lock file mechanism
    - Securely retrieves secrets from Azure Key Vault
    - Handles RunCommand timeouts with retry logic
    - Provides comprehensive logging to Azure Monitor
    - Ensures idempotent operation
    - Manages container lifecycle with timeout enforcement
    
.PARAMETER durationMinutes
    Trading session duration in minutes (1-85)
    
.PARAMETER resourceGroup
    Azure resource group containing the VM
    
.PARAMETER vmName
    Name of the Azure VM
    
.PARAMETER imageTag
    Docker image tag to use (default: vm-vmboot-9)
    
.PARAMETER keyVaultName
    Azure Key Vault name for secrets (optional, uses Managed Identity)
    
.PARAMETER kvSecretNames
    Comma-separated list of secret names to retrieve from Key Vault
    
.PARAMETER idempotencyToken
    Unique token to prevent duplicate executions
    
.PARAMETER maxDurationMinutes
    Maximum allowed duration to prevent Azure Agent timeout (default: 85)
    
.PARAMETER vmStartTimeoutSeconds
    Timeout for VM start operation (default: 180)
    
.PARAMETER maxRetries
    Maximum number of retry attempts (default: 3)
    
.PARAMETER retryDelaySeconds
    Initial delay between retry attempts (default: 30)
    
.PARAMETER forceRestart
    Force restart even if bot is already running (default: false)
    
.NOTES
    Author: Bearish Alpha Bot Team
    Version: 1.0.0
    Requires: Azure PowerShell modules, Managed Identity with appropriate permissions
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1, 85)]
    [int]$durationMinutes,
    
    [Parameter(Mandatory=$false)]
    [string]$resourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$false)]
    [string]$vmName = "BearishAlphaBot-VM-01",
    
    [Parameter(Mandatory=$false)]
    [string]$imageTag = "vm-vmboot-9",
    
    [Parameter(Mandatory=$false)]
    [string]$keyVaultName = "bearish-kv",
    
    [Parameter(Mandatory=$false)]
    [string]$kvSecretNames = "BINGX-KEY,BINGX-SECRET,TELEGRAM-BOT-TOKEN",
    
    [Parameter(Mandatory=$false)]
    [string]$idempotencyToken = "",
    
    [Parameter(Mandatory=$false)]
    [int]$maxDurationMinutes = 85,
    
    [Parameter(Mandatory=$false)]
    [int]$vmStartTimeoutSeconds = 180,
    
    [Parameter(Mandatory=$false)]
    [int]$maxRetries = 3,
    
    [Parameter(Mandatory=$false)]
    [int]$retryDelaySeconds = 30,
    
    [Parameter(Mandatory=$false)]
    [switch]$forceRestart = $false
)

#region Helper Functions

function Write-StructuredLog {
    <#
    .SYNOPSIS
        Writes structured log entries with timestamp and level
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        
        [Parameter(Mandatory=$false)]
        [ValidateSet('INFO', 'WARNING', 'ERROR', 'SUCCESS')]
        [string]$Level = 'INFO'
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ"
    $logEntry = @{
        timestamp = $timestamp
        level = $Level
        message = $Message
        runbook = $PSCommandPath
        jobId = $PSPrivateMetadata.JobId.Guid
    }
    
    $jsonLog = $logEntry | ConvertTo-Json -Compress
    Write-Output "STRUCTURED_LOG: $jsonLog"
    
    # Also write to standard output for Azure Monitor integration
    Write-Output "[$timestamp] [$Level] $Message"
}

function Invoke-WithRetry {
    <#
    .SYNOPSIS
        Executes a script block with exponential backoff retry logic
    #>
    param(
        [Parameter(Mandatory=$true)]
        [scriptblock]$ScriptBlock,
        
        [Parameter(Mandatory=$false)]
        [int]$MaxAttempts = 3,
        
        [Parameter(Mandatory=$false)]
        [int]$InitialDelaySeconds = 30,
        
        [Parameter(Mandatory=$false)]
        [string]$OperationName = "Operation"
    )
    
    $attempt = 1
    
    while ($attempt -le $MaxAttempts) {
        try {
            Write-StructuredLog "Executing $OperationName (Attempt $attempt of $MaxAttempts)" -Level INFO
            
            $result = & $ScriptBlock
            
            Write-StructuredLog "$OperationName completed successfully" -Level SUCCESS
            return $result
            
        } catch {
            $errorMessage = $_.Exception.Message
            Write-StructuredLog "$OperationName failed: $errorMessage" -Level ERROR
            
            if ($attempt -ge $MaxAttempts) {
                Write-StructuredLog "Max retries ($MaxAttempts) exceeded for $OperationName" -Level ERROR
                throw $_
            }
            
            # Exponential backoff
            $delaySeconds = $InitialDelaySeconds * [Math]::Pow(2, $attempt - 1)
            Write-StructuredLog "Retrying in $delaySeconds seconds..." -Level WARNING
            Start-Sleep -Seconds $delaySeconds
            
            $attempt++
        }
    }
}

function Test-ConcurrencyLock {
    <#
    .SYNOPSIS
        Checks if another bot instance is running via lock file
    #>
    param(
        [string]$ResourceGroup,
        [string]$VMName
    )
    
    $lockCheckScript = @'
LOCK_FILE="/tmp/bearish_bot_automation.lock"
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE")
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "LOCKED:$LOCK_PID"
        exit 0
    else
        echo "STALE_LOCK:$LOCK_PID"
        rm -f "$LOCK_FILE"
    fi
fi
echo "FREE"
'@
    
    Write-StructuredLog "Checking for concurrent executions..." -Level INFO
    
    $result = Invoke-AzVMRunCommand `
        -ResourceGroupName $ResourceGroup `
        -VMName $VMName `
        -CommandId 'RunShellScript' `
        -ScriptString $lockCheckScript `
        -ErrorAction Stop
    
    $output = $result.Value[0].Message
    
    if ($output -match "^LOCKED:(\d+)") {
        $lockPid = $Matches[1]
        Write-StructuredLog "Bot is already running (PID: $lockPid)" -Level WARNING
        return $false
    }
    
    if ($output -match "^STALE_LOCK") {
        Write-StructuredLog "Removed stale lock file" -Level INFO
    }
    
    Write-StructuredLog "No concurrent execution detected" -Level INFO
    return $true
}

function Set-ConcurrencyLock {
    <#
    .SYNOPSIS
        Creates a lock file to prevent concurrent executions
    #>
    param(
        [string]$ResourceGroup,
        [string]$VMName
    )
    
    $lockScript = @'
LOCK_FILE="/tmp/bearish_bot_automation.lock"
echo $$ > "$LOCK_FILE"
echo "Lock created with PID: $$"
'@
    
    Invoke-AzVMRunCommand `
        -ResourceGroupName $ResourceGroup `
        -VMName $VMName `
        -CommandId 'RunShellScript' `
        -ScriptString $lockScript `
        -ErrorAction Stop | Out-Null
    
    Write-StructuredLog "Concurrency lock acquired" -Level SUCCESS
}

function Remove-ConcurrencyLock {
    <#
    .SYNOPSIS
        Removes the lock file
    #>
    param(
        [string]$ResourceGroup,
        [string]$VMName
    )
    
    $unlockScript = 'rm -f /tmp/bearish_bot_automation.lock; echo "Lock removed"'
    
    try {
        Invoke-AzVMRunCommand `
            -ResourceGroupName $ResourceGroup `
            -VMName $VMName `
            -CommandId 'RunShellScript' `
            -ScriptString $unlockScript `
            -ErrorAction SilentlyContinue | Out-Null
        
        Write-StructuredLog "Concurrency lock released" -Level INFO
    } catch {
        Write-StructuredLog "Failed to release lock (non-critical): $($_.Exception.Message)" -Level WARNING
    }
}

function Test-ContainerStatus {
    <#
    .SYNOPSIS
        Checks if the bot container is currently running on the VM
    #>
    param(
        [string]$ResourceGroup,
        [string]$VMName,
        [string]$ContainerName = "bearish-bot"
    )
    
    $statusCheckScript = @"
#!/bin/bash
CONTAINER_NAME="$ContainerName"

# Check if container exists and is running
if sudo docker ps --filter "name=`$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^`${CONTAINER_NAME}`$"; then
    echo "STATUS:RUNNING"
    echo "STARTED:`$(sudo docker inspect `$CONTAINER_NAME --format '{{.State.StartedAt}}' 2>/dev/null || echo 'UNKNOWN')"
    echo "UPTIME:`$(sudo docker ps --filter "name=`$CONTAINER_NAME" --format '{{.Status}}')"
    echo "LOGS_START"
    sudo docker logs `$CONTAINER_NAME --tail 10 2>&1
    echo "LOGS_END"
    exit 0
elif sudo docker ps -a --filter "name=`$CONTAINER_NAME" --format "{{.Names}}" | grep -q "^`${CONTAINER_NAME}`$"; then
    echo "STATUS:STOPPED"
    echo "EXIT_CODE:`$(sudo docker inspect `$CONTAINER_NAME --format '{{.State.ExitCode}}' 2>/dev/null || echo 'UNKNOWN')"
    exit 0
else
    echo "STATUS:NOT_FOUND"
    exit 0
fi
"@
    
    Write-StructuredLog "Checking container status on VM..." -Level INFO
    
    try {
        $result = Invoke-AzVMRunCommand `
            -ResourceGroupName $ResourceGroup `
            -VMName $VMName `
            -CommandId 'RunShellScript' `
            -ScriptString $statusCheckScript `
            -ErrorAction Stop
        
        $output = $result.Value[0].Message
        
        # Parse output
        $statusInfo = @{
            Status = "UNKNOWN"
            StartedAt = $null
            Uptime = $null
            Logs = @()
            ExitCode = $null
        }
        
        $inLogs = $false
        foreach ($line in ($output -split "`n")) {
            $line = $line.Trim()
            
            if ($line -match "^STATUS:(.+)") {
                $statusInfo.Status = $Matches[1]
            }
            elseif ($line -match "^STARTED:(.+)") {
                $statusInfo.StartedAt = $Matches[1]
            }
            elseif ($line -match "^UPTIME:(.+)") {
                $statusInfo.Uptime = $Matches[1]
            }
            elseif ($line -match "^EXIT_CODE:(.+)") {
                $statusInfo.ExitCode = $Matches[1]
            }
            elseif ($line -eq "LOGS_START") {
                $inLogs = $true
            }
            elseif ($line -eq "LOGS_END") {
                $inLogs = $false
            }
            elseif ($inLogs -and $line) {
                $statusInfo.Logs += $line
            }
        }
        
        return $statusInfo
        
    } catch {
        Write-StructuredLog "Failed to check container status: $($_.Exception.Message)" -Level WARNING
        return @{Status = "ERROR"; Error = $_.Exception.Message}
    }
}

function Stop-ExistingContainer {
    <#
    .SYNOPSIS
        Stops and removes existing container
    #>
    param(
        [string]$ResourceGroup,
        [string]$VMName,
        [string]$ContainerName = "bearish-bot"
    )
    
    $stopScript = @"
#!/bin/bash
CONTAINER_NAME="$ContainerName"
echo "Stopping container: `$CONTAINER_NAME"
sudo docker stop `$CONTAINER_NAME 2>/dev/null || true
sudo docker rm `$CONTAINER_NAME 2>/dev/null || true
echo "Container stopped and removed"
"@
    
    Write-StructuredLog "Stopping existing container..." -Level INFO
    
    try {
        Invoke-AzVMRunCommand `
            -ResourceGroupName $ResourceGroup `
            -VMName $VMName `
            -CommandId 'RunShellScript' `
            -ScriptString $stopScript `
            -ErrorAction Stop | Out-Null
        
        Write-StructuredLog "Existing container stopped successfully" -Level SUCCESS
        Start-Sleep -Seconds 3
        
    } catch {
        Write-StructuredLog "Failed to stop container: $($_.Exception.Message)" -Level ERROR
        throw
    }
}

#endregion

#region Main Execution

try {
    Write-StructuredLog "=== Bearish Alpha Bot Automation Runbook Started ===" -Level INFO
    Write-StructuredLog "Parameters: Duration=$durationMinutes min, Image=$imageTag, VM=$vmName" -Level INFO
    
    # Validate parameters
    if ($durationMinutes -le 0 -or $durationMinutes -gt $maxDurationMinutes) {
        throw "durationMinutes must be between 1 and $maxDurationMinutes"
    }
    
    $durationSeconds = $durationMinutes * 60
    
    # Check idempotency
    if ($idempotencyToken) {
        $tokenStoragePath = "C:\Temp\runbook_tokens.txt"
        if (Test-Path $tokenStoragePath) {
            $existingTokens = Get-Content $tokenStoragePath
            if ($existingTokens -contains $idempotencyToken) {
                Write-StructuredLog "Idempotency token '$idempotencyToken' already processed. Exiting." -Level WARNING
                return
            }
        }
    }
    
    # Authenticate using Managed Identity
    Write-StructuredLog "Authenticating with Managed Identity..." -Level INFO
    try {
        Disable-AzContextAutosave -Scope Process | Out-Null
        $AzureContext = (Connect-AzAccount -Identity).Context
        $AzureContext = Set-AzContext -SubscriptionName $AzureContext.Subscription -DefaultProfile $AzureContext
        Write-StructuredLog "Successfully authenticated" -Level SUCCESS
    } catch {
        throw "Managed Identity authentication failed: $($_.Exception.Message)"
    }
    
    # Step 1: Ensure VM is running
    Write-StructuredLog "Step 1: Checking VM status..." -Level INFO
    
    $vmStatus = Invoke-WithRetry -ScriptBlock {
        Get-AzVM -ResourceGroupName $resourceGroup -Name $vmName -Status -ErrorAction Stop
    } -MaxAttempts 2 -InitialDelaySeconds 10 -OperationName "Get VM Status"
    
    $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like 'PowerState/*' }).DisplayStatus
    Write-StructuredLog "Current VM state: $powerState" -Level INFO
    
    if ($powerState -ne "VM running") {
        Write-StructuredLog "Starting VM..." -Level INFO
        
        Start-AzVM -ResourceGroupName $resourceGroup -Name $vmName -NoWait | Out-Null
        
        $startTime = Get-Date
        $isRunning = $false
        
        while (-not $isRunning -and ((Get-Date) -lt $startTime.AddSeconds($vmStartTimeoutSeconds))) {
            Start-Sleep -Seconds 10
            
            $vmStatus = Get-AzVM -ResourceGroupName $resourceGroup -Name $vmName -Status
            $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like 'PowerState/*' }).DisplayStatus
            
            if ($powerState -eq "VM running") {
                $isRunning = $true
                Write-StructuredLog "VM is now running" -Level SUCCESS
            } else {
                Write-StructuredLog "Waiting for VM... Current state: $powerState" -Level INFO
            }
        }
        
        if (-not $isRunning) {
            throw "VM failed to start within $vmStartTimeoutSeconds seconds"
        }
        
        # Wait for Azure VM Agent to be ready
        Write-StructuredLog "Waiting for VM Agent to be ready..." -Level INFO
        Start-Sleep -Seconds 15
    }
    
    # Step 2: Check if bot is already running
    Write-StructuredLog "Step 2: Checking if bot container is already running..." -Level INFO
    
    $containerStatus = Test-ContainerStatus -ResourceGroup $resourceGroup -VMName $vmName
    
    Write-StructuredLog "Container status: $($containerStatus.Status)" -Level INFO
    
    switch ($containerStatus.Status) {
        "RUNNING" {
            if (-not $forceRestart) {
                Write-StructuredLog "❌ Bot is already RUNNING. Aborting to prevent duplicate execution." -Level WARNING
                Write-StructuredLog "Container started at: $($containerStatus.StartedAt)" -Level INFO
                Write-StructuredLog "Container uptime: $($containerStatus.Uptime)" -Level INFO
                
                if ($containerStatus.Logs.Count -gt 0) {
                    Write-StructuredLog "Recent container logs:" -Level INFO
                    foreach ($logLine in $containerStatus.Logs) {
                        Write-StructuredLog "  $logLine" -Level INFO
                    }
                }
                
                Write-StructuredLog "To force restart, run with -forceRestart parameter" -Level INFO
                throw "Bot already running. Use -forceRestart to override."
            } else {
                Write-StructuredLog "⚠️ Bot is RUNNING but forceRestart=true. Stopping existing container..." -Level WARNING
                Stop-ExistingContainer -ResourceGroup $resourceGroup -VMName $vmName
                Write-StructuredLog "✅ Existing container stopped. Proceeding with fresh start." -Level SUCCESS
            }
        }
        "STOPPED" {
            Write-StructuredLog "⚠️ Container exists but is STOPPED (Exit Code: $($containerStatus.ExitCode)). Removing..." -Level WARNING
            Stop-ExistingContainer -ResourceGroup $resourceGroup -VMName $vmName
            Write-StructuredLog "✅ Stopped container removed. Proceeding with fresh start." -Level SUCCESS
        }
        "NOT_FOUND" {
            Write-StructuredLog "✅ No existing container found. Proceeding with fresh start." -Level INFO
        }
        default {
            Write-StructuredLog "⚠️ Unknown container status. Proceeding cautiously..." -Level WARNING
        }
    }
    
    # Step 3: Concurrency Control
    Write-StructuredLog "Step 3: Checking for concurrent executions..." -Level INFO
    
    $canProceed = Invoke-WithRetry -ScriptBlock {
        Test-ConcurrencyLock -ResourceGroup $resourceGroup -VMName $vmName
    } -MaxAttempts 2 -InitialDelaySeconds 10 -OperationName "Concurrency Check"
    
    if (-not $canProceed) {
        throw "Another bot instance is already running. Aborting to prevent concurrent execution."
    }
    
    # Acquire lock
    Set-ConcurrencyLock -ResourceGroup $resourceGroup -VMName $vmName
    
    # Step 4: Retrieve secrets from Key Vault
    Write-StructuredLog "Step 4: Retrieving secrets from Key Vault..." -Level INFO
    
    $envVars = @()
    
    if ($keyVaultName -and $kvSecretNames) {
        $secretNames = $kvSecretNames.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
        
        foreach ($secretName in $secretNames) {
            try {
                Write-StructuredLog "Retrieving secret: $secretName" -Level INFO
                
                $secret = Get-AzKeyVaultSecret -VaultName $keyVaultName -Name $secretName -AsPlainText -ErrorAction Stop
                
                # Escape single quotes for shell safety (using proper PowerShell escaping)
                $escapedValue = $secret -replace "'", "'\\''"
                $envVars += "$secretName='$escapedValue'"
                
                Write-StructuredLog "Secret retrieved: $secretName" -Level SUCCESS
            } catch {
                Write-StructuredLog "Failed to retrieve secret '$secretName': $($_.Exception.Message)" -Level ERROR
                throw
            }
        }
    }
    
    # Step 5: Prepare and execute container
    Write-StructuredLog "Step 5: Preparing container execution script..." -Level INFO
    
    $envFileContent = $envVars -join "`n"
    $base64Env = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($envFileContent))
    
    # Build bash script using string concatenation to avoid PowerShell parsing
    $containerScript = "#!/bin/bash`nset -euo pipefail`n`n"
    $containerScript += "LOCK_FILE=`"/tmp/bearish_bot_automation.lock`"`n"
    $containerScript += "ENV_FILE=`"/tmp/.bearish_env_tmp`"`n"
    $containerScript += "CONTAINER_NAME=`"bearish-bot`"`n"
    $containerScript += "IMAGE=`"bearishalphabot.azurecr.io/bearish-bot:$imageTag`"`n"
    $containerScript += "DURATION=$durationSeconds`n`n"
    
    $containerScript += "echo `"=== Bearish Bot Container Setup ===`"`n"
    $containerScript += "echo `"Duration: `$DURATION seconds`"`n"
    $containerScript += "echo `"Image: `$IMAGE`"`n`n"
    
    $containerScript += "# Create secure env file`n"
    $containerScript += "if [ -n `"$base64Env`" ]; then`n"
    $containerScript += "    echo `"$base64Env`" | base64 --decode > `"`$ENV_FILE`"`n"
    $containerScript += "    chmod 600 `"`$ENV_FILE`"`n"
    $containerScript += "    echo `"✓ Environment file created`"`n"
    $containerScript += "fi`n`n"
    
    $containerScript += "# Idempotent cleanup`n"
    $containerScript += "echo `"Cleaning up existing container...`"`n"
    $containerScript += "docker stop `"`$CONTAINER_NAME`" 2>/dev/null || true`n"
    $containerScript += "docker rm `"`$CONTAINER_NAME`" 2>/dev/null || true`n`n"
    
    $containerScript += "# Pull latest image`n"
    $containerScript += "echo `"Pulling Docker image...`"`n"
    $containerScript += "docker pull `"`$IMAGE`"`n`n"
    
    $containerScript += "# Start container with timeout`n"
    $containerScript += "echo `"Starting container (detached mode)...`"`n"
    $containerScript += "ENV_FILE_ARG=`"`"`n"
    $containerScript += "if [ -f `"`$ENV_FILE`" ]; then`n"
    $containerScript += "    ENV_FILE_ARG=`"--env-file `$ENV_FILE`"`n"
    $containerScript += "fi`n`n"
    
    $containerScript += "docker run -d ```n"
    $containerScript += "    --restart=no ```n"
    $containerScript += "    --name `"`$CONTAINER_NAME`" ```n"
    $containerScript += "    `$ENV_FILE_ARG ```n"
    $containerScript += "    -e TRADING_DURATION=`$DURATION ```n"
    $containerScript += "    -e TRADING_MODE=paper ```n"
    $containerScript += "    -v /mnt/bearish/logs:/app/logs ```n"
    $containerScript += "    -v /mnt/bearish/data:/app/data ```n"
    $containerScript += "    `"`$IMAGE`"`n`n"
    
    $containerScript += "echo `"✓ Container started`"`n`n"
    
    $containerScript += "# Wait for container with timeout`n"
    $containerScript += "echo `"Waiting for container completion (timeout: `$DURATION seconds)...`"`n"
    $containerScript += "timeout `$DURATION docker wait `"`$CONTAINER_NAME`" && {`n"
    $containerScript += "    echo `"✓ Container exited normally`"`n"
    $containerScript += "} || {`n"
    $containerScript += "    echo `"⚠ Timeout reached, stopping container gracefully...`"`n"
    $containerScript += "    docker stop -t 30 `"`$CONTAINER_NAME`" 2>/dev/null || true`n"
    $containerScript += "}`n`n"
    
    $containerScript += "# Cleanup`n"
    $containerScript += "echo `"Cleaning up...`"`n"
    $containerScript += "docker rm `"`$CONTAINER_NAME`" 2>/dev/null || true`n`n"
    
    $containerScript += "if [ -f `"`$ENV_FILE`" ]; then`n"
    $containerScript += "    shred -u `"`$ENV_FILE`" 2>/dev/null || rm -f `"`$ENV_FILE`"`n"
    $containerScript += "fi`n`n"
    
    $containerScript += "rm -f `"`$LOCK_FILE`"`n`n"
    $containerScript += "echo `"=== Container execution completed ===`"`n"
    
    # Step 6: Execute with retry
    Write-StructuredLog "Step 6: Executing container on VM..." -Level INFO
    
    # Base64 encode the entire bash script to pass as parameter
    $scriptBytes = [Text.Encoding]::UTF8.GetBytes($containerScript)
    $base64Script = [Convert]::ToBase64String($scriptBytes)
    
    # Simple wrapper script that decodes and executes the base64 payload
    $wrapperScript = @"
#!/bin/bash
echo "\$1" | base64 --decode | bash
"@
    
    $runResult = Invoke-WithRetry -ScriptBlock {
        Invoke-AzVMRunCommand `
            -ResourceGroupName $resourceGroup `
            -VMName $vmName `
            -CommandId 'RunShellScript' `
            -ScriptString $wrapperScript `
            -Parameter @{arg1 = $base64Script} `
            -ErrorAction Stop
    } -MaxAttempts $maxRetries -InitialDelaySeconds $retryDelaySeconds -OperationName "Container Execution"
    
    # Log output
    foreach ($msg in $runResult.Value) {
        if ($msg.Message) {
            Write-StructuredLog "VM Output: $($msg.Message)" -Level INFO
        }
    }
    
    # Step 7: Deallocate VM
    Write-StructuredLog "Step 7: Deallocating VM..." -Level INFO
    
    try {
        Stop-AzVM -ResourceGroupName $resourceGroup -Name $vmName -Force -NoWait | Out-Null
        Write-StructuredLog "VM deallocation initiated" -Level SUCCESS
    } catch {
        Write-StructuredLog "VM deallocation failed (non-critical): $($_.Exception.Message)" -Level WARNING
    }
    
    # Save idempotency token
    if ($idempotencyToken) {
        $tokenStoragePath = "C:\Temp\runbook_tokens.txt"
        New-Item -ItemType Directory -Path (Split-Path $tokenStoragePath) -Force | Out-Null
        Add-Content -Path $tokenStoragePath -Value $idempotencyToken
        Write-StructuredLog "Idempotency token saved" -Level INFO
    }
    
    Write-StructuredLog "=== Runbook completed successfully ===" -Level SUCCESS
    
} catch {
    $errorMessage = $_.Exception.Message
    $stackTrace = $_.ScriptStackTrace
    
    Write-StructuredLog "FATAL ERROR: $errorMessage" -Level ERROR
    Write-StructuredLog "Stack Trace: $stackTrace" -Level ERROR
    
    # Attempt to release lock on failure
    try {
        Remove-ConcurrencyLock -ResourceGroup $resourceGroup -VMName $vmName
    } catch {
        Write-StructuredLog "Failed to release lock during error handling" -Level WARNING
    }
    
    throw
} finally {
    Write-StructuredLog "Runbook execution ended" -Level INFO
}

#endregion
