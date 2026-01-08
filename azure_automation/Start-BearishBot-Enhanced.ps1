<#
.SYNOPSIS
    Azure VM üzerinde Docker Trade Bot'unu ASENKRON (Fire-and-Forget) modda başlatır.

.DESCRIPTION
    Bu Runbook:
    1. VM'e bağlanır ve disk temizliği yapar.
    2. Botu arka planda (Detached) başlatır.
    3. Botun ayağa kalktığını (Health Check) doğrular.
    4. Bot bitmeden hemen BAŞARIYLA sonlanır.
    
    NOT: Botun çalışıp çalışmadığının takibi Logic App (Polling Loop) tarafından yapılır.

.PARAMETER ResourceGroup
    VM'in bulunduğu kaynak grubu (Varsayılan: TradeBot)
.PARAMETER VMName
    Botun çalışacağı sanal makine adı (Varsayılan: BearishAlphaBot-VM-01)
.PARAMETER ImageTag
    Docker imajının tag'i (örn: vm-vmboot-12)
.PARAMETER DurationMinutes
    Botun kaç dakika çalışacağı
.PARAMETER IdempotencyToken
    Job ID (Takip için)
.PARAMETER ForceRestart
    Mevcut container varsa zorla durdurup yeniden başlat
#>

param(
    [Parameter(Mandatory=$false)]
    [string] $ResourceGroup = "TradeBot",
    
    [Parameter(Mandatory=$false)]
    [string] $VMName = "BearishAlphaBot-VM-01",
    
    [Parameter(Mandatory=$false)]
    [string] $ImageTag = "",
    
    [Parameter(Mandatory=$false)]
    [int] $DurationMinutes = 60,
    
    [Parameter(Mandatory=$false)]
    [string] $IdempotencyToken = "",
    
    [Parameter(Mandatory=$false)]
    [bool] $ForceRestart = $false,

    # Logic App payload-mapped params (may arrive as bool, int, or string)
    [Parameter(Mandatory=$false)]
    [object] $DebugMode = $null,

    [Parameter(Mandatory=$false)]
    [string] $LogLevel = "",

    # Logic App payload-mapped target environment (controls BingX routing only)
    # Allowed: "vst" | "prod" (default: prod if missing)
    [Parameter(Mandatory=$false)]
    [string] $TargetEnv = ""
)

$ErrorActionPreference = "Stop"

Write-Output "╔════════════════════════════════════════════════════════╗"
Write-Output "║   BEARISH BOT - ASYNC STARTUP (FIRE & FORGET)          ║"
Write-Output "╚════════════════════════════════════════════════════════╝"
Write-Output "Mode: Asynchronous (Logic App will monitor the running bot)"
Write-Output "Parameters:"
Write-Output "  VM: $VMName ($ResourceGroup)"
Write-Output "  Image: $ImageTag"
Write-Output "  Duration: $DurationMinutes min"
Write-Output ""

try {
    # 1. AUTHENTICATION
    Write-Output "[1/6] Authenticating..."
    Connect-AzAccount -Identity | Out-Null
    
    # 1b. FETCH IMAGE TAG FROM APP CONFIGURATION (if not provided)
    if ([string]::IsNullOrEmpty($ImageTag)) {
        Write-Output "[1b/6] Fetching image tag from Azure App Configuration..."
        try {
            $appConfigName = "appcs-bearish-bot"
            $keyName = "DOCKER_IMAGE_TAG"
            $label = "production"
            
            $imageTagValue = Get-AzAppConfigurationKeyValue `
                -Endpoint "https://$appConfigName.azconfig.io" `
                -Key $keyName `
                -Label $label `
                -ErrorAction Stop
            
            $ImageTag = $imageTagValue.Value
            Write-Output "      ✅ Image tag from App Config: $ImageTag"
        }
        catch {
            $ImageTag = "appconfig-rest-api-v2"  # Production fallback
            Write-Output "      ⚠️ App Config fetch failed, using fallback: $ImageTag"
            Write-Output "      Error: $($_.Exception.Message)"
        }
    }
    else {
        Write-Output "      ℹ️ Using provided image tag: $ImageTag"
    }
    
    # 2. VM STATUS CHECK
    Write-Output "[2/6] Checking VM status..."
    $vmStatus = Get-AzVM -ResourceGroupName $ResourceGroup -Name $VMName -Status
    $powerState = ($vmStatus.Statuses | Where-Object { $_.Code -like "PowerState/*" }).DisplayStatus
    
    if ($powerState -ne "VM running") {
        throw "VM is not running ($powerState). Logic App should start VM first."
    }

    function Normalize-Bool($val) {
        if ($null -eq $val) { return $null }
        $s = $val.ToString().Trim().ToLower()
        if ($s -in @("true","1","yes")) { return $true }
        if ($s -in @("false","0","no")) { return $false }
        return $null
    }

    function Validate-LogLevel([string]$val) {
        $allowed = @("DEBUG","INFO","WARNING","ERROR","CRITICAL")
        $s = $val.Trim().ToUpper()
        if ($allowed -contains $s) { return $s }
        throw "Invalid logLevel: '$val'. Allowed: $($allowed -join ', ')"
    }

    function Normalize-TargetEnv([string]$val) {
        if ([string]::IsNullOrWhiteSpace($val)) { return "prod" }
        $s = $val.Trim().ToLower()
        if ($s -in @("prod","production")) { return "prod" }
        if ($s -in @("vst","demo","sandbox")) { return "vst" }
        throw "Invalid targetEnv: '$val'. Allowed: vst|prod."
    }

    # === Normalize payload-mapped values ===
    $debugModeNorm = Normalize-Bool $DebugMode
    $logLevelNorm = if ([string]::IsNullOrWhiteSpace($LogLevel)) { "" } else { Validate-LogLevel $LogLevel }
    $targetEnvNorm = Normalize-TargetEnv $TargetEnv

    # Defaulting rules:
    # - debugMode=true + missing logLevel => logLevel=DEBUG
    # - logLevel=DEBUG + missing debugMode => debugMode=true
    if ($debugModeNorm -eq $true -and -not $logLevelNorm) {
        $logLevelNorm = "DEBUG"
    }
    if ($logLevelNorm -eq "DEBUG" -and $null -eq $debugModeNorm) {
        $debugModeNorm = $true
    }

    # Fail fast if DebugMode is present but unparseable
    if ($null -eq $debugModeNorm -and $null -ne $DebugMode) {
        throw "Invalid debugMode: '$DebugMode'. Expected true/false/1/0/yes/no."
    }

    $debugModeStr = if ($null -ne $debugModeNorm) { if ($debugModeNorm) { "true" } else { "false" } } else { "" }
    $logLevelStr = $logLevelNorm

    # Force restart only when override is requested
    $overrideRequested = [bool]($debugModeStr -or $logLevelStr)
    if ($overrideRequested) {
        $ForceRestart = $true
    }

    $forceRecreateStr = if ($ForceRestart) { "true" } else { "false" }

    Write-Output "Run config: debugMode=$debugModeStr logLevel=$logLevelStr forceRestart=$ForceRestart"
    Write-Output "Routing: targetEnv=$targetEnvNorm => BINGX_ENV=$targetEnvNorm (single API key; base URL routing only)"
    if ($overrideRequested) {
        Write-Output "Override requested => container recreate enforced"
    }

    # 3. PREPARE SCRIPT
    Write-Output "[3/6] Preparing startup script..."

    $tradingDurationSeconds = if ($DurationMinutes -eq 0) { "" } else { $DurationMinutes * 60 }

    Write-Output "vm_run_session flags: debugMode=$debugModeStr logLevel=$logLevelStr --force-recreate $forceRecreateStr"

    # IMPORTANT: Use a single-quoted here-string so PowerShell does NOT evaluate bash syntax like $(...) or $VAR.
    $startupScript = @'
#!/usr/bin/env bash
set -euo pipefail

echo "=== BOT INITIALIZATION ==="
echo "Date: $(date)"

# --- ADIM 1: TEMİZLİK (DISK FULL ÖNLEMİ) ---
echo "1. Cleaning Docker system..."
# Kullanılmayan her şeyi sil (Volume'lar dahil)
docker system prune -af --volumes || true

# Eski durdurma bayraklarını temizle
echo "1b. Removing stale stop flags..."
sudo rm -f /tmp/bearish_bot_manual_stop.flag

# --- ADIM 2: (Optional) recreate is handled by vm_run_session.py ---
FORCE_RECREATE="__FORCE_RECREATE__"
TRADING_DURATION_SECONDS="__TRADING_DURATION_SECONDS__"
IMAGE_TAG="__IMAGE_TAG__"
DEBUG_MODE_STR="__DEBUG_MODE__"
LOG_LEVEL_STR="__LOG_LEVEL__"
BINGX_ENV="__BINGX_ENV__"

echo "2. Force recreate: $FORCE_RECREATE"

# --- ADIM 3: ENV AYARLARI ---
if [ -n "$TRADING_DURATION_SECONDS" ]; then
    echo "3. Updating duration in env file..."
    sudo sed -i "s/^#\\? *TRADING_DURATION=.*/TRADING_DURATION=$TRADING_DURATION_SECONDS/" /home/azureuser/bearish-bot.env
fi

# --- ADIM 3b: AZURE APP CONFIGURATION ENV VARS ---
echo "3b. Ensuring Azure App Configuration environment variables..."
ENV_FILE="/home/azureuser/bearish-bot.env"

# Ensure AZURE_APPCONFIG_ENDPOINT is set
if ! grep -q "^AZURE_APPCONFIG_ENDPOINT=" "$ENV_FILE"; then
    echo "   Adding AZURE_APPCONFIG_ENDPOINT..."
    echo "AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io" | sudo tee -a "$ENV_FILE" > /dev/null
fi

# Ensure AZURE_APPCONFIG_LABEL is set
if ! grep -q "^AZURE_APPCONFIG_LABEL=" "$ENV_FILE"; then
    echo "   Adding AZURE_APPCONFIG_LABEL..."
    echo "AZURE_APPCONFIG_LABEL=production" | sudo tee -a "$ENV_FILE" > /dev/null
fi

echo "   ✓ App Configuration environment variables configured"

# --- ADIM 4: BOTU BAŞLAT (PYTHON WRAPPER) ---
echo "4. Launching bot container..."
cd /home/azureuser

IMAGE="bearishalphabot.azurecr.io/bearish-bot:${IMAGE_TAG}"
NAME="bearish-bot"

# Prefer vm_run_session.py if it supports the new flags; otherwise fall back
# to a direct docker run with equivalent options.
HAS_NEW_WRAPPER=0
if sudo python3 vm_run_session.py --help 2>/dev/null | grep -q -- '--force-recreate'; then
    HAS_NEW_WRAPPER=1
fi

echo "   Wrapper supports new flags: $HAS_NEW_WRAPPER"

if [ "$HAS_NEW_WRAPPER" -eq 1 ]; then
    # vm_run_session.py container'ı --detach (arka plan) modunda başlatır.
    ARGS=(--image "$IMAGE" --name "$NAME" --force-recreate "$FORCE_RECREATE")
    # Prefer env overrides instead of editing env-file in place (safer for VST/PROD routing).
    if [ -n "$BINGX_ENV" ]; then
        ARGS+=(--env "BINGX_ENV=$BINGX_ENV")
    fi
    if [ -n "$DEBUG_MODE_STR" ]; then
        ARGS+=(--debug-mode "$DEBUG_MODE_STR")
    fi
    if [ -n "$LOG_LEVEL_STR" ]; then
        ARGS+=(--log-level "$LOG_LEVEL_STR")
    fi
    sudo python3 vm_run_session.py "${ARGS[@]}"
else
    echo "   ⚠️ vm_run_session.py is outdated; using direct docker fallback."
    echo "   forceRecreate=$FORCE_RECREATE debugMode=$DEBUG_MODE_STR logLevel=$LOG_LEVEL_STR"

    # Always pull the image first (same as wrapper).
    docker pull "$IMAGE"

    # Apply recreate semantics.
    if [ "$FORCE_RECREATE" = "true" ]; then
        docker stop "$NAME" || true
        docker rm "$NAME" || true
    else
        # If container exists but isn't running, remove it so docker run can succeed.
        if docker ps -a --filter "name=^${NAME}$" | grep -q "$NAME"; then
            if ! docker ps --filter "name=^${NAME}$" --filter "status=running" | grep -q "$NAME"; then
                docker rm "$NAME" || true
            fi
        fi
    fi

    # Build env overrides AFTER --env-file.
    EXTRA_ENV_ARGS=""
    if [ -n "$BINGX_ENV" ]; then
        EXTRA_ENV_ARGS="$EXTRA_ENV_ARGS -e BINGX_ENV=$BINGX_ENV"
    fi
    if [ -n "$DEBUG_MODE_STR" ]; then
        EXTRA_ENV_ARGS="$EXTRA_ENV_ARGS -e DEBUG_MODE=$DEBUG_MODE_STR"
    fi
    if [ -n "$LOG_LEVEL_STR" ]; then
        EXTRA_ENV_ARGS="$EXTRA_ENV_ARGS -e LOG_LEVEL=$LOG_LEVEL_STR"
    fi

    # Only start if force recreate OR container isn't already running.
    if [ "$FORCE_RECREATE" = "true" ] || ! docker ps --filter "name=^${NAME}$" --filter "status=running" | grep -q "$NAME"; then
        # shellcheck disable=SC2086
        docker run -d --name "$NAME" --env-file "$ENV_FILE" $EXTRA_ENV_ARGS \
            -v /mnt/bearish/logs:/app/logs \
            -v /mnt/bearish/data:/app/data \
            "$IMAGE"
    else
        echo "   ℹ️ Container already running; skipping start."
    fi
fi

# --- ADIM 5: SAĞLIK KONTROLÜ (HEALTH CHECK) ---
echo "5. Verifying startup health (10s wait)..."
sleep 10

# Container 'running' durumunda mı?
if docker ps --filter "name=^bearish-bot$" --filter "status=running" | grep -q "bearish-bot"; then
    echo "✅ SUCCESS: Bot container is UP and RUNNING."
    CID=$(docker ps --filter "name=^bearish-bot$" --format "{{.ID}}" | head -n 1)
    echo "   Container ID: $CID"
    echo "--- ENV OVERRIDE DIAGNOSTICS ---"
    docker inspect bearish-bot --format '{{range .Config.Env}}{{println .}}{{end}}' | egrep 'BINGX_ENV|DEBUG_MODE|LOG_LEVEL' || true
    docker exec bearish-bot env | egrep 'BINGX_ENV|DEBUG_MODE|LOG_LEVEL' || true
    LATEST=$(ls -t /mnt/bearish/logs/live_trading_*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST" ]; then
        grep -m 5 " - DEBUG - " "$LATEST" || echo "NO DEBUG lines"
        echo "--- ROUTING CONFIRMATION (best-effort) ---"
        echo "Expected BINGX_ENV=$BINGX_ENV"
        GREP_OK=1
        if ! grep -m 1 "\\[BINGX-ENV\\] env=$BINGX_ENV " "$LATEST" >/dev/null 2>&1; then
            echo "?? WARNING: Missing or mismatched [BINGX-ENV] line in log (expected env=$BINGX_ENV)."
            GREP_OK=0
        fi
        if ! grep -m 1 "\\[MODE-BANNER\\].*BINGX_ENV=$BINGX_ENV" "$LATEST" >/dev/null 2>&1; then
            echo "?? WARNING: Missing or mismatched [MODE-BANNER] line in log (expected BINGX_ENV=$BINGX_ENV)."
            GREP_OK=0
        fi
        if [ "$GREP_OK" -eq 0 ]; then
            echo "?? WARNING: Routing confirmation failed. Verify container logs with: docker logs --tail 200 bearish-bot"
        else
            echo "? Routing confirmation OK ([BINGX-ENV] and [MODE-BANNER] match)."
        fi
    else
        echo "NO log file found under /mnt/bearish/logs"
    fi
    exit 0
else
    echo "❌ CRITICAL FAILURE: Bot container died immediately!"
    echo "=== RECENT LOGS ==="
    docker logs --tail 20 bearish-bot
    exit 1
fi
'@

    # Substitute placeholders (keep bash $... intact)
    $startupScript = $startupScript.Replace('__FORCE_RECREATE__', $forceRecreateStr)
    $startupScript = $startupScript.Replace('__TRADING_DURATION_SECONDS__', $tradingDurationSeconds)
    $startupScript = $startupScript.Replace('__IMAGE_TAG__', $ImageTag)
    $startupScript = $startupScript.Replace('__DEBUG_MODE__', $debugModeStr)
    $startupScript = $startupScript.Replace('__LOG_LEVEL__', $logLevelStr)
    $startupScript = $startupScript.Replace('__BINGX_ENV__', $targetEnvNorm)

    # 4. EXECUTE ON VM
    Write-Output "[5/6] Sending command to VM..."
    
    $invokeParams = @{
        ResourceGroupName = $ResourceGroup
        VMName = $VMName
        CommandId = 'RunShellScript'
        ScriptString = $startupScript
    }
    
    # VM RunCommand can be busy for several minutes (e.g., another RunCommand still running).
    # Use a longer retry window to avoid failing the whole runbook on transient 409/Conflict.
    $maxAttempts = 15
    $attempt = 0
    $result = $null

    while ($attempt -lt $maxAttempts) {
        $attempt++
        try {
            if ($attempt -gt 1) {
                Write-Output "Retrying VM Run Command (attempt $attempt/$maxAttempts)..."
            }
            $result = Invoke-AzVMRunCommand @invokeParams
            break
        }
        catch {
            $msg = $_.Exception.Message
            # Common transient failure when the RunCommand extension is already executing.
            if ($msg -match "execution is in progress" -or $msg -match "Conflict") {
                if ($attempt -ge $maxAttempts) { throw }
                # Exponential backoff capped at 120s: 10, 20, 40, 80, 120, 120, ...
                $delaySeconds = [Math]::Min(120, [int](10 * [Math]::Pow(2, ($attempt - 1))))
                Write-Output "VM Run Command busy (Conflict). Waiting ${delaySeconds}s then retry..."
                Start-Sleep -Seconds $delaySeconds
                continue
            }
            throw
        }
    }
    
    # 5. ANALYZE OUTPUT
    Write-Output "[6/6] Analyzing result..."
    
    $output = ""
    if ($result.Value -and $result.Value[0].Message) { $output = $result.Value[0].Message }
    elseif ($result.Value[1] -and $result.Value[1].Message) { $output = $result.Value[1].Message }
    
    Write-Output ""
    Write-Output "--- VM OUTPUT ---"
    Write-Output $output
    Write-Output "-----------------"
    Write-Output ""

    # 6. FINAL VERDICT
    if ($output -match "SUCCESS: Bot container is UP and RUNNING") {
        Write-Output "╔════════════════════════════════════════════════════════╗"
        Write-Output "║             ✅ RUNBOOK COMPLETED SUCCESSFULLY          ║"
        Write-Output "╚════════════════════════════════════════════════════════╝"
        Write-Output "Bot is running in background."
        Write-Output "Logic App will now start polling the container status."
    } else {
        Write-Error "Bot failed to start properly. See VM Output above for logs."
    }

} catch {
    Write-Error "Runbook execution failed: $($_.Exception.Message)"
    throw $_
}
