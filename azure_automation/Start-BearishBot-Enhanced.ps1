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
    [string] $TargetEnv = "",

    # Key Vault integration (VM Managed Identity should have secrets/get permission).
    # When provided, the VM RunCommand script will fetch BingX credentials from Key Vault
    # and inject them into a temporary runtime env-file for docker run.
    [Parameter(Mandatory=$false)]
    [string] $KeyVaultName = "",

    [Parameter(Mandatory=$false)]
    [string] $BingxKeySecretName = "",

    [Parameter(Mandatory=$false)]
    [string] $BingxSecretSecretName = "",

    # Optional Telegram credentials from Key Vault (recommended; avoids storing bot token in env-file)
    [Parameter(Mandatory=$false)]
    [string] $TelegramBotTokenSecretName = "",

    [Parameter(Mandatory=$false)]
    [string] $TelegramChatIdSecretName = ""
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
KEYVAULT_NAME="__KEYVAULT_NAME__"
BINGX_KEY_SECRET_NAME="__BINGX_KEY_SECRET_NAME__"
BINGX_SECRET_SECRET_NAME="__BINGX_SECRET_SECRET_NAME__"
TELEGRAM_BOT_TOKEN_SECRET_NAME="__TELEGRAM_BOT_TOKEN_SECRET_NAME__"
TELEGRAM_CHAT_ID_SECRET_NAME="__TELEGRAM_CHAT_ID_SECRET_NAME__"
KV_TOKEN_ID_LOGGED=0

echo "2. Force recreate: $FORCE_RECREATE"

# --- ADIM 3: ENV AYARLARI ---
if [ -n "$TRADING_DURATION_SECONDS" ]; then
    echo "3. Updating duration in env file..."
    sudo sed -i "s/^#\\? *TRADING_DURATION=.*/TRADING_DURATION=$TRADING_DURATION_SECONDS/" /home/azureuser/bearish-bot.env
fi

# --- ADIM 3b: AZURE APP CONFIGURATION ENV VARS ---
echo "3b. Ensuring Azure App Configuration environment variables..."
ENV_FILE_BASE="/home/azureuser/bearish-bot.env"

# Ensure AZURE_APPCONFIG_ENDPOINT is set
if ! grep -q "^AZURE_APPCONFIG_ENDPOINT=" "$ENV_FILE_BASE"; then
    echo "   Adding AZURE_APPCONFIG_ENDPOINT..."
    echo "AZURE_APPCONFIG_ENDPOINT=https://appcs-bearish-bot.azconfig.io" | sudo tee -a "$ENV_FILE_BASE" > /dev/null
fi

# Ensure AZURE_APPCONFIG_LABEL is set
if ! grep -q "^AZURE_APPCONFIG_LABEL=" "$ENV_FILE_BASE"; then
    echo "   Adding AZURE_APPCONFIG_LABEL..."
    echo "AZURE_APPCONFIG_LABEL=production" | sudo tee -a "$ENV_FILE_BASE" > /dev/null
fi

echo "   ✓ App Configuration environment variables configured"

# --- ADIM 3c: KEY VAULT (BingX credentials) -> runtime env file (no secrets at rest) ---
echo "3c. Preparing runtime env-file..."
ENV_FILE="/tmp/bearish-bot.env.runtime"
cp "$ENV_FILE_BASE" "$ENV_FILE"
chmod 600 "$ENV_FILE" || true

cleanup_env_file() {
    rm -f "$ENV_FILE" || true
}
trap cleanup_env_file EXIT

read_env_value() {
    local key="$1"
    local file="$2"
    # shellcheck disable=SC2002
    cat "$file" 2>/dev/null | grep -E "^${key}=" | tail -n 1 | cut -d= -f2- || true
}

log_kv_token_identity() {
    if [ "${KV_TOKEN_ID_LOGGED:-0}" = "1" ]; then
        return 0
    fi
    if [ -z "${ACCESS_TOKEN:-}" ]; then
        return 0
    fi
    ident="$(
        python3 - <<'PY' 2>/dev/null
import os, json, base64
token = os.environ.get("ACCESS_TOKEN", "")
parts = token.split(".")
if len(parts) < 2:
    raise SystemExit(0)
payload = parts[1].replace("-", "+").replace("_", "/")
payload += "=" * (-len(payload) % 4)
data = json.loads(base64.b64decode(payload.encode("utf-8")).decode("utf-8"))
appid = data.get("appid") or data.get("azp") or ""
oid = data.get("oid") or ""
tid = data.get("tid") or ""
iss = data.get("iss") or ""
print(f"appid={appid} oid={oid} tid={tid} iss={iss}")
PY
    )"
    if [ -n "${ident:-}" ]; then
        echo "   KeyVault token identity: ${ident}"
        KV_TOKEN_ID_LOGGED=1
    fi
}

needs_bingx_secrets=0
CUR_BINGX_KEY="$(read_env_value "BINGX_KEY" "$ENV_FILE")"
CUR_BINGX_SECRET="$(read_env_value "BINGX_SECRET" "$ENV_FILE")"
if [ -z "${CUR_BINGX_KEY:-}" ] || [ "${CUR_BINGX_KEY:-}" = "CHANGEME" ]; then
    needs_bingx_secrets=1
fi
if [ -z "${CUR_BINGX_SECRET:-}" ] || [ "${CUR_BINGX_SECRET:-}" = "CHANGEME" ]; then
    needs_bingx_secrets=1
fi

if [ "$needs_bingx_secrets" -eq 1 ]; then
    # Allow names to be supplied via Runbook params OR persisted in the base env-file.
    if [ -z "${KEYVAULT_NAME:-}" ] || [ "${KEYVAULT_NAME:-}" = "__KEYVAULT_NAME__" ]; then
        KEYVAULT_NAME="$(read_env_value "KEYVAULT_NAME" "$ENV_FILE_BASE")"
    fi
    if [ -z "${BINGX_KEY_SECRET_NAME:-}" ] || [ "${BINGX_KEY_SECRET_NAME:-}" = "__BINGX_KEY_SECRET_NAME__" ]; then
        BINGX_KEY_SECRET_NAME="$(read_env_value "BINGX_KEY_SECRET_NAME" "$ENV_FILE_BASE")"
    fi
    if [ -z "${BINGX_SECRET_SECRET_NAME:-}" ] || [ "${BINGX_SECRET_SECRET_NAME:-}" = "__BINGX_SECRET_SECRET_NAME__" ]; then
        BINGX_SECRET_SECRET_NAME="$(read_env_value "BINGX_SECRET_SECRET_NAME" "$ENV_FILE_BASE")"
    fi

    if [ -z "${KEYVAULT_NAME:-}" ] || [ -z "${BINGX_KEY_SECRET_NAME:-}" ] || [ -z "${BINGX_SECRET_SECRET_NAME:-}" ]; then
        echo "❌ Missing Key Vault settings for BingX credentials."
        echo "   Provide KEYVAULT_NAME + BINGX_KEY_SECRET_NAME + BINGX_SECRET_SECRET_NAME (either as Runbook params or in $ENV_FILE_BASE)."
        echo "   Or set BINGX_KEY/BINGX_SECRET directly (not recommended)."
        exit 1
    fi

    echo "   Fetching BingX credentials from Key Vault (managed identity; values not logged)..."
    echo "   KeyVault=${KEYVAULT_NAME} secrets: BINGX_KEY_SECRET_NAME=${BINGX_KEY_SECRET_NAME} BINGX_SECRET_SECRET_NAME=${BINGX_SECRET_SECRET_NAME}"

    ACCESS_TOKEN="$(
        curl -sS -H Metadata:true \
          "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))'
    )"

    if [ -z "${ACCESS_TOKEN:-}" ]; then
        echo "❌ Failed to obtain managed identity token for Key Vault."
        echo "   Ensure the VM has a Managed Identity and it has secrets/get permission on the Key Vault."
        exit 1
    fi
    log_kv_token_identity

    fetch_kv_secret_value() {
        local secret_name="$1"
        local url="https://${KEYVAULT_NAME}.vault.azure.net/secrets/${secret_name}?api-version=7.4"
        local resp http body
        resp="$(curl -sS -H "Authorization: Bearer ${ACCESS_TOKEN}" -w $'\\n%{http_code}' "$url" || true)"
        http="$(printf '%s' "$resp" | tail -n 1 | tr -d '\r')"
        body="$(printf '%s' "$resp" | sed '$d')"
        if [ "$http" != "200" ]; then
            err="$(
                printf '%s' "$body" | python3 -c 'import json,sys; 
try: data=json.load(sys.stdin)
except Exception: print("non-json response"); sys.exit(0)
e=(data.get("error") or {})
code=str(e.get("code") or "")
msg=str(e.get("message") or "")
out=(code + (" " if code and msg else "") + msg).strip()
print(out[:220])' 2>/dev/null
            )"
            echo "❌ Key Vault secret fetch failed: name=${secret_name} status=${http} error=${err}" >&2
            printf '%s' ""
            return 0
        fi
        printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("value") or "").strip())' 2>/dev/null
    }

    NEW_BINGX_KEY="$(fetch_kv_secret_value "$BINGX_KEY_SECRET_NAME")"
    NEW_BINGX_SECRET="$(fetch_kv_secret_value "$BINGX_SECRET_SECRET_NAME")"

    if [ -z "${NEW_BINGX_KEY:-}" ] || [ -z "${NEW_BINGX_SECRET:-}" ]; then
        echo "❌ Failed to read BingX secrets from Key Vault (empty value)."
        echo "   Most common causes: wrong secret names, VM Managed Identity lacks secrets/get, or Key Vault firewall/private endpoint blocks the VM."
        exit 1
    fi

    # Update runtime env-file safely (avoid sed escaping issues; do not print values).
    export ENV_FILE NEW_BINGX_KEY NEW_BINGX_SECRET
    python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ENV_FILE"])
updates = {
    "BINGX_KEY": os.environ["NEW_BINGX_KEY"],
    "BINGX_SECRET": os.environ["NEW_BINGX_SECRET"],
}

lines = path.read_text(encoding="utf-8").splitlines()
out = []
seen = set()
for line in lines:
    if not line or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    key, _ = line.split("=", 1)
    if key in updates:
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
fi

# --- Telegram credentials (optional) ---
needs_telegram_secrets=0
CUR_TG_TOKEN="$(read_env_value "TELEGRAM_BOT_TOKEN" "$ENV_FILE")"
CUR_TG_CHAT_ID="$(read_env_value "TELEGRAM_CHAT_ID" "$ENV_FILE")"
if [ -z "${CUR_TG_TOKEN:-}" ] || [ "${CUR_TG_TOKEN:-}" = "CHANGEME" ]; then
    needs_telegram_secrets=1
fi
if [ -z "${CUR_TG_CHAT_ID:-}" ] || [ "${CUR_TG_CHAT_ID:-}" = "CHANGEME" ]; then
    needs_telegram_secrets=1
fi

if [ "$needs_telegram_secrets" -eq 1 ]; then
    if [ -z "${KEYVAULT_NAME:-}" ] || [ "${KEYVAULT_NAME:-}" = "__KEYVAULT_NAME__" ]; then
        KEYVAULT_NAME="$(read_env_value "KEYVAULT_NAME" "$ENV_FILE_BASE")"
    fi
    if [ -z "${TELEGRAM_BOT_TOKEN_SECRET_NAME:-}" ] || [ "${TELEGRAM_BOT_TOKEN_SECRET_NAME:-}" = "__TELEGRAM_BOT_TOKEN_SECRET_NAME__" ]; then
        TELEGRAM_BOT_TOKEN_SECRET_NAME="$(read_env_value "TELEGRAM_BOT_TOKEN_SECRET_NAME" "$ENV_FILE_BASE")"
    fi
    if [ -z "${TELEGRAM_CHAT_ID_SECRET_NAME:-}" ] || [ "${TELEGRAM_CHAT_ID_SECRET_NAME:-}" = "__TELEGRAM_CHAT_ID_SECRET_NAME__" ]; then
        TELEGRAM_CHAT_ID_SECRET_NAME="$(read_env_value "TELEGRAM_CHAT_ID_SECRET_NAME" "$ENV_FILE_BASE")"
    fi

    if [ -z "${KEYVAULT_NAME:-}" ] || [ -z "${TELEGRAM_BOT_TOKEN_SECRET_NAME:-}" ] || [ -z "${TELEGRAM_CHAT_ID_SECRET_NAME:-}" ]; then
        echo "❌ Missing Key Vault settings for Telegram."
        echo "   Provide KEYVAULT_NAME + TELEGRAM_BOT_TOKEN_SECRET_NAME + TELEGRAM_CHAT_ID_SECRET_NAME (either as Runbook params or in $ENV_FILE_BASE)."
        exit 1
    fi

    echo "   Fetching Telegram credentials from Key Vault (managed identity; values not logged)..."
    echo "   KeyVault=${KEYVAULT_NAME} secrets: TELEGRAM_BOT_TOKEN_SECRET_NAME=${TELEGRAM_BOT_TOKEN_SECRET_NAME} TELEGRAM_CHAT_ID_SECRET_NAME=${TELEGRAM_CHAT_ID_SECRET_NAME}"

    if [ -z "${ACCESS_TOKEN:-}" ]; then
        ACCESS_TOKEN="$(
            curl -sS -H Metadata:true \
              "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fvault.azure.net" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin).get("access_token",""))'
        )"
        if [ -z "${ACCESS_TOKEN:-}" ]; then
            echo "❌ Failed to obtain managed identity token for Key Vault."
            echo "   Ensure the VM has a Managed Identity and it has secrets/get permission on the Key Vault."
            exit 1
        fi
        log_kv_token_identity
    fi

    # Ensure helper exists even when BingX branch was skipped.
    fetch_kv_secret_value() {
        local secret_name="$1"
        local url="https://${KEYVAULT_NAME}.vault.azure.net/secrets/${secret_name}?api-version=7.4"
        local resp http body
        resp="$(curl -sS -H "Authorization: Bearer ${ACCESS_TOKEN}" -w $'\\n%{http_code}' "$url" || true)"
        http="$(printf '%s' "$resp" | tail -n 1 | tr -d '\r')"
        body="$(printf '%s' "$resp" | sed '$d')"
        if [ "$http" != "200" ]; then
            err="$(
                printf '%s' "$body" | python3 -c 'import json,sys; 
try: data=json.load(sys.stdin)
except Exception: print("non-json response"); sys.exit(0)
e=(data.get("error") or {})
code=str(e.get("code") or "")
msg=str(e.get("message") or "")
out=(code + (" " if code and msg else "") + msg).strip()
print(out[:220])' 2>/dev/null
            )"
            echo "❌ Key Vault secret fetch failed: name=${secret_name} status=${http} error=${err}" >&2
            printf '%s' ""
            return 0
        fi
        printf '%s' "$body" | python3 -c 'import json,sys; print((json.load(sys.stdin).get("value") or "").strip())' 2>/dev/null
    }

    NEW_TG_TOKEN="$(fetch_kv_secret_value "$TELEGRAM_BOT_TOKEN_SECRET_NAME")"
    NEW_TG_CHAT_ID="$(fetch_kv_secret_value "$TELEGRAM_CHAT_ID_SECRET_NAME")"

    if [ -z "${NEW_TG_TOKEN:-}" ] || [ -z "${NEW_TG_CHAT_ID:-}" ]; then
        echo "❌ Failed to read Telegram secrets from Key Vault (empty value)."
        echo "   Check secret names and Key Vault permissions for the VM identity."
        exit 1
    fi

    export ENV_FILE NEW_TG_TOKEN NEW_TG_CHAT_ID
    python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ENV_FILE"])
updates = {
    "TELEGRAM_BOT_TOKEN": os.environ["NEW_TG_TOKEN"],
    "TELEGRAM_CHAT_ID": os.environ["NEW_TG_CHAT_ID"],
}

lines = path.read_text(encoding="utf-8").splitlines()
out = []
seen = set()
for line in lines:
    if not line or line.lstrip().startswith("#") or "=" not in line:
        out.append(line)
        continue
    key, _ = line.split("=", 1)
    if key in updates:
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
fi

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
    ARGS=(--image "$IMAGE" --name "$NAME" --env-file "$ENV_FILE" --force-recreate "$FORCE_RECREATE")
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
    echo "--- TELEGRAM DIAGNOSTICS (no secrets) ---"
    # Do not print TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID values; only confirm presence and length.
    docker exec bearish-bot sh -lc 'echo "TELEGRAM_STARTUP_PING=${TELEGRAM_STARTUP_PING:-unset}"; echo "TELEGRAM_BOT_TOKEN_LEN=${#TELEGRAM_BOT_TOKEN}"; echo "TELEGRAM_CHAT_ID_SET=${TELEGRAM_CHAT_ID:+set}"' || true
    LATEST=$(ls -t /mnt/bearish/logs/live_trading_*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST" ]; then
        grep -m 5 " - DEBUG - " "$LATEST" || echo "NO DEBUG lines"
        echo "--- ROUTING CONFIRMATION (best-effort) ---"
        echo "Expected BINGX_ENV=$BINGX_ENV"

        check_routing_in_file() {
            local file="$1"
            [ -n "$file" ] || return 1
            grep -m 1 "\\[BINGX-ENV\\].*env=${BINGX_ENV}\\b" "$file" >/dev/null 2>&1 || return 1
            grep -m 1 "\\[MODE-BANNER\\].*BINGX_ENV=${BINGX_ENV}\\b" "$file" >/dev/null 2>&1 || return 1
            return 0
        }

        check_routing_in_docker_logs() {
            docker logs --tail 400 bearish-bot 2>/dev/null | grep -m 1 "\\[BINGX-ENV\\].*env=${BINGX_ENV}\\b" >/dev/null 2>&1 || return 1
            docker logs --tail 400 bearish-bot 2>/dev/null | grep -m 1 "\\[MODE-BANNER\\].*BINGX_ENV=${BINGX_ENV}\\b" >/dev/null 2>&1 || return 1
            return 0
        }

        # The bot may need >10s to reach exchange init / core init; retry briefly to reduce false warnings.
        GREP_OK=0
        for attempt in $(seq 1 6); do
            LATEST=$(ls -t /mnt/bearish/logs/live_trading_*.log 2>/dev/null | head -n 1 || true)
            echo "Routing check attempt ${attempt}/6 (log file: ${LATEST:-<none>})"

            if check_routing_in_file "$LATEST"; then
                echo "? Routing confirmation OK (file)"
                grep -m 1 "\\[BINGX-ENV\\].*env=${BINGX_ENV}\\b" "$LATEST" || true
                grep -m 1 "\\[MODE-BANNER\\].*BINGX_ENV=${BINGX_ENV}\\b" "$LATEST" || true
                GREP_OK=1
                break
            fi

            if check_routing_in_docker_logs; then
                echo "? Routing confirmation OK (docker logs)"
                docker logs --tail 400 bearish-bot 2>/dev/null | grep -m 1 "\\[BINGX-ENV\\].*env=${BINGX_ENV}\\b" || true
                docker logs --tail 400 bearish-bot 2>/dev/null | grep -m 1 "\\[MODE-BANNER\\].*BINGX_ENV=${BINGX_ENV}\\b" || true
                GREP_OK=1
                break
            fi

            sleep 5
        done

        if [ "$GREP_OK" -eq 0 ]; then
            echo "?? WARNING: Routing confirmation failed after retries."
            echo "   Expected env=$BINGX_ENV but did not find [BINGX-ENV] / [MODE-BANNER] in file or docker logs yet."
            echo "   Next steps (safe literal grep; avoids regex escaping issues):"
            echo "     docker logs --since 30m bearish-bot | grep -F '[MODE-BANNER]' -m 5 || true"
            echo "     docker logs --since 30m bearish-bot | grep -F '[BINGX-ENV]' -m 5 || true"
            echo "   Or check the file log (most reliable when logs are file-based):"
            echo "     LATEST=\$(ls -t /mnt/bearish/logs/live_trading_*.log 2>/dev/null | head -n 1 || true); \\"
            echo "       [ -n \"\$LATEST\" ] && (grep -F '[MODE-BANNER]' \"\$LATEST\" -m 5 || true; grep -F '[BINGX-ENV]' \"\$LATEST\" -m 5 || true)"
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
    $startupScript = $startupScript.Replace('__KEYVAULT_NAME__', $KeyVaultName)
    $startupScript = $startupScript.Replace('__BINGX_KEY_SECRET_NAME__', $BingxKeySecretName)
    $startupScript = $startupScript.Replace('__BINGX_SECRET_SECRET_NAME__', $BingxSecretSecretName)
    $startupScript = $startupScript.Replace('__TELEGRAM_BOT_TOKEN_SECRET_NAME__', $TelegramBotTokenSecretName)
    $startupScript = $startupScript.Replace('__TELEGRAM_CHAT_ID_SECRET_NAME__', $TelegramChatIdSecretName)

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
