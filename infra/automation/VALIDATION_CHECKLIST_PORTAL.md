# Validation Checklist (Portal): Logic App → Automation → VM → Docker env

This checklist verifies that `debugMode` / `logLevel` from the Logic App trigger payload makes it all the way into the running Docker container env.

## What “good” looks like
- Logic App run shows `Start_Automation_Runbook` succeeded.
- Azure Automation job parameters include `debugMode` and `logLevel`.
- Inside the running container:
  - `DEBUG_MODE=true|false`
  - `LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL`

## 1) Logic App (Request trigger)
1. Azure Portal → **Logic Apps** → open workflow **bearish-bot-orchestrator**.
2. Go to **Runs history** → open the latest run.
3. Open the trigger (usually **manual** / Request trigger):
   - Confirm the trigger input body includes the keys you sent:
     - `debugMode` (boolean or string)
     - `logLevel` (string)

### 1a) Confirm the mapping to Automation job parameters
Inside the same run:
1. Open action **Start_Automation_Runbook**.
2. Expand **Inputs**.
3. Under `body → properties → parameters`, confirm it contains:
   - `durationMinutes`, `forceRestart`, `imageTag`, `idempotencyToken`
   - `debugMode` (only if you provided it)
   - `logLevel` (only if you provided it)

If `debugMode/logLevel` are missing here, the issue is **still in Logic App mapping**, not in the runbook/VM.

## 2) Azure Automation (job parameters)
1. Azure Portal → **Automation Accounts** → open **tradebot-automation**.
2. Go to **Jobs**.
3. Open the job created at the same time as your Logic App run.
4. Confirm:
   - **Runbook**: `Start-BearishBot-Enhanced`
   - **Status**: `Completed` (or `Running` while in progress)
   - **Parameters** includes:
     - `debugMode`: `True`/`False` (may be stringified)
     - `logLevel`: `DEBUG`/`INFO`/etc

If the job has `debugMode/logLevel` but the container doesn’t, the issue is **downstream on the VM/Docker side**.

### 2a) Known transient failure mode: VM RunCommand conflict (409)
If job fails with an error like:
- “Run command extension execution is in progress”
- “Conflict (409)”

This usually means another **VM RunCommand** is already running. Waiting a bit and retrying the Logic App run is often enough.

## 3) VM + Docker (verify env inside container)
Goal: confirm env vars in the running container.

### Option A (recommended): connect to VM and run docker commands
1. Azure Portal → **Virtual machines** → open **BearishAlphaBot-VM-01**.
2. Connect via **Bastion** or **SSH**.
3. Run:
   - `docker ps --filter name=bearish-bot`
   - `docker exec bearish-bot /bin/sh -c "echo DEBUG_MODE=$DEBUG_MODE; echo LOG_LEVEL=$LOG_LEVEL"`

Expected output should match your payload.

### Option B: VM “Run command” (works, but can conflict)
1. Azure Portal → VM **BearishAlphaBot-VM-01** → **Run command**.
2. Use **RunShellScript**.
3. Script:
   - `docker exec bearish-bot /bin/sh -c 'echo DEBUG_MODE=$DEBUG_MODE; echo LOG_LEVEL=$LOG_LEVEL'`

If you get an “execution is in progress” conflict, use Option A or wait until current RunCommand completes.

## Quick troubleshooting guide
- **Logic App run fails early** with template/expression errors:
  - Fix is in the Logic App definition (action expressions/functions).
- **Logic App “Start_Automation_Runbook” succeeded but Automation job has no debug/log parameters**:
  - Fix is in Logic App mapping into `properties.parameters`.
- **Automation job has parameters but container env still defaults**:
  - Confirm container was recreated when overrides changed.
  - Confirm Docker precedence: `--env-file` first, then `-e DEBUG_MODE=... -e LOG_LEVEL=...`.
- **Container not running**:
  - Check Automation job output + VM docker logs.

## References (Microsoft docs)
- Azure Logic Apps run history: https://learn.microsoft.com/azure/logic-apps/logic-apps-monitor
- Azure Automation jobs: https://learn.microsoft.com/azure/automation/automation-runbook-execution
- VM Run Command: https://learn.microsoft.com/azure/virtual-machines/run-command
