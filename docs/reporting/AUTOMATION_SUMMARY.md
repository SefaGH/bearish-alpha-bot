# Reporting Automation Summary

## Overview
Implemented automatic report generation and email delivery triggered by the bot's shutdown sequence.

## Components

### 1. Azure Function (`bearish-reporting-func`)
- **Functionality**: Generates a PDF report from ADX events and emails it via SendGrid.
- **Trigger**: HTTP POST to `/api/run-report` with `{"run_id": "..."}`.
- **Status**: Deployed and verified.

### 2. Bot Integration (`scripts/live_trading_launcher.py`)
- **Mechanism**: Added `_trigger_report()` method called during `cleanup()`.
- **Logic**:
  1.  Extracts `run_id` from the active log file (via `core.logger.CURRENT_LOG_FILE`).
  2.  Sends an asynchronous POST request to the Azure Function.
  3.  Logs the result (Success/Failure).

### 3. Logger (`src/core/logger.py`)
- **Enhancement**: Exposed `CURRENT_LOG_FILE` global variable to allow other components to identify the active session's log file.

## Workflow
1.  Bot starts -> `setup_logger` creates a log file (e.g., `live_trading_20251127_...log`).
2.  Bot runs trading logic.
3.  Bot shuts down (Graceful or Error).
4.  `cleanup()` is called.
5.  `_trigger_report()` is executed:
    - Gets `run_id` from log filename.
    - Calls Azure Function.
6.  Azure Function queries ADX for events with that `run_id`.
7.  PDF is generated and emailed to `sefaasar@hotmail.com`.

## Verification
- **Manual Trigger**: Verified via `Invoke-RestMethod`.
- **Automatic Trigger**: Will occur on next bot run/shutdown.
