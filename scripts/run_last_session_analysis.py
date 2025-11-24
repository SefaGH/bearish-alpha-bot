import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_last_session_analysis")


def main() -> int:
    """Run the standard log analyzer on the latest live_trading_*.log.

    This is a thin wrapper around diagnostics/log_analyzer_auto_plus.py so that
    operators can call a single, stable entrypoint after any session ends
    (whether it stopped via TRADING_DURATION, manual stop, or an error).

    VM/Azure-friendly enhancement: before running the analyzer, refresh the
    latest live_trading_*.log copy into the host-mounted logs directory
    (if such a mount exists), so operators can inspect the same file directly
    from the VM host without docker/volume gymnastics.
    """

    # Optional: keep the latest live log in a host-accessible logs/ dir
    try:
        from scripts.refresh_latest_live_log import refresh_latest_live_log  # type: ignore

        log.info("[RUN-LAST-SESSION] Refreshing latest live_trading log for host access...")
        refresh_latest_live_log()
    except Exception as refresh_error:  # pragma: no cover - best-effort helper
        log.warning("[RUN-LAST-SESSION] Failed to refresh latest live log: %s", refresh_error)

    analyzer = Path("diagnostics/log_analyzer_auto_plus.py")
    if not analyzer.exists():
        log.error("Analyzer script not found at %s", analyzer)
        return 1

    log.info("========================================")
    log.info("Running session analysis using %s", analyzer)
    log.info("Note: This always analyzes the latest live_trading_*.log in logs/.")
    log.info("It does not depend on how the bot stopped (duration/manual/error).")
    log.info("========================================")

    cmd = [sys.executable, str(analyzer)]
    result = subprocess.run(cmd)

    if result.returncode == 0:
        log.info("✅ Session analysis completed successfully")
    else:
        log.warning("⚠️ Session analysis finished with non-zero exit code %s", result.returncode)

    return result.returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
