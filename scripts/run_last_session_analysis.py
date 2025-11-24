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
    """

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
