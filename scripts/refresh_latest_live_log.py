import glob
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def refresh_latest_live_log(host_logs_dir: str | None = None) -> None:
    """Copy the most recent live_trading_*.log to a mounted host logs dir.

    This is primarily intended for Azure VM / Docker containers where
    the working directory is a bind mount of the repo (e.g.
    /mnt/c/Users/sefaa/bearish-alpha-bot), and logs/ is mounted back
    to the host.
    """

    cwd = Path.cwd()
    logs_pattern = str(cwd / "logs" / "live_trading_*.log")
    log_files = sorted(glob.glob(logs_pattern))

    if not log_files:
        logging.info("[REFRESH-LATEST-LOG] No live_trading_*.log found under %s", logs_pattern)
        return

    latest = Path(log_files[-1])
    logging.info("[REFRESH-LATEST-LOG] Latest live log detected: %s", latest)

    if host_logs_dir is None:
        host_logs_dir = os.environ.get("HOST_LOGS_DIR") or str(cwd / "logs")

    host_logs_path = Path(host_logs_dir)
    try:
        host_logs_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.error("[REFRESH-LATEST-LOG] Failed to ensure host logs dir %s: %s", host_logs_path, exc)
        return

    target = host_logs_path / latest.name
    try:
        data = latest.read_bytes()
        target.write_bytes(data)
        logging.info("[REFRESH-LATEST-LOG] Copied %s -> %s", latest, target)
    except Exception as exc:
        logging.error("[REFRESH-LATEST-LOG] Failed to copy %s -> %s: %s", latest, target, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    refresh_latest_live_log()
