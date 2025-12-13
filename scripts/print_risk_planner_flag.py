import os
import sys
from pathlib import Path

# Ensure src/ is on path for package-style imports
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config.live_trading_config import LiveTradingConfiguration
from config.risk_config import RiskConfiguration
from core.risk_manager import RiskManager


def main() -> None:
    env_value = os.getenv("RISK_SIZE_PLANNER_ENABLED")

    cfg = LiveTradingConfiguration.load()
    risk_cfg = RiskConfiguration(custom_limits=cfg.get('risk', {}))

    # Use the same resolution path as production: RiskManager computes the flag.
    rm = RiskManager(portfolio_value=risk_cfg.initial_capital, risk_config=risk_cfg, rules=[])
    resolved_mode = 'active' if rm._is_size_planner_enabled() else 'shadow'

    config_value = None
    try:
        config_value = (risk_cfg.to_dict().get('risk') or {}).get('size_planner_enabled')
    except Exception:
        config_value = None

    print(
        {
            "env_value": env_value,
            "config_value": config_value,
            "resolved_mode": resolved_mode,
        }
    )


if __name__ == "__main__":
    main()
