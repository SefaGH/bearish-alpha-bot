import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.safety.signal_integrity_guard import SignalIntegrityGuard
from src.core.transition_policy import PositionTransitionPolicy


class DummyMarketDataPipeline:
    async def get_latest_price(self, symbol: str, timeframe: str = "1m") -> float:
        return 100.0


def _fmt(result: dict) -> str:
    return f"valid={result.get('valid')} action={result.get('action')} reason={result.get('reason')}"


async def run_stale_candle_test() -> None:
    print("\n[TEST] Stale Candle Guard")
    config = {
        "signals": {
            "integrity_guard": {
                "enabled": True,
                "max_staleness_ms": 10_000,
                "max_deviation_pct": 0.001,
            }
        }
    }
    guard = SignalIntegrityGuard(config, DummyMarketDataPipeline())

    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=90)
    signal = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "5m",
        "entry": 100.0,
        "timestamp": int(stale_ts.timestamp() * 1000),
        "meta": {
            "price_meta": {
                "price_used": 100.0,
                "candle_close_ts": stale_ts,
            }
        },
    }

    result = await guard.validate(signal, current_position=None)
    print("Result:", _fmt(result))
    if result.get("action") == "reject" and "stale" in str(result.get("reason", "")):
        print("✅ PASS: stale signal rejected")
    else:
        print("❌ FAIL: stale signal not rejected")


async def run_whipsaw_test() -> None:
    print("\n[TEST] Transition Policy (Whipsaw Prevention)")
    config = {
        "signals": {
            "transition_policy": {
                "enabled": True,
            }
        }
    }
    policy = PositionTransitionPolicy(config)

    current_position = {
        "strategy": "adaptive_ob",
        "side": "long",
        "unrealized_pnl_pct": 0.1,
    }
    incoming_signal = {
        "strategy": "mean_reversion",
        "strategy_name": "mean_reversion",
        "side": "short",
        "meta": {},
    }

    result = policy.evaluate(current_position, incoming_signal, inferred_intent="reverse")
    print("Result:", result)
    if result.get("action") == "convert_to_close":
        print("✅ PASS: reverse converted to close (no flip)")
    else:
        print("❌ FAIL: reverse not converted")


async def main() -> None:
    await run_stale_candle_test()
    await run_whipsaw_test()


if __name__ == "__main__":
    asyncio.run(main())
