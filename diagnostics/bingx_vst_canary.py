import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ccxt_client import CcxtClient  # noqa: E402
from core.execution_env import get_bingx_env, get_execution_backend, get_trading_mode  # noqa: E402
from core.order_manager import SmartOrderManager  # noqa: E402


logger = logging.getLogger("bingx_vst_canary")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_env_file_if_present(path: str) -> bool:
    env_path = REPO_ROOT / path
    if not env_path.exists():
        return False
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and os.getenv(key) is None:
                os.environ[key] = value
        return True
    except Exception:
        return False


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _resolve_symbol(exchange, requested_symbol: str) -> str:
    exchange.load_markets()
    if requested_symbol in exchange.markets:
        return requested_symbol
    variants = [
        requested_symbol.replace(":USDT", ""),
        "BTC/USDT:USDT",
        "BTC/USDT",
    ]
    for v in variants:
        if v in exchange.markets:
            return v
    raise RuntimeError(f"Symbol not found in markets: {requested_symbol}")


def _compute_amount(exchange, symbol: str, notional_usdt: float, min_notional_usdt: float) -> float:
    ticker = exchange.fetch_ticker(symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if last <= 0:
        raise RuntimeError("Failed to fetch last price for sizing")

    effective_notional = max(notional_usdt, min_notional_usdt or 0.0)
    raw_amount = effective_notional / last

    market = exchange.market(symbol)
    min_amount = _safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"))
    precision_amount = _safe_float(((market.get("precision") or {}).get("amount")))
    step_floor = precision_amount if precision_amount > 0 else 0.0
    hard_floor = max(min_amount, step_floor)

    raw_amount = max(raw_amount, hard_floor)
    try:
        rounded_str = exchange.amount_to_precision(symbol, raw_amount)
        amount = _safe_float(rounded_str)
    except Exception:
        amount = raw_amount

    if hard_floor and amount < hard_floor:
        amount = hard_floor

    return amount


def _sanitize_positions_snapshot(snapshot: Any) -> Any:
    if isinstance(snapshot, dict):
        data = snapshot.get("data")
        if isinstance(data, list):
            return {"code": snapshot.get("code"), "data_len": len(data)}
        return {"code": snapshot.get("code"), "data": data}
    return snapshot


async def _run_canary(args: argparse.Namespace) -> Dict[str, Any]:
    # Force safe canary routing (do not rely on ambient env defaults).
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    if get_bingx_env() != "vst":
        raise RuntimeError("This canary only supports VST. Set BINGX_ENV=vst.")

    if get_trading_mode() != "live" or get_execution_backend() != "ccxt":
        raise RuntimeError("Canary requires TRADING_MODE=live and EXECUTION_BACKEND=ccxt.")

    key = _require_env("BINGX_KEY")
    secret = _require_env("BINGX_SECRET")

    client = CcxtClient("bingx", {"apiKey": key, "secret": secret})
    exchange = client.ex

    symbol = _resolve_symbol(exchange, args.symbol)
    amount = _compute_amount(exchange, symbol, args.notional_usdt, args.min_notional_usdt)

    try:
        client.ensure_bingx_hedge_mode(symbol, require_hedged=True)
    except Exception as exc:
        raise RuntimeError(f"Hedge mode check failed: {exc}") from exc

    order_manager = SmartOrderManager(market_data_pipeline=None, risk_manager=None, exchange_clients={"bingx": client})

    open_side = "buy" if args.side == "long" else "sell"
    close_side = "sell" if open_side == "buy" else "buy"

    open_request = {
        "symbol": symbol,
        "side": open_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": False},
    }

    logger.info("🟢 [CANARY] Opening position: %s", {"symbol": symbol, "side": open_side, "amount": amount})
    open_result = await order_manager.place_order(open_request, execution_algo="market")
    if not open_result.get("success"):
        raise RuntimeError(f"Open order failed: {open_result.get('reason')}")

    open_order_id = open_result.get("order_id")
    time.sleep(1.0)

    positions_after_open_rest = client.get_bingx_positions(symbol)
    logger.info("🟢 [CANARY] Positions after open (REST): %s", _sanitize_positions_snapshot(positions_after_open_rest))

    close_request = {
        "symbol": symbol,
        "side": close_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": True},
    }

    logger.info("🟢 [CANARY] Closing position: %s", {"symbol": symbol, "side": close_side, "amount": amount})
    close_result = await order_manager.place_order(close_request, execution_algo="market")
    if not close_result.get("success"):
        logger.warning("🟠 [CANARY] Close with reduceOnly failed; retrying without reduceOnly: %s", close_result.get("reason"))
        close_request["execution_params"] = {"reduceOnly": False}
        close_result = await order_manager.place_order(close_request, execution_algo="market")
        if not close_result.get("success"):
            raise RuntimeError(f"Close order failed: {close_result.get('reason')}")

    close_order_id = close_result.get("order_id")
    time.sleep(1.0)

    positions_after_close_rest = client.get_bingx_positions(symbol)
    logger.info("🟢 [CANARY] Positions after close (REST): %s", _sanitize_positions_snapshot(positions_after_close_rest))

    # Defense-in-depth: cancel any open orders left behind
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for o in open_orders or []:
            oid = o.get("id")
            if oid:
                try:
                    client.cancel_order(oid, symbol)
                    logger.info("🟢 [CANARY] Canceled open order: %s", oid)
                except Exception as exc:
                    logger.warning("🟠 [CANARY] Cancel failed for %s: %s", oid, exc)
    except Exception as exc:
        logger.warning("🟠 [CANARY] fetch_open_orders failed: %s", exc)

    return {
        "ts_start": args.ts_start,
        "ts_end": _utc_now_iso(),
        "symbol": symbol,
        "env_loaded": bool(getattr(args, "env_loaded", False)),
        "env_file": getattr(args, "env_file", None),
        "bingx_env": get_bingx_env(),
        "execution_backend": get_execution_backend(),
        "trading_mode": get_trading_mode(),
        "ccxt_sandbox": bool(
            getattr(exchange, "sandbox", False)
            or ("open-api-vst" in str(((exchange.urls or {}).get("api") or {}).get("swap") or ""))
        ),
        "ccxt_api_swap_url": ((exchange.urls or {}).get("api") or {}).get("swap"),
        "rest_base_url": getattr(client, "_bingx_rest_base_url", None),
        "open_order_id": open_order_id,
        "close_order_id": close_order_id,
        "positions_after_open_rest": _sanitize_positions_snapshot(positions_after_open_rest),
        "positions_after_close_rest": _sanitize_positions_snapshot(positions_after_close_rest),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="BingX VST canary (swap): open + close + REST readback")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument("--min-notional-usdt", type=float, default=0.0)
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "diagnostics" / "vst"))
    ap.add_argument(
        "--env-file",
        default="bearish-bot.env.local",
        help=(
            "Optional env file to load (key=value lines). "
            "Defaults to bearish-bot.env.local, then falls back to .env.local and .env."
        ),
    )
    args = ap.parse_args()

    args.ts_start = _utc_now_iso()

    args.env_loaded = False
    if args.env_file:
        args.env_loaded = _load_env_file_if_present(args.env_file) or args.env_loaded
    args.env_loaded = _load_env_file_if_present(".env.local") or args.env_loaded
    args.env_loaded = _load_env_file_if_present(".env") or args.env_loaded

    # Avoid Windows console encoding issues with emoji-rich logs.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"bingx_vst_canary_{ts}.log"
    summary_path = out_dir / f"bingx_vst_canary_summary_{ts}.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )

    try:
        summary = asyncio.run(_run_canary(args))
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        # Convenience copies (stable filenames) for quick inspection / PR evidence.
        latest_summary_path = out_dir / f"bingx_vst_canary_summary_latest_{args.side}.json"
        latest_log_path = out_dir / f"bingx_vst_canary_latest_{args.side}.log"
        latest_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        latest_log_path.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("Wrote: %s", summary_path)
        logger.info("Wrote: %s", latest_summary_path)
        logger.info("Wrote: %s", log_path)
        logger.info("Wrote: %s", latest_log_path)
        return 0
    except Exception as exc:
        logger.error("CANARY FAILED: %s", exc, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
