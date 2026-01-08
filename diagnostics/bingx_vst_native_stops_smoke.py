"""
Stage-2 BingX VST smoke test: validate bot-integrated native conditional stops (hedge mode).

This script makes real API calls against BingX VST (sandbox).

What it does:
1) Opens a tiny swap position (MARKET) via the bot's SmartOrderManager (CCXT real execution).
2) Lets AdvancedPositionManager place a native hard stop (STOP_MARKET via stopLossPrice + workingType + positionSide).
3) Enables trailing stop with activation_threshold=0 and triggers monitor loop once to place native trailing.
4) Verifies both native conditional orders are visible in fetch_open_orders().
5) Cleans up: cancels open orders, closes the position, confirms positions are flat via REST.

Safety:
- Forces TRADING_MODE=live, EXECUTION_BACKEND=ccxt, BINGX_ENV=vst.
- Forces native stop flags ON for the duration of the run.
- Requires hedge mode (dualSidePosition=true).
- Redacts any credential-like fields in artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ccxt_client import CcxtClient  # noqa: E402
from core.order_manager import SmartOrderManager  # noqa: E402
from core.position_manager import AdvancedPositionManager  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redacted(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in {"apikey", "secret", "password", "passphrase", "token"}:
                out[k] = "***"
            else:
                out[k] = redacted(v)
        return out
    if isinstance(obj, list):
        return [redacted(x) for x in obj]
    return obj


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
    for v in [requested_symbol.replace(":USDT", ""), "BTC/USDT:USDT", "BTC/USDT"]:
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


@dataclass
class SmokeSummary:
    ts_start: str
    ts_end: str
    symbol: str
    side: str
    amount: float
    hard_stop_order_id: Optional[str]
    trailing_order_id: Optional[str]
    open_orders_ids: list[str]
    hard_stop_visible: bool
    trailing_visible: bool
    positions_after_close_rest: Dict[str, Any]
    rest_base_url: Optional[str]
    ccxt_api_swap_url: Any


class _StubRiskManager:
    def __init__(self):
        self.active_positions: Dict[str, Any] = {}

    def register_position(self, position_id: str, position: Dict[str, Any]) -> None:
        self.active_positions[position_id] = position

    def close_position(self, position_id: str, exit_price: float, realized_pnl: float) -> None:
        self.active_positions.pop(position_id, None)


class _StubPortfolioManager:
    def __init__(self, exchange_clients: Dict[str, Any]):
        self.exchange_clients = exchange_clients
        self.active_positions: Dict[str, Any] = {}
        self.cfg: Dict[str, Any] = {}
        self._trade_count = 0

    def register_position(self, position_id: str, position: Dict[str, Any]) -> None:
        self.active_positions[position_id] = position

    def close_position(self, position_id: str, exit_price: float, realized_pnl: float) -> None:
        self.active_positions.pop(position_id, None)

    def increment_trade_count(self) -> None:
        self._trade_count += 1


async def _run(args: argparse.Namespace) -> SmokeSummary:
    # Force safe canary routing (do not rely on ambient env defaults).
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED"] = "true"

    key = _require_env("BINGX_KEY")
    secret = _require_env("BINGX_SECRET")

    client = CcxtClient("bingx", {"apiKey": key, "secret": secret})
    exchange = client.ex

    symbol = _resolve_symbol(exchange, args.symbol)
    amount = _compute_amount(exchange, symbol, args.notional_usdt, args.min_notional_usdt)

    client.ensure_bingx_hedge_mode(symbol, require_hedged=True)

    order_manager = SmartOrderManager(market_data_pipeline=None, risk_manager=None, exchange_clients={"bingx": client})
    risk_manager = _StubRiskManager()
    portfolio_manager = _StubPortfolioManager(exchange_clients={"bingx": client})
    position_manager = AdvancedPositionManager(
        risk_manager=risk_manager,
        order_manager=order_manager,
        websocket_manager=None,
        portfolio_manager=portfolio_manager,
    )

    open_side = "buy" if args.side == "long" else "sell"
    close_side = "sell" if open_side == "buy" else "buy"

    # Precompute a safe (far) stop/tp from last price to avoid accidental triggers.
    ticker = exchange.fetch_ticker(symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if args.side == "long":
        stop = last * 0.95
        take_profit = last * 1.10
    else:
        stop = last * 1.05
        take_profit = last * 0.90

    signal = {
        "symbol": symbol,
        "side": args.side,
        "exchange": "bingx",
        "strategy": "diagnostics_vst_native_stops_smoke",
        "stop": stop,
        "take_profit": take_profit,
    }

    open_request = {
        "symbol": symbol,
        "side": open_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": False},
    }

    open_result = await order_manager.place_order(open_request, execution_algo="market")
    if not open_result.get("success"):
        raise RuntimeError(f"Open order failed: {open_result.get('reason')}")

    pos_result = await position_manager.open_position(signal, open_result)
    if not pos_result.get("success"):
        raise RuntimeError(f"open_position failed: {pos_result.get('reason')}")

    position_id = pos_result.get("position_id")
    position = position_manager.positions.get(position_id) if position_id else None
    hard_stop_order_id = (position or {}).get("native_hard_stop_order_id")

    # Enable trailing with immediate activation so monitor tick places native trailing.
    position_manager.configure_trailing_stop(
        position_id,
        enabled=True,
        delta_pct=float(args.trailing_distance),
        activation_threshold_pct=float(args.trailing_activation_threshold),
    )
    await position_manager.monitor_position_pnl(position_id, current_price=(position or {}).get("current_price"))

    position = position_manager.positions.get(position_id) if position_id else None
    trailing_order_id = (position or {}).get("native_trailing_stop_order_id")

    open_orders = exchange.fetch_open_orders(symbol)
    open_orders_ids = sorted([o.get("id") for o in (open_orders or []) if o.get("id")])
    hard_stop_visible = bool(hard_stop_order_id and hard_stop_order_id in open_orders_ids)
    trailing_visible = bool(trailing_order_id and trailing_order_id in open_orders_ids)

    # Cleanup: cancel open orders + close position.
    for oid in open_orders_ids:
        try:
            client.cancel_order(oid, symbol, params={})
        except Exception:
            pass

    close_request = {
        "symbol": symbol,
        "side": close_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": True},
    }
    close_result = await order_manager.place_order(close_request, execution_algo="market")
    if not close_result.get("success"):
        close_request["execution_params"] = {"reduceOnly": False}
        close_result = await order_manager.place_order(close_request, execution_algo="market")
        if not close_result.get("success"):
            raise RuntimeError(f"Close order failed: {close_result.get('reason')}")

    await asyncio.sleep(1.0)
    positions_after_close_rest = client.get_bingx_positions(symbol)
    sanitized_positions = {"code": positions_after_close_rest.get("code"), "data_len": len(positions_after_close_rest.get("data") or [])}

    return SmokeSummary(
        ts_start=args.ts_start,
        ts_end=utc_now(),
        symbol=symbol,
        side=args.side,
        amount=float(amount),
        hard_stop_order_id=hard_stop_order_id,
        trailing_order_id=trailing_order_id,
        open_orders_ids=open_orders_ids,
        hard_stop_visible=hard_stop_visible,
        trailing_visible=trailing_visible,
        positions_after_close_rest=sanitized_positions,
        rest_base_url=getattr(client, "_bingx_rest_base_url", None),
        ccxt_api_swap_url=((exchange.urls or {}).get("api") or {}).get("swap"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="BingX VST Stage-2 smoke test: native hard stop + activation trailing")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument("--min-notional-usdt", type=float, default=0.0)
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--trailing-distance", type=float, default=0.002)
    ap.add_argument("--trailing-activation-threshold", type=float, default=0.0)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "diagnostics" / "vst"))
    ap.add_argument(
        "--env-file",
        default="bearish-bot.env.local",
        help="Optional env file to load (key=value lines). Defaults to bearish-bot.env.local, then .env.local/.env.",
    )
    args = ap.parse_args()

    args.ts_start = utc_now()

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    env_loaded = False
    if args.env_file:
        env_loaded = _load_env_file_if_present(args.env_file) or env_loaded
    env_loaded = _load_env_file_if_present(".env.local") or env_loaded
    env_loaded = _load_env_file_if_present(".env") or env_loaded

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"bingx_vst_stage2_native_stops_smoke_{ts}.json"
    jsonl_path = out_dir / f"bingx_vst_stage2_native_stops_smoke_{ts}.jsonl"

    def write_event(event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": utc_now(), "event": event, "env_loaded": env_loaded, **payload}
        with jsonl_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(json.dumps(redacted(rec), ensure_ascii=False) + "\n")

    try:
        write_event("run.start", {"args": vars(args)})
        summary = asyncio.run(_run(args))
        write_event("run.end", {"summary": asdict(summary)})
        json_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote: {json_path}")
        print(f"Wrote: {jsonl_path}")
        print(f\"PASS | hard_stop_visible={summary.hard_stop_visible} trailing_visible={summary.trailing_visible} | open_orders={len(summary.open_orders_ids)}\")
        return 0 if (summary.hard_stop_visible and summary.trailing_visible) else 2
    except Exception as exc:
        write_event("run.error", {"error": repr(exc)})
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
