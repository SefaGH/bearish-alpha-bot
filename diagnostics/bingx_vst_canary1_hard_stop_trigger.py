"""
VST Canary-1: prove exchange-side HARD_STOP close + skip_market_exit behavior (BingX swap, hedge mode).

This script makes real API calls against BingX VST (sandbox).

Goal:
  1) Open a tiny position (LONG or SHORT).
  2) Place an exchange-native hard stop via the bot-integrated path (AdvancedPositionManager.open_position()).
  3) Wait for the exchange to close the position via the stop trigger (tight stop distance).
  4) Verify the bot detects "exchange already flat" and emits skip_market_exit (no market close sent).

Artifacts written to diagnostics/vst/ (gitignored):
  - vst_canary1_hard_stop_trigger_<ts>.json
  - vst_canary1_hard_stop_trigger_<ts>.jsonl

Safety:
  - Forces TRADING_MODE=live, EXECUTION_BACKEND=ccxt, BINGX_ENV=vst.
  - Forces BINGX_NATIVE_HARD_STOP_ENABLED=true and disables native trailing.
  - Requires hedge mode (dualSidePosition=true).
  - Always attempts to exit cleanly on timeout (market close), then cancels open orders.
  - Redacts credential-like fields in artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ccxt_client import CcxtClient  # noqa: E402
from core.execution_env import get_bingx_env, get_execution_backend, get_trading_mode  # noqa: E402
from core.order_manager import SmartOrderManager  # noqa: E402
from core.position_manager import AdvancedPositionManager  # noqa: E402


logger = logging.getLogger("bingx_vst_canary1_hard_stop_trigger")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redacted(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"apikey", "secret", "password", "passphrase", "token"}:
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


def _infer_amount_step(market: Dict[str, Any]) -> float:
    precision = ((market.get("precision") or {}).get("amount"))
    if isinstance(precision, (int, float)):
        if precision >= 1:
            return 10 ** (-int(precision))
        if precision > 0:
            return float(precision)
    return 0.0


def _compute_amount(exchange, symbol: str, notional_usdt: float, min_notional_usdt: float) -> float:
    ticker = exchange.fetch_ticker(symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if last <= 0:
        raise RuntimeError("Failed to fetch last price for sizing")

    effective_notional = max(notional_usdt, min_notional_usdt or 0.0)
    raw_amount = effective_notional / last

    market = exchange.market(symbol)
    min_amount = _safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"))
    step = _infer_amount_step(market)
    hard_floor = max(min_amount, step)
    raw_amount = max(raw_amount, hard_floor)

    try:
        rounded_str = exchange.amount_to_precision(symbol, raw_amount)
        amount = _safe_float(rounded_str)
    except Exception:
        amount = raw_amount

    if amount <= 0 and hard_floor > 0:
        amount = hard_floor
    return float(amount)


def _sanitize_order(order: Any) -> Dict[str, Any]:
    if not isinstance(order, dict):
        return {"type": str(type(order))}
    keep = ("id", "clientOrderId", "symbol", "type", "side", "status", "amount", "filled", "remaining", "price", "average")
    out: Dict[str, Any] = {k: order.get(k) for k in keep if k in order}
    info = order.get("info")
    if isinstance(info, dict):
        info_keep = {}
        for k in ("workingType", "positionSide", "stopPrice", "stopLossPrice", "triggerPrice", "reduceOnly", "orderType"):
            if k in info:
                info_keep[k] = info.get(k)
        if info_keep:
            out["info"] = info_keep
    return out


def _summarize_positions(resp: Any) -> Dict[str, Any]:
    if not isinstance(resp, dict):
        return {"type": str(type(resp))}
    data = resp.get("data")
    out: Dict[str, Any] = {"code": resp.get("code"), "data_len": len(data or []) if isinstance(data, list) else None}
    if isinstance(data, list):
        slim: List[Dict[str, Any]] = []
        for pos in data[:20]:
            if not isinstance(pos, dict):
                continue
            slim.append(
                {
                    "symbol": pos.get("symbol"),
                    "positionSide": pos.get("positionSide"),
                    "positionAmt": pos.get("positionAmt"),
                    "avgPrice": pos.get("avgPrice") or pos.get("entryPrice"),
                    "markPrice": pos.get("markPrice"),
                }
            )
        out["data"] = slim
    return out


def _pos_amt(pos: Any) -> float:
    if not isinstance(pos, dict):
        return 0.0
    return abs(_safe_float(pos.get("positionAmt")) or 0.0)


def _write_jsonl(path: Path, event: str, payload: Dict[str, Any]) -> None:
    record = {"ts": utc_now(), "event": event, "payload": redacted(payload)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class _StubRiskManager:
    def __init__(self) -> None:
        self.active_positions: Dict[str, Any] = {}

    def register_position(self, position_id: str, position: Dict[str, Any]) -> None:
        self.active_positions[position_id] = position

    def close_position(self, position_id: str, exit_price: float, realized_pnl: float) -> None:
        self.active_positions.pop(position_id, None)


class _StubPortfolioManager:
    def __init__(self, exchange_clients: Dict[str, Any]) -> None:
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


async def _preflight_or_cleanup(
    *,
    client: CcxtClient,
    exchange,
    order_manager: SmartOrderManager,
    symbol: str,
    allow_cleanup: bool,
    jsonl_path: Path,
) -> Dict[str, Any]:
    """
    Fail-fast if open orders/positions exist unless allow_cleanup=true.
    Cleanup is best-effort and limited to the single symbol.
    """
    result: Dict[str, Any] = {"ok": False, "allow_cleanup": allow_cleanup, "errors": [], "before": {}, "after": {}}
    try:
        open_orders = exchange.fetch_open_orders(symbol) or []
    except Exception as exc:
        open_orders = []
        result["errors"].append(f"fetch_open_orders_failed:{str(exc)[:120]}")

    try:
        positions_raw = client.get_bingx_positions(symbol)
    except Exception as exc:
        positions_raw = {"error": str(exc)[:120]}
        result["errors"].append(f"get_positions_failed:{str(exc)[:120]}")

    positions_data = positions_raw.get("data") if isinstance(positions_raw, dict) else None
    open_pos_items = [p for p in (positions_data or []) if isinstance(p, dict) and _pos_amt(p) > 0] if isinstance(positions_data, list) else []

    result["before"] = {
        "open_orders_count": len(open_orders or []),
        "open_positions_count": len(open_pos_items or []),
        "positions": _summarize_positions(positions_raw),
        "open_orders": [_sanitize_order(o) for o in (open_orders or [])][:20],
    }
    _write_jsonl(jsonl_path, "preflight.before", result["before"])

    if (open_orders or open_pos_items) and not allow_cleanup:
        result["errors"].append("dirty_state")
        return result

    if allow_cleanup:
        # Cancel open orders.
        canceled: List[str] = []
        for o in open_orders or []:
            oid = (o or {}).get("id") if isinstance(o, dict) else None
            if not oid:
                continue
            try:
                client.cancel_order(str(oid), symbol, params={})
                canceled.append(str(oid))
            except Exception:
                continue
        _write_jsonl(jsonl_path, "cleanup.cancel_open_orders", {"count": len(canceled), "order_ids": canceled})

        # Close any open positions (hedge LONG/SHORT) for the symbol.
        closed: List[Dict[str, Any]] = []
        for p in open_pos_items:
            position_side = str(p.get("positionSide") or "").upper()
            amt = _pos_amt(p)
            if amt <= 0:
                continue
            close_side = "sell" if position_side == "LONG" else ("buy" if position_side == "SHORT" else None)
            if not close_side:
                continue
            req = {
                "symbol": symbol,
                "side": close_side,
                "amount": amt,
                "exchange": "bingx",
                "execution_params": {"reduceOnly": True, "positionSide": position_side},
            }
            try:
                r = await order_manager.place_order(req, execution_algo="market")
                closed.append({"positionSide": position_side, "qty": amt, "ok": bool(r.get("success")), "order_id": r.get("order_id")})
            except Exception as exc:
                closed.append({"positionSide": position_side, "qty": amt, "ok": False, "error": str(exc)[:120]})
            await asyncio.sleep(0.2)
        _write_jsonl(jsonl_path, "cleanup.close_positions", {"closed": closed})

    # Verify clean state after cleanup.
    try:
        after_open_orders = exchange.fetch_open_orders(symbol) or []
    except Exception:
        after_open_orders = []
    try:
        after_positions_raw = client.get_bingx_positions(symbol)
    except Exception:
        after_positions_raw = {}
    after_positions_data = after_positions_raw.get("data") if isinstance(after_positions_raw, dict) else None
    after_open_pos_items = [p for p in (after_positions_data or []) if isinstance(p, dict) and _pos_amt(p) > 0] if isinstance(after_positions_data, list) else []

    result["after"] = {
        "open_orders_count": len(after_open_orders or []),
        "open_positions_count": len(after_open_pos_items or []),
        "positions": _summarize_positions(after_positions_raw),
        "open_orders": [_sanitize_order(o) for o in (after_open_orders or [])][:20],
    }
    _write_jsonl(jsonl_path, "preflight.after", result["after"])

    if after_open_orders or after_open_pos_items:
        result["errors"].append("dirty_state_after_cleanup")
        return result

    result["ok"] = True
    return result


@dataclass
class Canary1Summary:
    ts_start: str
    ts_end: str
    ok: bool
    timed_out: bool
    side: str
    symbol: str
    amount: float
    entry_order_id: Optional[str]
    entry_avg_price: Optional[float]
    hard_stop_order_id: Optional[str]
    hard_stop_stop_price: Optional[float]
    hard_stop_position_side: Optional[str]
    open_orders_after_entry: List[Dict[str, Any]]
    positions_after_entry_rest: Dict[str, Any]
    skip_market_exit_signal: Dict[str, Any]
    local_close_result: Dict[str, Any]
    market_close_sent: bool
    market_close_order_id: Optional[str]
    positions_at_end_rest: Dict[str, Any]
    open_orders_at_end: List[Dict[str, Any]]
    env: Dict[str, Any]
    endpoints: Dict[str, Any]


async def _run(args: argparse.Namespace) -> Canary1Summary:
    ts_start = utc_now()

    out_dir = Path.cwd() / "diagnostics" / "vst"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"vst_canary1_hard_stop_trigger_{ts_slug}.jsonl"
    json_path = out_dir / f"vst_canary1_hard_stop_trigger_{ts_slug}.json"
    jsonl_path.write_text("", encoding="utf-8")

    if args.env_file:
        _load_env_file_if_present(args.env_file)
    else:
        _load_env_file_if_present("bearish-bot.env.local")
        _load_env_file_if_present(".env.local")

    # Force safe VST routing.
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"

    # Force native hard stop and fast reconcile; disable native trailing for this canary.
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED"] = "false"
    os.environ["BINGX_NATIVE_EXIT_RECONCILE_INTERVAL_S"] = str(max(1, int(args.reconcile_interval_s)))
    os.environ["BINGX_NATIVE_ORDER_SYNC_INTERVAL_S"] = "2"

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

    preflight = await _preflight_or_cleanup(
        client=client,
        exchange=exchange,
        order_manager=order_manager,
        symbol=symbol,
        allow_cleanup=bool(args.allow_cleanup),
        jsonl_path=jsonl_path,
    )
    if not preflight.get("ok"):
        raise RuntimeError(f"Preflight failed: {preflight.get('errors')}")

    ticker = exchange.fetch_ticker(symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if last <= 0:
        raise RuntimeError("Failed to fetch last price for stop computation")

    stop_distance = float(max(0.00001, args.stop_distance_pct))
    if args.side == "long":
        stop = last * (1 - stop_distance)
        take_profit = last * 1.50
        open_side = "buy"
        close_side = "sell"
        position_side = "LONG"
    else:
        stop = last * (1 + stop_distance)
        take_profit = last * 0.50
        open_side = "sell"
        close_side = "buy"
        position_side = "SHORT"

    signal = {
        "symbol": symbol,
        "side": args.side,
        "exchange": "bingx",
        "strategy": "diagnostics_vst_canary1_hard_stop_trigger",
        "stop": float(stop),
        "take_profit": float(take_profit),
    }

    open_request = {
        "symbol": symbol,
        "side": open_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": False, "positionSide": position_side},
    }

    logger.info("[CANARY-1] Opening %s %s amt=%.8f (stop_distance_pct=%.5f)", symbol, args.side, amount, stop_distance)
    open_result = await order_manager.place_order(open_request, execution_algo="market")
    _write_jsonl(jsonl_path, "order.open.result", {"open_result": open_result})
    if not open_result.get("success"):
        raise RuntimeError(f"Open order failed: {open_result.get('reason')}")

    entry_order_id = open_result.get("order_id")
    entry_avg = _safe_float(open_result.get("avg_price")) or None

    pos_result = await position_manager.open_position(signal, open_result)
    _write_jsonl(jsonl_path, "position.open.result", {"pos_result": pos_result})
    if not pos_result.get("success"):
        raise RuntimeError(f"open_position failed: {pos_result.get('reason')}")

    position_id = pos_result.get("position_id")
    if not position_id:
        raise RuntimeError("open_position returned missing position_id")

    position = position_manager.positions.get(position_id)
    if not isinstance(position, dict):
        raise RuntimeError("position_manager missing position after open_position()")

    hard_stop_order_id = position.get("native_hard_stop_order_id")
    hard_stop_stop_price = _safe_float(position.get("native_hard_stop_stop_price")) or None
    hard_stop_position_side = position.get("native_hard_stop_position_side")

    open_orders_entry_raw = exchange.fetch_open_orders(symbol) or []
    open_orders_after_entry = [_sanitize_order(o) for o in open_orders_entry_raw]
    _write_jsonl(
        jsonl_path,
        "openOrders.after_entry",
        {"count": len(open_orders_after_entry), "order_ids": [o.get("id") for o in open_orders_after_entry]},
    )

    positions_after_entry_rest = _summarize_positions(client.get_bingx_positions(symbol))
    _write_jsonl(jsonl_path, "positions.after_entry", {"positions": positions_after_entry_rest})

    timed_out = False
    market_close_sent = False
    market_close_order_id: Optional[str] = None
    skip_market_exit_signal: Dict[str, Any] = {}
    local_close_result: Dict[str, Any] = {}

    started = time.monotonic()
    polls = 0
    while True:
        polls += 1
        await asyncio.sleep(float(max(0.5, args.poll_interval_s)))

        ticker = exchange.fetch_ticker(symbol)
        last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
        if isinstance(position, dict) and last > 0:
            position["exit_price"] = last
            position["native_exit_reconcile_last_ts"] = 0.0  # force reconcile each tick for evidence runs

        skip_market_exit_signal = await position_manager.manage_position_exits(position_id)
        _write_jsonl(
            jsonl_path,
            "pm.manage_position_exits",
            {"poll": polls, "last": last, "signal": skip_market_exit_signal},
        )

        if skip_market_exit_signal.get("skip_market_exit") is True:
            logger.warning("[CANARY-1] Exchange-side close detected (skip_market_exit). position_id=%s", position_id)
            break

        if (time.monotonic() - started) >= float(max(5.0, args.timeout_s)):
            timed_out = True
            logger.warning("[CANARY-1] Timeout waiting for stop trigger; closing by MARKET for safety. position_id=%s", position_id)

            close_request = {
                "symbol": symbol,
                "side": close_side,
                "amount": float(position.get("amount") or amount),
                "exchange": "bingx",
                "execution_params": {"reduceOnly": True, "positionSide": position_side},
            }
            close_result = await order_manager.place_order(close_request, execution_algo="market")
            _write_jsonl(jsonl_path, "order.market_close.timeout.result", {"close_result": close_result})
            if not close_result.get("success"):
                close_request["execution_params"] = {"reduceOnly": False, "positionSide": position_side}
                close_result = await order_manager.place_order(close_request, execution_algo="market")
                _write_jsonl(jsonl_path, "order.market_close.timeout.retry.result", {"close_result": close_result})
            market_close_sent = True
            market_close_order_id = close_result.get("order_id")

            raw_exit_price = close_result.get("avg_price") or last
            exit_price = float(_safe_float(raw_exit_price) or last)
            local_close_result = await position_manager.close_position(
                position_id,
                exit_price,
                "timeout_market_close",
                exit_order_id=market_close_order_id,
            )
            _write_jsonl(jsonl_path, "pm.close_position.timeout.result", {"position_id": position_id, "result": local_close_result})
            break

    # Close locally (skip_market_exit path) if exchange closed it.
    if not timed_out and skip_market_exit_signal.get("skip_market_exit") is True:
        raw_exit_price = skip_market_exit_signal.get("exit_price") or (position.get("exit_price") if isinstance(position, dict) else None) or last
        local_exit_price = float(_safe_float(raw_exit_price) or last)
        exit_reason = skip_market_exit_signal.get("exit_reason") or "stop_loss"
        local_close_result = await position_manager.close_position(position_id, local_exit_price, exit_reason)
        _write_jsonl(jsonl_path, "pm.close_position.result", {"position_id": position_id, "result": local_close_result})

    # Final state snapshots.
    positions_at_end_rest = _summarize_positions(client.get_bingx_positions(symbol))
    try:
        open_orders_at_end_raw = exchange.fetch_open_orders(symbol) or []
    except Exception:
        open_orders_at_end_raw = []
    open_orders_at_end = [_sanitize_order(o) for o in open_orders_at_end_raw]
    _write_jsonl(jsonl_path, "exchange.final_state", {"positions": positions_at_end_rest, "open_orders": open_orders_at_end})

    ok = bool(skip_market_exit_signal.get("skip_market_exit") is True and not market_close_sent and not timed_out)

    ts_end = utc_now()
    summary = Canary1Summary(
        ts_start=ts_start,
        ts_end=ts_end,
        ok=ok,
        timed_out=timed_out,
        side=args.side,
        symbol=symbol,
        amount=float(amount),
        entry_order_id=entry_order_id,
        entry_avg_price=entry_avg,
        hard_stop_order_id=hard_stop_order_id,
        hard_stop_stop_price=hard_stop_stop_price,
        hard_stop_position_side=hard_stop_position_side,
        open_orders_after_entry=open_orders_after_entry,
        positions_after_entry_rest=positions_after_entry_rest,
        skip_market_exit_signal=skip_market_exit_signal,
        local_close_result=local_close_result,
        market_close_sent=market_close_sent,
        market_close_order_id=market_close_order_id,
        positions_at_end_rest=positions_at_end_rest,
        open_orders_at_end=open_orders_at_end,
        env={
            "TRADING_MODE": get_trading_mode(),
            "EXECUTION_BACKEND": get_execution_backend(),
            "BINGX_ENV": get_bingx_env(),
            "BINGX_NATIVE_HARD_STOP_ENABLED": os.getenv("BINGX_NATIVE_HARD_STOP_ENABLED", ""),
            "BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED": os.getenv("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", ""),
            "BINGX_NATIVE_EXIT_RECONCILE_INTERVAL_S": os.getenv("BINGX_NATIVE_EXIT_RECONCILE_INTERVAL_S", ""),
        },
        endpoints={
            "ccxt_sandbox": bool(getattr(exchange, "sandbox", False)),
            "ccxt_swap_url": (((getattr(exchange, "urls", None) or {}).get("api") or {}).get("swap")),
            "rest_base_url": getattr(client, "_bingx_rest_base_url", None),
            "hedged": getattr(client, "_bingx_is_hedged", None),
        },
    )

    json_path.write_text(json.dumps(redacted(asdict(summary)), indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_jsonl(jsonl_path, "run.end", {"summary": asdict(summary)})
    return summary


def _configure_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="BingX VST Canary-1: exchange-side hard-stop trigger evidence")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--side", choices=["long", "short"], required=True)
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument("--min-notional-usdt", type=float, default=5.0)
    ap.add_argument("--stop-distance-pct", type=float, default=0.001)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--poll-interval-s", type=float, default=2.0)
    ap.add_argument("--reconcile-interval-s", type=int, default=1)
    ap.add_argument("--allow-cleanup", action="store_true")
    ap.add_argument("--env-file", default="")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()

    _configure_logging(args.verbose)

    try:
        asyncio.run(_run(args))
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
