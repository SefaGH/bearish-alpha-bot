"""
Stage-2 BingX VST evidence runner: validate bot-integrated native conditional stops (hedge mode).

This script makes real API calls against BingX VST (sandbox).

Artifacts are written to diagnostics/vst/ (gitignored):
  - stage2_run_long.json / .jsonl / .log
  - stage2_run_short.json / .jsonl / .log
  - stage2_evidence.md (summarized excerpts)

Evidence captured (per side):
  - openOrders snapshot after entry (hard stop present)
  - openOrders snapshot after trailing activation (trailing present)
  - manage_position_exits() suppression signals for stop-loss + trailing
  - skip_market_exit signal when exchange is already flat (simulated by exchange-side close)
  - cancel-on-close outcomes (idempotent tolerant)

Safety:
  - Forces TRADING_MODE=live, EXECUTION_BACKEND=ccxt, BINGX_ENV=vst.
  - Forces BINGX_NATIVE_* flags ON for the duration of the run.
  - Requires hedge mode (dualSidePosition=true).
  - Redacts any credential-like fields in artifacts.
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
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ccxt_client import CcxtClient  # noqa: E402
from core.order_manager import SmartOrderManager  # noqa: E402
from core.position_manager import AdvancedPositionManager  # noqa: E402


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
    keep = ("id", "clientOrderId", "symbol", "type", "side", "status", "amount", "filled", "remaining", "price", "average", "reduceOnly")
    out: Dict[str, Any] = {k: order.get(k) for k in keep if k in order}
    info = order.get("info")
    if isinstance(info, dict):
        info_keep = {}
        for k in ("workingType", "positionSide", "stopPrice", "stopLossPrice", "triggerPrice", "priceRate", "trailingType", "reduceOnly", "orderType"):
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


def _extract_position_amt(resp: Any, *, native_symbol: str, position_side: str) -> float:
    """
    Return abs(positionAmt) for the requested symbol+positionSide.
    Falls back to 0.0 if not present/parsable.
    """
    if not isinstance(resp, dict):
        return 0.0
    data = resp.get("data")
    if not isinstance(data, list):
        return 0.0
    for pos in data:
        if not isinstance(pos, dict):
            continue
        if str(pos.get("symbol") or "") != str(native_symbol):
            continue
        if str(pos.get("positionSide") or "").upper() != str(position_side).upper():
            continue
        amt = _safe_float(pos.get("positionAmt")) or 0.0
        return float(abs(amt))
    return 0.0


async def _preclean_bingx_symbol(
    *,
    client: CcxtClient,
    exchange,
    order_manager: SmartOrderManager,
    resolved_symbol: str,
    jsonl_path: Path,
) -> None:
    """Best-effort cleanup: cancel open orders and close any remaining hedge positions for the symbol."""
    try:
        open_orders = exchange.fetch_open_orders(resolved_symbol) or []
        canceled: List[str] = []
        for o in open_orders:
            oid = (o or {}).get("id") if isinstance(o, dict) else None
            if not oid:
                continue
            try:
                client.cancel_order(str(oid), resolved_symbol, params={})
                canceled.append(str(oid))
            except Exception:
                continue
        if canceled:
            _write_jsonl(jsonl_path, "preclean.cancel_open_orders", {"count": len(canceled), "order_ids": canceled})
    except Exception as exc:
        _write_jsonl(jsonl_path, "preclean.cancel_open_orders.error", {"error": repr(exc)})

    try:
        resp = client.get_bingx_positions(resolved_symbol)
        data = resp.get("data") if isinstance(resp, dict) else None
        if isinstance(data, list):
            native_symbol = str(getattr(client, "_get_bingx_native_symbol", lambda s: s)(resolved_symbol))
            for pos in data:
                if not isinstance(pos, dict):
                    continue
                if str(pos.get("symbol") or "") != native_symbol:
                    continue
                amt = _safe_float(pos.get("positionAmt")) or 0.0
                if amt == 0:
                    continue
                pos_side = str(pos.get("positionSide") or "").upper() or ("LONG" if amt > 0 else "SHORT")
                close_side = "sell" if pos_side == "LONG" else "buy"
                qty = float(abs(amt))
                close_req = {
                    "symbol": resolved_symbol,
                    "side": close_side,
                    "amount": qty,
                    "exchange": "bingx",
                    "execution_params": {"reduceOnly": True, "positionSide": pos_side},
                }
                close_result = await order_manager.place_order(close_req, execution_algo="market")
                _write_jsonl(
                    jsonl_path,
                    "preclean.close_position.market",
                    {"positionSide": pos_side, "amount": qty, "result": close_result},
                )
    except Exception as exc:
        _write_jsonl(jsonl_path, "preclean.close_position.market.error", {"error": repr(exc)})

    # Cancel again after closes (orphan cleanup).
    try:
        open_orders = exchange.fetch_open_orders(resolved_symbol) or []
        canceled: List[str] = []
        for o in open_orders:
            oid = (o or {}).get("id") if isinstance(o, dict) else None
            if not oid:
                continue
            try:
                client.cancel_order(str(oid), resolved_symbol, params={})
                canceled.append(str(oid))
            except Exception:
                continue
        if canceled:
            _write_jsonl(jsonl_path, "preclean.cancel_open_orders.after_close", {"count": len(canceled), "order_ids": canceled})
    except Exception:
        pass


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


@dataclass
class EvidenceSummary:
    ts_start: str
    ts_end: str
    side: str
    symbol: str
    amount: float
    entry_order_id: Optional[str]
    entry_avg_price: Optional[float]
    hard_stop_order_id: Optional[str]
    trailing_order_id: Optional[str]
    open_orders_after_entry: List[Dict[str, Any]]
    open_orders_after_trailing: List[Dict[str, Any]]
    stop_loss_suppression_result: Dict[str, Any]
    trailing_suppression_result: Dict[str, Any]
    exchange_close_order_id: Optional[str]
    exchange_close_result: Dict[str, Any]
    skip_market_exit_signal: Dict[str, Any]
    local_close_result: Dict[str, Any]
    positions_after_entry_rest: Dict[str, Any]
    positions_after_exchange_close_rest: Dict[str, Any]
    positions_after_local_close_rest: Dict[str, Any]
    rest_base_url: Optional[str]
    ccxt_swap_url: Optional[str]
    ccxt_sandbox: bool


def _configure_file_logging(log_path: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def _write_jsonl(jsonl_path: Path, event: str, payload: Dict[str, Any]) -> None:
    rec = {"ts": utc_now(), "event": event, **payload}
    with jsonl_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(json.dumps(redacted(rec), ensure_ascii=False, default=str) + "\n")


async def _run_one_side(
    *,
    side: str,
    symbol: str,
    notional_usdt: float,
    min_notional_usdt: float,
    trailing_distance: float,
    trailing_activation_threshold: float,
    env_file: str,
    out_prefix: Path,
) -> EvidenceSummary:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    jsonl_path = out_prefix.with_suffix(".jsonl")
    log_path = out_prefix.with_suffix(".log")

    _configure_file_logging(log_path)

    ts_start = utc_now()
    _write_jsonl(jsonl_path, "run.start", {"side": side, "symbol": symbol, "notional_usdt": notional_usdt})

    env_loaded = False
    if env_file:
        env_loaded = _load_env_file_if_present(env_file) or env_loaded
    env_loaded = _load_env_file_if_present(".env.local") or env_loaded
    env_loaded = _load_env_file_if_present(".env") or env_loaded
    _write_jsonl(jsonl_path, "env.loaded", {"env_loaded": env_loaded, "env_file": env_file})

    # Force safe canary routing (do not rely on ambient env defaults).
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_EXIT_RECONCILE_INTERVAL_S"] = "1"
    os.environ["BINGX_NATIVE_ORDER_SYNC_INTERVAL_S"] = "1"

    key = _require_env("BINGX_KEY")
    secret = _require_env("BINGX_SECRET")

    client = CcxtClient("bingx", {"apiKey": key, "secret": secret})
    exchange = client.ex

    resolved_symbol = _resolve_symbol(exchange, symbol)
    amount = _compute_amount(exchange, resolved_symbol, notional_usdt, min_notional_usdt)

    client.ensure_bingx_hedge_mode(resolved_symbol, require_hedged=True)

    order_manager = SmartOrderManager(market_data_pipeline=None, risk_manager=None, exchange_clients={"bingx": client})
    risk_manager = _StubRiskManager()
    portfolio_manager = _StubPortfolioManager(exchange_clients={"bingx": client})
    position_manager = AdvancedPositionManager(
        risk_manager=risk_manager,
        order_manager=order_manager,
        websocket_manager=None,
        portfolio_manager=portfolio_manager,
    )

    open_side = "buy" if side == "long" else "sell"
    close_side = "sell" if open_side == "buy" else "buy"
    position_side = "LONG" if side == "long" else "SHORT"

    await _preclean_bingx_symbol(
        client=client,
        exchange=exchange,
        order_manager=order_manager,
        resolved_symbol=resolved_symbol,
        jsonl_path=jsonl_path,
    )

    ticker = exchange.fetch_ticker(resolved_symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if last <= 0:
        raise RuntimeError("Failed to fetch last price for stop/tp computation")

    if side == "long":
        stop = last * 0.95
        take_profit = last * 1.10
    else:
        stop = last * 1.05
        take_profit = last * 0.90

    signal = {
        "symbol": resolved_symbol,
        "side": side,
        "exchange": "bingx",
        "strategy": "diagnostics_vst_stage2_native_stops_evidence",
        "stop": stop,
        "take_profit": take_profit,
    }

    open_request = {
        "symbol": resolved_symbol,
        "side": open_side,
        "amount": amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": False, "positionSide": position_side},
    }

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
    if hard_stop_order_id:
        _write_jsonl(jsonl_path, "native.hard_stop.placed", {"position_id": position_id, "order_id": hard_stop_order_id})

    open_orders_entry_raw = exchange.fetch_open_orders(resolved_symbol) or []
    open_orders_after_entry = [_sanitize_order(o) for o in open_orders_entry_raw]
    _write_jsonl(
        jsonl_path,
        "openOrders.after_entry",
        {"count": len(open_orders_after_entry), "order_ids": [o.get("id") for o in open_orders_after_entry]},
    )

    positions_after_entry_rest = _summarize_positions(client.get_bingx_positions(resolved_symbol))
    _write_jsonl(jsonl_path, "positions.after_entry", {"positions": positions_after_entry_rest})

    # Enable trailing with immediate activation so monitor tick places native trailing.
    position_manager.configure_trailing_stop(
        position_id,
        enabled=True,
        delta_pct=float(trailing_distance),
        activation_threshold_pct=float(trailing_activation_threshold),
    )
    await position_manager.monitor_position_pnl(position_id, current_price=position.get("current_price"))

    position = position_manager.positions.get(position_id)
    trailing_order_id = (position or {}).get("native_trailing_stop_order_id") if isinstance(position, dict) else None
    if trailing_order_id:
        _write_jsonl(jsonl_path, "native.trailing.placed", {"position_id": position_id, "order_id": trailing_order_id})

    open_orders_trailing_raw = exchange.fetch_open_orders(resolved_symbol) or []
    open_orders_after_trailing = [_sanitize_order(o) for o in open_orders_trailing_raw]
    _write_jsonl(
        jsonl_path,
        "openOrders.after_trailing",
        {"count": len(open_orders_after_trailing), "order_ids": [o.get("id") for o in open_orders_after_trailing]},
    )

    # Evidence: synthetic stop-loss/trailing logic is suppressed when native ids exist.
    position = position_manager.positions.get(position_id)
    if not isinstance(position, dict):
        raise RuntimeError("position missing before suppression checks")

    position["native_hard_stop_suppress_last_log_ts"] = 0.0
    position["native_trailing_suppress_last_log_ts"] = 0.0

    stop_loss = _safe_float(position.get("stop_loss")) or stop
    trailing_distance_val = _safe_float(position.get("trailing_stop_distance")) or float(trailing_distance)
    entry_price = _safe_float(position.get("entry_price")) or (entry_avg or last)

    stop_hit_price = stop_loss * (0.99 if side == "long" else 1.01)
    position["exit_price"] = stop_hit_price
    stop_loss_suppression_result = await position_manager.manage_position_exits(position_id)
    _write_jsonl(
        jsonl_path,
        "suppression.stop_loss",
        {"position_id": position_id, "exit_price": stop_hit_price, "result": stop_loss_suppression_result},
    )

    if side == "long":
        position["highest_price"] = max(_safe_float(position.get("highest_price")) or entry_price, entry_price * 1.01)
        trailing_stop_level = float(position["highest_price"]) * (1 - trailing_distance_val)
        trail_hit_price = trailing_stop_level * 0.999
    else:
        position["lowest_price"] = min(_safe_float(position.get("lowest_price")) or entry_price, entry_price * 0.99)
        trailing_stop_level = float(position["lowest_price"]) * (1 + trailing_distance_val)
        trail_hit_price = trailing_stop_level * 1.001

    position["exit_price"] = trail_hit_price
    trailing_suppression_result = await position_manager.manage_position_exits(position_id)
    _write_jsonl(
        jsonl_path,
        "suppression.trailing",
        {"position_id": position_id, "exit_price": trail_hit_price, "result": trailing_suppression_result},
    )

    # Evidence: exchange-side close -> skip_market_exit signal (deterministic).
    # Use exchange-reported position amount for closure sizing (defense-in-depth).
    raw_positions_after_entry = client.get_bingx_positions(resolved_symbol)
    native_symbol = str(getattr(client, "_get_bingx_native_symbol", lambda s: s)(resolved_symbol))
    exchange_pos_amt = _extract_position_amt(raw_positions_after_entry, native_symbol=native_symbol, position_side=position_side)
    close_amount = exchange_pos_amt if exchange_pos_amt > 0 else amount

    close_request = {
        "symbol": resolved_symbol,
        "side": close_side,
        "amount": close_amount,
        "exchange": "bingx",
        "execution_params": {"reduceOnly": True, "positionSide": position_side},
    }
    exchange_close_result = await order_manager.place_order(close_request, execution_algo="market")
    _write_jsonl(jsonl_path, "order.exchange_close.result", {"close_result": exchange_close_result})
    if not exchange_close_result.get("success"):
        close_request["execution_params"] = {"reduceOnly": False, "positionSide": position_side}
        exchange_close_result = await order_manager.place_order(close_request, execution_algo="market")
        _write_jsonl(jsonl_path, "order.exchange_close.retry.result", {"close_result": exchange_close_result})
        if not exchange_close_result.get("success"):
            raise RuntimeError(f"Exchange close failed: {exchange_close_result.get('reason')}")

    exchange_close_order_id = exchange_close_result.get("order_id")

    await asyncio.sleep(3.0)
    positions_after_exchange_close_rest = _summarize_positions(client.get_bingx_positions(resolved_symbol))
    _write_jsonl(jsonl_path, "positions.after_exchange_close", {"positions": positions_after_exchange_close_rest})

    # Now let the bot detect flat state and emit skip_market_exit.
    position = position_manager.positions.get(position_id)
    if isinstance(position, dict):
        position["native_exit_reconcile_last_ts"] = 0.0
        position["exit_price"] = _safe_float(exchange_close_result.get("avg_price")) or _safe_float(position.get("current_price")) or last

    skip_market_exit_signal = await position_manager.manage_position_exits(position_id)
    _write_jsonl(jsonl_path, "pm.skip_market_exit.signal", {"position_id": position_id, "signal": skip_market_exit_signal})

    # Close locally (as LiveTradingEngine would do on skip_market_exit).
    raw_exit_price = skip_market_exit_signal.get("exit_price") or exchange_close_result.get("avg_price") or last
    local_exit_price = float(_safe_float(raw_exit_price) or last)
    exit_reason = skip_market_exit_signal.get("exit_reason") or "manual"
    local_close_result = await position_manager.close_position(position_id, local_exit_price, exit_reason)
    _write_jsonl(jsonl_path, "pm.close_position.result", {"position_id": position_id, "result": local_close_result})

    await asyncio.sleep(1.0)
    positions_after_local_close_rest = _summarize_positions(client.get_bingx_positions(resolved_symbol))
    _write_jsonl(jsonl_path, "positions.after_local_close", {"positions": positions_after_local_close_rest})

    ts_end = utc_now()
    summary = EvidenceSummary(
        ts_start=ts_start,
        ts_end=ts_end,
        side=side,
        symbol=resolved_symbol,
        amount=float(amount),
        entry_order_id=entry_order_id,
        entry_avg_price=entry_avg,
        hard_stop_order_id=hard_stop_order_id,
        trailing_order_id=trailing_order_id,
        open_orders_after_entry=open_orders_after_entry,
        open_orders_after_trailing=open_orders_after_trailing,
        stop_loss_suppression_result=stop_loss_suppression_result,
        trailing_suppression_result=trailing_suppression_result,
        exchange_close_order_id=exchange_close_order_id,
        exchange_close_result=exchange_close_result,
        skip_market_exit_signal=skip_market_exit_signal,
        local_close_result=local_close_result,
        positions_after_entry_rest=positions_after_entry_rest,
        positions_after_exchange_close_rest=positions_after_exchange_close_rest,
        positions_after_local_close_rest=positions_after_local_close_rest,
        rest_base_url=getattr(client, "_bingx_rest_base_url", None),
        ccxt_swap_url=((getattr(exchange, "urls", None) or {}).get("api") or {}).get("swap"),
        ccxt_sandbox=bool(getattr(exchange, "sandbox", False)),
    )

    json_path.write_text(json.dumps(redacted(asdict(summary)), indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_jsonl(jsonl_path, "run.end", {"summary": asdict(summary)})
    return summary


def _write_evidence_markdown(out_dir: Path, summaries: List[EvidenceSummary]) -> Path:
    md_path = out_dir / "stage2_evidence.md"
    lines: List[str] = []
    lines.append(f"# Stage-2 VST Evidence (generated)\n")
    lines.append(f"- Generated at: `{utc_now()}`")
    lines.append(f"- Outputs directory: `{out_dir}`\n")

    for s in summaries:
        prefix = out_dir / f"stage2_run_{s.side}"
        lines.append(f"## {s.side.upper()}\n")
        lines.append(f"- Summary: `{prefix.with_suffix('.json')}`")
        lines.append(f"- Events: `{prefix.with_suffix('.jsonl')}`")
        lines.append(f"- Logs: `{prefix.with_suffix('.log')}`")
        lines.append(f"- CCXT swap URL: `{s.ccxt_swap_url}`")
        lines.append(f"- REST base URL: `{s.rest_base_url}`\n")

        entry_ids = [o.get("id") for o in s.open_orders_after_entry if o.get("id")]
        trailing_ids = [o.get("id") for o in s.open_orders_after_trailing if o.get("id")]

        lines.append("**Evidence**")
        lines.append(f"- Hard stop orderId: `{s.hard_stop_order_id}` (openOrders-after-entry contains={str(s.hard_stop_order_id in entry_ids)})")
        lines.append(f"- Trailing orderId: `{s.trailing_order_id}` (openOrders-after-trailing contains={str(s.trailing_order_id in trailing_ids)})")
        lines.append(f"- Stop-loss suppression: `{s.stop_loss_suppression_result}`")
        lines.append(f"- Trailing suppression: `{s.trailing_suppression_result}`")
        lines.append(f"- skip_market_exit signal: `{s.skip_market_exit_signal}`")
        lines.append(f"- close_position result: `{s.local_close_result}`\n")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(description="BingX VST Stage-2 evidence runner (native hard stop + activation trailing)")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument("--min-notional-usdt", type=float, default=0.0)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--trailing-distance", type=float, default=0.002)
    ap.add_argument("--trailing-activation-threshold", type=float, default=0.0)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "diagnostics" / "vst"))
    ap.add_argument(
        "--env-file",
        default="bearish-bot.env.local",
        help="Optional env file to load (key=value lines). Defaults to bearish-bot.env.local, then .env.local/.env.",
    )
    args = ap.parse_args()

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sides = ["long", "short"] if args.side == "both" else [args.side]
    summaries: List[EvidenceSummary] = []
    for s in sides:
        prefix = out_dir / f"stage2_run_{s}"
        summaries.append(
            asyncio.run(
                _run_one_side(
                    side=s,
                    symbol=args.symbol,
                    notional_usdt=float(args.notional_usdt),
                    min_notional_usdt=float(args.min_notional_usdt),
                    trailing_distance=float(args.trailing_distance),
                    trailing_activation_threshold=float(args.trailing_activation_threshold),
                    env_file=str(args.env_file or ""),
                    out_prefix=prefix,
                )
            )
        )

    md_path = _write_evidence_markdown(out_dir, summaries)
    print(f"Wrote: {md_path}")
    for s in summaries:
        print(f"Wrote: {out_dir / f'stage2_run_{s.side}.json'}")
        print(f"Wrote: {out_dir / f'stage2_run_{s.side}.jsonl'}")
        print(f"Wrote: {out_dir / f'stage2_run_{s.side}.log'}")

    # Basic pass/fail exit code:
    ok = True
    for s in summaries:
        ok = ok and bool(s.hard_stop_order_id) and bool(s.trailing_order_id)
        ok = ok and s.skip_market_exit_signal.get("skip_market_exit") is True
        ok = ok and s.local_close_result.get("success") is True
        ok = ok and (s.positions_after_local_close_rest.get("data_len") in (0, None))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
