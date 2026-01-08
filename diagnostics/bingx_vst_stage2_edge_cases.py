"""
Stage-2 BingX VST edge-case runner (hedge mode).

Runs VST-only safety checks for:
  - partial close / size change -> conditional qty resync
  - race: exchange already flat vs engine close -> preflight skip_market_exit
  - cancel idempotency (already-canceled / not-found tolerance)

Artifacts are written to diagnostics/vst/ (gitignored):
  - stage2_edge_cases_matrix.json
  - stage2_edge_cases_events.jsonl
  - stage2_edge_cases.log
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
# Support both `import core.*` and modules that expect `import src.*`.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.ccxt_client import CcxtClient  # noqa: E402
from core.live_trading_engine import LiveTradingEngine  # noqa: E402
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


def _extract_position_amt(resp: Any, *, native_symbol: str, position_side: str) -> float:
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
        self.cfg: Dict[str, Any] = {}
        self.active_positions: Dict[str, Any] = {}
        self._trade_count = 0

    def register_position(self, position_id: str, position: Dict[str, Any]) -> None:
        self.active_positions[position_id] = position

    def close_position(self, position_id: str, exit_price: float, realized_pnl: float) -> None:
        self.active_positions.pop(position_id, None)

    def increment_trade_count(self) -> None:
        self._trade_count += 1


@dataclass
class EdgeCaseResult:
    name: str
    passed: bool
    details: Dict[str, Any]


async def _run_one_side(
    *,
    side: str,
    symbol: str,
    notional_usdt: float,
    min_notional_usdt: float,
    trailing_distance: float,
    env_file: str,
    jsonl_path: Path,
) -> List[EdgeCaseResult]:
    results: List[EdgeCaseResult] = []

    # Force safe VST routing + native flags for the duration of the run.
    os.environ["TRADING_MODE"] = "live"
    os.environ["EXECUTION_BACKEND"] = "ccxt"
    os.environ["BINGX_ENV"] = "vst"
    os.environ["BINGX_NATIVE_HARD_STOP_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED"] = "true"
    os.environ["BINGX_NATIVE_EXIT_RECONCILE_INTERVAL_S"] = "1"
    os.environ["BINGX_NATIVE_ORDER_SYNC_INTERVAL_S"] = "1"

    env_loaded = False
    if env_file:
        env_loaded = _load_env_file_if_present(env_file) or env_loaded
    env_loaded = _load_env_file_if_present(".env.local") or env_loaded
    env_loaded = _load_env_file_if_present(".env") or env_loaded
    _write_jsonl(jsonl_path, "env.loaded", {"env_loaded": env_loaded, "env_file": env_file, "side": side})

    key = _require_env("BINGX_KEY")
    secret = _require_env("BINGX_SECRET")

    client = CcxtClient("bingx", {"apiKey": key, "secret": secret})
    exchange = client.ex

    resolved_symbol = _resolve_symbol(exchange, symbol)
    amount = _compute_amount(exchange, resolved_symbol, notional_usdt, min_notional_usdt)
    # For partial-close edge cases we need an amount that can be split while respecting min step.
    market = exchange.market(resolved_symbol)
    min_amount = _safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"))
    step = _infer_amount_step(market)
    min_qty = max(min_amount, step, 0.0)
    if min_qty > 0 and amount < (2.0 * min_qty):
        bumped = 2.0 * min_qty
        try:
            bumped = _safe_float(exchange.amount_to_precision(resolved_symbol, bumped))
        except Exception:
            pass
        amount = float(bumped)

    client.ensure_bingx_hedge_mode(resolved_symbol, require_hedged=True)

    order_manager = SmartOrderManager(market_data_pipeline=None, risk_manager=None, exchange_clients={"bingx": client})

    # Pre-clean: cancel open orders and close any existing positions for this symbol
    # (avoids false positives from prior runs).
    try:
        open_orders = exchange.fetch_open_orders(resolved_symbol) or []
        for o in open_orders:
            oid = (o or {}).get("id") if isinstance(o, dict) else None
            if not oid:
                continue
            try:
                client.cancel_order(str(oid), resolved_symbol, params={})
            except Exception:
                pass

        raw_positions = client.get_bingx_positions(resolved_symbol)
        data = raw_positions.get("data") if isinstance(raw_positions, dict) else None
        native_symbol = str(getattr(client, "_get_bingx_native_symbol", lambda s: s)(resolved_symbol))
        if isinstance(data, list):
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
        _write_jsonl(jsonl_path, "preclean.error", {"side": side, "error": repr(exc)})

    position_manager = AdvancedPositionManager(
        risk_manager=_StubRiskManager(),
        order_manager=order_manager,
        websocket_manager=None,
        portfolio_manager=_StubPortfolioManager(exchange_clients={"bingx": client}),
    )

    open_side = "buy" if side == "long" else "sell"
    close_side = "sell" if open_side == "buy" else "buy"
    position_side = "LONG" if side == "long" else "SHORT"

    ticker = exchange.fetch_ticker(resolved_symbol)
    last = _safe_float(ticker.get("last") or ticker.get("close") or (ticker.get("info") or {}).get("lastPrice"))
    if last <= 0:
        raise RuntimeError("Failed to fetch last price for stop/tp computation")

    stop = last * (0.95 if side == "long" else 1.05)
    take_profit = last * (1.10 if side == "long" else 0.90)

    signal = {
        "symbol": resolved_symbol,
        "side": side,
        "exchange": "bingx",
        "strategy": "diagnostics_vst_stage2_edge_cases",
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
    _write_jsonl(jsonl_path, "order.open.result", {"side": side, "open_result": open_result})
    if not open_result.get("success"):
        raise RuntimeError(f"Open order failed: {open_result.get('reason')}")

    pos_result = await position_manager.open_position(signal, open_result)
    _write_jsonl(jsonl_path, "position.open.result", {"side": side, "pos_result": pos_result})
    if not pos_result.get("success"):
        raise RuntimeError(f"open_position failed: {pos_result.get('reason')}")

    position_id = pos_result.get("position_id")
    position = position_manager.positions.get(position_id) if position_id else None
    hard_id = (position or {}).get("native_hard_stop_order_id")

    # Activate trailing immediately and place native trailing.
    position_manager.configure_trailing_stop(position_id, enabled=True, delta_pct=float(trailing_distance), activation_threshold_pct=0.0)
    await position_manager.monitor_position_pnl(position_id, current_price=(position or {}).get("current_price"))
    position = position_manager.positions.get(position_id)
    trailing_id = (position or {}).get("native_trailing_stop_order_id") if isinstance(position, dict) else None

    results.append(
        EdgeCaseResult(
            name=f"{side}:baseline_native_ids_present",
            passed=bool(hard_id and trailing_id),
            details={"position_id": position_id, "hard_stop_order_id": hard_id, "trailing_order_id": trailing_id},
        )
    )

    # B1: Partial close / size-change -> conditional qty resync.
    try:
        raw_positions = client.get_bingx_positions(resolved_symbol)
        native_symbol = str(getattr(client, "_get_bingx_native_symbol", lambda s: s)(resolved_symbol))
        exchange_amt = _extract_position_amt(raw_positions, native_symbol=native_symbol, position_side=position_side)
        before_local_amt = _safe_float((position_manager.positions.get(position_id) or {}).get("amount")) or 0.0

        # Close the minimum allowed quantity (partial close) rather than "half" which may round to 0.
        half_amt = float(min_qty) if min_qty > 0 else (exchange_amt / 2.0 if exchange_amt > 0 else before_local_amt / 2.0)
        half_amt = max(half_amt, 0.0)
        try:
            half_amt = _safe_float(exchange.amount_to_precision(resolved_symbol, half_amt))
        except Exception:
            pass
        if half_amt <= 0 or (exchange_amt > 0 and half_amt >= exchange_amt):
            raise RuntimeError(f"Invalid partial close amount computed: half_amt={half_amt} exchange_amt={exchange_amt}")

        partial_close_req = {
            "symbol": resolved_symbol,
            "side": close_side,
            "amount": half_amt,
            "exchange": "bingx",
            "execution_params": {"reduceOnly": True, "positionSide": position_side},
        }
        partial_close_result = await order_manager.place_order(partial_close_req, execution_algo="market")
        _write_jsonl(jsonl_path, "order.partial_close.result", {"side": side, "half_amt": half_amt, "result": partial_close_result})
        if not partial_close_result.get("success"):
            raise RuntimeError(f"Partial close failed: {partial_close_result.get('reason')}")

        # Give exchange a moment, then force reconcile + sync by calling manage_position_exits().
        await asyncio.sleep(3.0)
        (position_manager.positions.get(position_id) or {})["native_exit_reconcile_last_ts"] = 0.0
        _ = await position_manager.manage_position_exits(position_id)
        await asyncio.sleep(1.0)

        position_after = position_manager.positions.get(position_id)
        after_local_amt = _safe_float((position_after or {}).get("amount")) or 0.0
        hard_qty = _safe_float((position_after or {}).get("native_hard_stop_qty")) or 0.0
        trailing_qty = _safe_float((position_after or {}).get("native_trailing_stop_qty")) or 0.0

        passed = after_local_amt > 0 and hard_qty > 0 and trailing_qty > 0 and abs(hard_qty - after_local_amt) < 1e-12 and abs(trailing_qty - after_local_amt) < 1e-12
        results.append(
            EdgeCaseResult(
                name=f"{side}:partial_close_resync_qty",
                passed=passed,
                details={
                    "before_local_amt": before_local_amt,
                    "exchange_amt_before": exchange_amt,
                    "half_amt_closed": half_amt,
                    "after_local_amt": after_local_amt,
                    "hard_qty": hard_qty,
                    "trailing_qty": trailing_qty,
                },
            )
        )
    except Exception as exc:
        results.append(EdgeCaseResult(name=f"{side}:partial_close_resync_qty", passed=False, details={"error": repr(exc)}))

    # B3: Cancel error tolerance (idempotent).
    try:
        pos = position_manager.positions.get(position_id)
        if not isinstance(pos, dict):
            raise RuntimeError("Position missing before cancel test")

        await position_manager._cancel_bingx_native_conditional_orders(pos, context="edge_cancel_1")
        await position_manager._cancel_bingx_native_conditional_orders(pos, context="edge_cancel_2")
        results.append(EdgeCaseResult(name=f"{side}:cancel_idempotent", passed=True, details={}))
    except Exception as exc:
        results.append(EdgeCaseResult(name=f"{side}:cancel_idempotent", passed=False, details={"error": repr(exc)}))

    # B2: Race - exchange already flat vs engine close.
    try:
        # Close any remaining position on exchange, but do not close locally yet.
        raw_positions = client.get_bingx_positions(resolved_symbol)
        native_symbol = str(getattr(client, "_get_bingx_native_symbol", lambda s: s)(resolved_symbol))
        exchange_amt = _extract_position_amt(raw_positions, native_symbol=native_symbol, position_side=position_side)
        close_amt = exchange_amt if exchange_amt > 0 else _safe_float((position_manager.positions.get(position_id) or {}).get("amount")) or 0.0
        if close_amt <= 0:
            raise RuntimeError("No remaining size found for exchange-close race setup")

        close_req = {
            "symbol": resolved_symbol,
            "side": close_side,
            "amount": close_amt,
            "exchange": "bingx",
            "execution_params": {"reduceOnly": True, "positionSide": position_side},
        }
        close_result = await order_manager.place_order(close_req, execution_algo="market")
        _write_jsonl(jsonl_path, "order.exchange_close_for_race.result", {"side": side, "result": close_result})
        if not close_result.get("success"):
            raise RuntimeError(f"Exchange close for race failed: {close_result.get('reason')}")

        await asyncio.sleep(3.0)

        # Engine should preflight-check and skip placing a second market close.
        called = {"place_order": 0}

        class _FailIfCalledOrderManager:
            async def place_order(self, order_request, execution_algo="market"):
                called["place_order"] += 1
                raise RuntimeError("order_manager.place_order should not be called when exchange already flat")

        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine.order_manager = _FailIfCalledOrderManager()
        engine.position_manager = position_manager
        engine.active_positions = {position_id: position_manager.positions.get(position_id)}

        exit_out = await engine._execute_position_exit(position_id, {"exit_reason": "manual"})
        _write_jsonl(jsonl_path, "engine.exit.result", {"side": side, "exit_out": exit_out, "calls": called})

        passed = exit_out.get("skip_market_exit") is True and exit_out.get("preflight_skip_market_exit") is True and called["place_order"] == 0
        results.append(EdgeCaseResult(name=f"{side}:race_preflight_skip", passed=passed, details={"exit_out": exit_out, "calls": called}))
    except Exception as exc:
        results.append(EdgeCaseResult(name=f"{side}:race_preflight_skip", passed=False, details={"error": repr(exc)}))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="BingX VST Stage-2 edge-case runner")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument("--min-notional-usdt", type=float, default=0.0)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--trailing-distance", type=float, default=0.002)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "diagnostics" / "vst"))
    ap.add_argument("--env-file", default="bearish-bot.env.local")
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
    json_path = out_dir / "stage2_edge_cases_matrix.json"
    jsonl_path = out_dir / "stage2_edge_cases_events.jsonl"
    log_path = out_dir / "stage2_edge_cases.log"
    _configure_file_logging(log_path)

    sides = ["long", "short"] if args.side == "both" else [args.side]
    all_results: List[EdgeCaseResult] = []
    for s in sides:
        all_results.extend(
            asyncio.run(
                _run_one_side(
                    side=s,
                    symbol=args.symbol,
                    notional_usdt=float(args.notional_usdt),
                    min_notional_usdt=float(args.min_notional_usdt),
                    trailing_distance=float(args.trailing_distance),
                    env_file=str(args.env_file or ""),
                    jsonl_path=jsonl_path,
                )
            )
        )

    payload = {
        "ts": utc_now(),
        "results": [asdict(r) for r in all_results],
        "passed": all(r.passed for r in all_results),
    }
    json_path.write_text(json.dumps(redacted(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {log_path}")

    failed = [r for r in all_results if not r.passed]
    if failed:
        print("FAIL:")
        for r in failed:
            print(f"- {r.name}: {r.details.get('error') or r.details}")
        return 2

    print("PASS: all edge cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
