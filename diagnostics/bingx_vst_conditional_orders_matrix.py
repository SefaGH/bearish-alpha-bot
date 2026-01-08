"""
BingX VST/sandbox validation for conditional orders (swap):
- STOP_MARKET (via stopLossPrice)
- TRIGGER_MARKET (via triggerPrice)
- TRAILING_STOP_MARKET (via trailingPercent/priceRate)

Validates:
1) reduceOnly + positionSide behavior
2) oversize quantity behavior (reject vs flip risk)
3) coexistence (hard stop + trailing simultaneously)
4) cancel + (emulated) edit/cancelReplace reliability

Requirements:
- ccxt installed
- env vars: BINGX_KEY, BINGX_SECRET

This script makes real API calls against BingX VST (sandbox).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redacted(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in {"apiKey", "secret", "password", "passphrase"}:
                out[k] = "***"
            else:
                out[k] = redacted(v)
        return out
    if isinstance(obj, list):
        return [redacted(x) for x in obj]
    return obj


@dataclass
class CheckResult:
    check: str
    passed: bool
    notes: str
    evidence: Dict[str, Any]


def _as_boolish(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def check_reduce_only_echo(order: Dict[str, Any], requested: Optional[bool]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    BingX sometimes rejects or ignores reduceOnly depending on position mode.
    This check verifies whether the exchange echoed reduceOnly=true/false.
    """
    info = (order or {}).get("info") or {}
    echoed = info.get("reduceOnly")
    echoed_bool = _as_boolish(echoed)
    ok = True
    note = "Exchange echoed reduceOnly as expected."
    if requested is not None and echoed_bool is not None and echoed_bool != requested:
        ok = False
        note = f"Exchange echoed reduceOnly={echoed_bool} but requested reduceOnly={requested}."
    evidence = {"requested_reduceOnly": requested, "echoed_reduceOnly_raw": echoed, "echoed_reduceOnly": echoed_bool}
    return ok, note, evidence


class MatrixLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_tag = ts
        self.jsonl_path = os.path.join(out_dir, f"bingx_vst_conditional_orders_{ts}.jsonl")
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")

    def write_event(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": utc_now(), "event": event, **payload}
        self._fh.write(json.dumps(redacted(rec), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def load_env_file_if_present(path: str) -> bool:
    """
    Minimal .env loader (KEY=VALUE per line, ignores comments).
    Returns True if file existed (regardless of whether it set anything).
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and (os.getenv(k) is None):
                    os.environ[k] = v
        return True
    except Exception:
        return True


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def get_best_swap_symbol(exchange, requested_symbol: str) -> str:
    exchange.load_markets()
    if requested_symbol in exchange.markets:
        return requested_symbol
    # try common variants
    variants = [
        requested_symbol.replace(":USDT", ""),
        requested_symbol.replace(":USDT", "").replace("/", "/"),
        "BTC/USDT:USDT",
        "BTC/USDT",
    ]
    for v in variants:
        if v in exchange.markets:
            return v
    raise RuntimeError(
        f"Symbol not found in loaded markets: {requested_symbol}. "
        f"Examples: {sorted([s for s in exchange.markets.keys() if 'BTC' in s][:10])}"
    )


def fetch_last_price(exchange, symbol: str) -> float:
    t = exchange.fetch_ticker(symbol)
    last = t.get("last")
    if last is None:
        # fallback
        last = t.get("close") or t.get("info", {}).get("lastPrice")
    return safe_float(last)


def fetch_open_position(exchange, symbol: str) -> Dict[str, Any]:
    # returns unified position dict if possible; else empty
    try:
        positions = exchange.fetch_positions([symbol])
        for p in positions or []:
            if p.get("symbol") == symbol:
                contracts = safe_float(p.get("contracts") or p.get("contractSize") or p.get("info", {}).get("positionAmt"))
                if abs(contracts) > 0:
                    return p
    except Exception:
        pass
    return {}


def close_any_position(exchange, symbol: str, matrix_log: MatrixLogger) -> None:
    try:
        pos = fetch_open_position(exchange, symbol)
        if not pos:
            return
        matrix_log.write_event("cleanup.position_found", {"position": pos})
        # try closePosition (one-click) if available
        try:
            resp = exchange.close_position(symbol, params={})
            matrix_log.write_event("cleanup.close_position", {"response": resp})
            return
        except Exception as e:
            matrix_log.write_event("cleanup.close_position_failed", {"error": repr(e)})
    except Exception:
        return


def cancel_all_open_orders(exchange, symbol: str, matrix_log: MatrixLogger) -> None:
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        matrix_log.write_event("cleanup.open_orders", {"open_orders": open_orders})
        for o in open_orders or []:
            try:
                resp = exchange.cancel_order(o.get("id"), symbol)
                matrix_log.write_event("cleanup.cancel_order", {"order_id": o.get("id"), "response": resp})
            except Exception as e:
                matrix_log.write_event("cleanup.cancel_order_failed", {"order_id": o.get("id"), "error": repr(e)})
    except Exception as e:
        matrix_log.write_event("cleanup.fetch_open_orders_failed", {"error": repr(e)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--out-dir", default=os.path.join("diagnostics", "vst"))
    ap.add_argument("--notional-usdt", type=float, default=5.0)
    ap.add_argument(
        "--min-notional-usdt",
        type=float,
        default=0.0,
        help="Optional floor for notional sizing (helps satisfy min amount on high-priced assets).",
    )
    ap.add_argument("--leverage", type=int, default=1)
    ap.add_argument("--skip-open-position", action="store_true", help="Only validates order placement/cancel without opening a position.")
    ap.add_argument(
        "--scenario",
        choices=["long", "short", "orphan"],
        default="long",
        help="Which validation scenario to run.",
    )
    ap.add_argument(
        "--ensure-hedged",
        action="store_true",
        help="Best-effort: set hedged/dualSidePosition=true before running (requires permissions).",
    )
    args = ap.parse_args()

    import ccxt  # local import to keep script importable without deps

    env_loaded = load_env_file_if_present(".env")
    key = require_env("BINGX_KEY")
    secret = require_env("BINGX_SECRET")

    matrix_log = MatrixLogger(args.out_dir)
    results: List[CheckResult] = []

    exchange = ccxt.bingx(
        {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    exchange.set_sandbox_mode(True)  # VST

    matrix_log.write_event(
        "session.start",
        {
            "ccxt_version": getattr(ccxt, "__version__", "unknown"),
            "exchange_id": exchange.id,
            "sandbox": True,
            "env_loaded_from_dotenv": env_loaded,
            "urls": exchange.urls,
        },
    )

    symbol = get_best_swap_symbol(exchange, args.symbol)
    matrix_log.write_event("symbol.resolved", {"requested": args.symbol, "resolved": symbol})

    # Determine current position mode (one-way vs hedged)
    position_mode: Optional[Dict[str, Any]] = None
    try:
        position_mode = exchange.fetch_position_mode(symbol)
        matrix_log.write_event("position_mode", {"position_mode": position_mode})
    except Exception as e:
        matrix_log.write_event("position_mode_failed", {"error": repr(e)})

    if args.ensure_hedged:
        try:
            resp = exchange.set_position_mode(True, symbol)
            matrix_log.write_event("position_mode.set_hedged", {"response": resp})
            try:
                position_mode = exchange.fetch_position_mode(symbol)
                matrix_log.write_event("position_mode.after_set", {"position_mode": position_mode})
            except Exception:
                pass
        except Exception as e:
            matrix_log.write_event("position_mode.set_hedged_failed", {"error": repr(e)})

    last = fetch_last_price(exchange, symbol)
    if last <= 0:
        raise RuntimeError("Could not fetch last price")
    matrix_log.write_event("price.last", {"symbol": symbol, "last": last})

    # Try to set leverage (best-effort; may require extra params)
    try:
        resp = exchange.set_leverage(args.leverage, symbol)
        matrix_log.write_event("leverage.set", {"leverage": args.leverage, "response": resp})
    except Exception as e:
        matrix_log.write_event("leverage.set_failed", {"leverage": args.leverage, "error": repr(e)})

    # Compute a safe base amount (respect min amount + precision)
    market = exchange.market(symbol)
    min_amount = safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"))
    precision_amount = safe_float(((market.get("precision") or {}).get("amount")))

    # If precision is given (e.g., 0.0001), treat it as an absolute minimum step floor.
    # CCXT enforces: amount >= precision['amount'] for some exchanges/markets.
    step_floor = precision_amount if precision_amount > 0 else 0.0
    hard_floor = max(min_amount, step_floor)

    effective_notional = max(args.notional_usdt, args.min_notional_usdt or 0.0)
    raw_amount = effective_notional / last
    raw_amount = max(raw_amount, hard_floor)

    # Round using exchange rules; if rounding dips below floor, bump to floor.
    try:
        rounded_str = exchange.amount_to_precision(symbol, raw_amount)
        amount = safe_float(rounded_str)
    except Exception:
        amount = float(f"{raw_amount:.8f}")

    if amount < hard_floor:
        try:
            amount = safe_float(exchange.amount_to_precision(symbol, hard_floor))
        except Exception:
            amount = hard_floor

    matrix_log.write_event(
        "sizing",
        {
            "notional_usdt": args.notional_usdt,
            "min_notional_usdt": args.min_notional_usdt,
            "effective_notional_usdt": effective_notional,
            "raw_amount": raw_amount,
            "min_amount": min_amount,
            "precision_amount": precision_amount,
            "hard_floor": hard_floor,
            "amount": amount,
        },
    )

    stop_order_id = None
    trigger_order_id = None
    trailing_order_id = None

    try:
        cancel_all_open_orders(exchange, symbol, matrix_log)
        close_any_position(exchange, symbol, matrix_log)

        if not args.skip_open_position and args.scenario in {"long", "short", "orphan"}:
            # Open a tiny position in the requested direction.
            open_side = "buy" if args.scenario in {"long", "orphan"} else "sell"
            open_params: Dict[str, Any] = {"reduceOnly": False}
            matrix_log.write_event(
                f"open.{args.scenario}.request",
                {"symbol": symbol, "side": open_side, "amount": amount, "params": open_params},
            )
            open_order = exchange.create_order(symbol, "market", open_side, amount, None, open_params)
            matrix_log.write_event(f"open.{args.scenario}.response", {"order": open_order})
            time.sleep(1.0)

        pos = fetch_open_position(exchange, symbol)
        matrix_log.write_event("position.after_open", {"position": pos})

        if args.scenario == "long":
            close_side = "sell"
            position_side = "LONG"
            stop_price = last * 0.95  # far away
            trigger_price = last * 0.96
            trailing_percent = 0.2
        elif args.scenario == "short":
            close_side = "buy"
            position_side = "SHORT"
            stop_price = last * 1.05  # above entry for shorts
            trigger_price = last * 1.04
            trailing_percent = 0.2
        else:  # orphan scenario uses LONG leg first
            close_side = "sell"
            position_side = "LONG"
            stop_price = last * 0.80  # far away; should not trigger soon
            trigger_price = last * 0.85
            trailing_percent = 0.2

        # STOP_MARKET (via stopLossPrice mapping)
        stop_params = {"stopLossPrice": stop_price, "workingType": "MARK_PRICE", "reduceOnly": True, "positionSide": position_side}
        matrix_log.write_event(
            "stop_market.request",
            {"symbol": symbol, "side": close_side, "amount": amount, "stopLossPrice": stop_price, "params": stop_params},
        )
        stop_order = exchange.create_order(symbol, "market", close_side, amount, None, stop_params)
        stop_order_id = stop_order.get("id")
        matrix_log.write_event("stop_market.response", {"order": stop_order})
        ro_ok, ro_note, ro_ev = check_reduce_only_echo(stop_order, True)
        results.append(
            CheckResult(
                check="STOP_MARKET placement (stopLossPrice + workingType + reduceOnly)",
                passed=bool(stop_order_id),
                notes="Expected: accepted, open order visible.",
                evidence={"order_id": stop_order_id, "order": stop_order, "params": stop_params},
            )
        )
        results.append(
            CheckResult(
                check="STOP_MARKET reduceOnly echoed by exchange",
                passed=ro_ok,
                notes=ro_note,
                evidence={"order_id": stop_order_id, **ro_ev, "positionSide": (stop_order.get("info") or {}).get("positionSide")},
            )
        )

        # TRIGGER_MARKET (generic trigger order)
        trigger_params = {"triggerPrice": trigger_price, "workingType": "MARK_PRICE", "reduceOnly": True, "positionSide": position_side}
        matrix_log.write_event(
            "trigger_market.request",
            {"symbol": symbol, "side": close_side, "amount": amount, "triggerPrice": trigger_price, "params": trigger_params},
        )
        trigger_order = exchange.create_trigger_order(symbol, "market", close_side, amount, None, trigger_price, trigger_params)
        trigger_order_id = trigger_order.get("id")
        matrix_log.write_event("trigger_market.response", {"order": trigger_order})
        ro_ok, ro_note, ro_ev = check_reduce_only_echo(trigger_order, True)
        results.append(
            CheckResult(
                check="TRIGGER_MARKET placement (createTriggerOrder + workingType + reduceOnly)",
                passed=bool(trigger_order_id),
                notes="Expected: accepted, open order visible.",
                evidence={"order_id": trigger_order_id, "order": trigger_order, "params": trigger_params},
            )
        )
        results.append(
            CheckResult(
                check="TRIGGER_MARKET reduceOnly echoed by exchange",
                passed=ro_ok,
                notes=ro_note,
                evidence={"order_id": trigger_order_id, **ro_ev, "positionSide": (trigger_order.get("info") or {}).get("positionSide")},
            )
        )

        # TRAILING_STOP_MARKET (percent). CCXT converts trailingPercent% -> priceRate.
        trailing_params = {"reduceOnly": True, "trailingType": "TRAILING_STOP_MARKET", "positionSide": position_side}
        matrix_log.write_event(
            "trailing_percent.request",
            {
                "symbol": symbol,
                "side": close_side,
                "amount": amount,
                "trailingPercent": trailing_percent,
                "params": trailing_params,
            },
        )
        trailing_order = exchange.create_trailing_percent_order(
            symbol, "market", close_side, amount, None, trailing_percent, None, trailing_params
        )
        trailing_order_id = trailing_order.get("id")
        matrix_log.write_event("trailing_percent.response", {"order": trailing_order})
        ro_ok, ro_note, ro_ev = check_reduce_only_echo(trailing_order, True)
        results.append(
            CheckResult(
                check="TRAILING_STOP_MARKET placement (createTrailingPercentOrder + reduceOnly)",
                passed=bool(trailing_order_id),
                notes="Expected: accepted; validate coexistence with hard stop.",
                evidence={"order_id": trailing_order_id, "order": trailing_order, "params": trailing_params},
            )
        )
        results.append(
            CheckResult(
                check="TRAILING_STOP_MARKET reduceOnly echoed by exchange",
                passed=ro_ok,
                notes=ro_note,
                evidence={"order_id": trailing_order_id, **ro_ev, "positionSide": (trailing_order.get("info") or {}).get("positionSide")},
            )
        )

        # Coexistence check: all three should appear in open orders.
        open_orders = exchange.fetch_open_orders(symbol)
        matrix_log.write_event("open_orders.after_placement", {"open_orders": open_orders})
        ids = {o.get("id") for o in open_orders or []}
        coexist_ok = all(x in ids for x in [stop_order_id, trigger_order_id, trailing_order_id] if x)
        results.append(
            CheckResult(
                check="Coexistence (hard stop + trigger + trailing simultaneously)",
                passed=coexist_ok,
                notes="If FAIL, BingX may limit simultaneous conditional orders per position/symbol.",
                evidence={"expected_ids": [stop_order_id, trigger_order_id, trailing_order_id], "open_order_ids": sorted(list(ids))},
            )
        )

        # Oversize quantity behavior (should be rejected; if accepted, treat as risk).
        oversize_amount = amount * 3.0
        oversize_amount = float(f"{oversize_amount:.6f}")
        oversize_stop_price = (last * 0.94) if args.scenario != "short" else (last * 1.06)
        oversize_params = {
            "stopLossPrice": oversize_stop_price,
            "workingType": "MARK_PRICE",
            "reduceOnly": True,
            "positionSide": position_side,
        }
        matrix_log.write_event(
            "oversize_stop_market.request",
            {
                "symbol": symbol,
                "side": close_side,
                "amount": oversize_amount,
                "stopLossPrice": oversize_stop_price,
                "params": oversize_params,
            },
        )
        oversize_ok = False
        oversize_resp: Any = None
        oversize_err: Optional[str] = None
        try:
            oversize_resp = exchange.create_order(symbol, "market", close_side, oversize_amount, None, oversize_params)
            oversize_ok = True
            matrix_log.write_event("oversize_stop_market.response", {"order": oversize_resp})
        except Exception as e:
            oversize_err = repr(e)
            matrix_log.write_event("oversize_stop_market.rejected", {"error": oversize_err})
        # We *prefer* reject. Accepting is a red flag (could flip if triggered).
        results.append(
            CheckResult(
                check="Oversize STOP_MARKET reduce-only behavior",
                passed=not oversize_ok,
                notes="PASS = rejected (safer). FAIL = accepted (requires strict sizing guardrails).",
                evidence={"oversize_amount": oversize_amount, "accepted": oversize_ok, "response": oversize_resp, "error": oversize_err},
            )
        )

        # Cancel reliability (best effort): cancel the created orders.
        cancel_ok = True
        canceled: Dict[str, Any] = {}
        for name, oid in [("stop", stop_order_id), ("trigger", trigger_order_id), ("trailing", trailing_order_id)]:
            if not oid:
                continue
            try:
                resp = exchange.cancel_order(oid, symbol)
                canceled[name] = resp
                matrix_log.write_event("cancel.ok", {"name": name, "order_id": oid, "response": resp})
            except Exception as e:
                cancel_ok = False
                matrix_log.write_event("cancel.failed", {"name": name, "order_id": oid, "error": repr(e)})
        results.append(
            CheckResult(
                check="Cancel reliability (cancel_order)",
                passed=cancel_ok,
                notes="Expected: all cancels succeed; intermittent failures indicate retry/idempotency needs.",
                evidence={"canceled": canceled},
            )
        )

        if args.scenario == "orphan":
            # Orphan safety checks: do conditional orders auto-cancel when position is closed?
            # 1) Open LONG already done above. Create three close orders far away already done above.
            # 2) Close position (market close).
            try:
                matrix_log.write_event("orphan.close_position.request", {"symbol": symbol})
                resp = exchange.close_position(symbol, params={})
                matrix_log.write_event("orphan.close_position.response", {"response": resp})
            except Exception as e:
                matrix_log.write_event("orphan.close_position.failed", {"error": repr(e)})

            time.sleep(1.0)

            # 3) Observe whether close orders remain open.
            open_orders = exchange.fetch_open_orders(symbol)
            matrix_log.write_event("orphan.open_orders.after_close", {"open_orders": open_orders})
            ids_after = {o.get("id") for o in open_orders or []}
            still_open = [x for x in [stop_order_id, trigger_order_id, trailing_order_id] if x and x in ids_after]
            results.append(
                CheckResult(
                    check="Orphan behavior: orders auto-canceled on position close",
                    passed=len(still_open) == 0,
                    notes="PASS = exchange removed them; FAIL = orders remain open (must cancel on close).",
                    evidence={"still_open_order_ids": still_open, "open_order_ids": sorted(list(ids_after))},
                )
            )

            # 4) If any remain open, attempt to force-trigger by editing stopPrice to a trivially satisfied value.
            # We do this ONLY as a controlled test, then immediately cleanup.
            # Note: In hedge mode, BingX may reject reduceOnly in cancelReplace, so we omit reduceOnly here.
            force_attempts: List[Dict[str, Any]] = []
            for oid, kind in [(stop_order_id, "STOP_MARKET"), (trigger_order_id, "TRIGGER_MARKET")]:
                if not oid or oid not in ids_after:
                    continue
                try:
                    # For a SELL close, use a stopPrice far above current to likely satisfy immediately for <= semantics;
                    # if semantics are >=, then use far below. We try both by cancel+create a new trigger order.
                    # Since edit semantics vary, we record outcomes and then cleanup.
                    hi = last * 1.50
                    lo = last * 0.50
                    for sp in (hi, lo):
                        params = {"stopPrice": sp, "workingType": "MARK_PRICE", "positionSide": position_side}
                        try:
                            # editOrder requires type/side/amount/price; for market variants, use market + price=None.
                            edited = exchange.edit_order(oid, symbol, "market", close_side, amount, None, params)
                            matrix_log.write_event("orphan.force_edit.response", {"order_id": oid, "stopPrice": sp, "edited": edited})
                            force_attempts.append({"order_id": oid, "stopPrice": sp, "edited": edited})
                            break
                        except Exception as ee:
                            matrix_log.write_event("orphan.force_edit.failed", {"order_id": oid, "stopPrice": sp, "error": repr(ee)})
                except Exception as e:
                    matrix_log.write_event("orphan.force_edit.exception", {"order_id": oid, "error": repr(e)})
            # After force attempts, check whether any position was opened.
            time.sleep(1.0)
            pos_after = fetch_open_position(exchange, symbol)
            matrix_log.write_event("orphan.position.after_force", {"position": pos_after})
            results.append(
                CheckResult(
                    check="Orphan behavior: forcing trigger does not open new exposure",
                    passed=not bool(pos_after),
                    notes="PASS = no position opened; FAIL = exposure reopened (must cancel on close, and use closePosition semantics if available).",
                    evidence={"force_attempts": force_attempts, "position_after_force": pos_after},
                )
            )

        # Cancel/replace reliability (editOrder emulated): place a trigger limit then edit.
        # We avoid actually placing a limit that could execute by setting a far price.
        edit_passed = False
        edit_evidence: Dict[str, Any] = {}
        try:
            base_trigger = last * 1.10
            base_price = last * 1.09
            p = {"triggerPrice": base_trigger, "workingType": "MARK_PRICE", "reduceOnly": True}
            o1 = exchange.create_order(symbol, "limit", "sell", amount, base_price, p)
            matrix_log.write_event("edit.base_order", {"order": o1})
            new_trigger = last * 1.11
            new_price = last * 1.095
            p2 = {"triggerPrice": new_trigger, "workingType": "MARK_PRICE", "reduceOnly": True}
            o2 = exchange.edit_order(o1.get("id"), symbol, "limit", "sell", amount, new_price, p2)
            matrix_log.write_event("edit.response", {"order": o2})
            # Cleanup
            try:
                exchange.cancel_order(o2.get("id") or o1.get("id"), symbol)
            except Exception:
                pass
            info = (o2 or {}).get("info") or {}
            replace_ok = _as_boolish(info.get("replaceResult"))
            # For BingX, editOrder is "cancel then place new". We treat it as PASS only if replaceResult is true.
            edit_passed = bool(replace_ok)
            edit_evidence = {"base": o1, "edited": o2, "replaceResult": info.get("replaceResult"), "replaceMsg": info.get("replaceMsg")}
        except Exception as e:
            edit_evidence = {"error": repr(e)}
            matrix_log.write_event("edit.failed", edit_evidence)
        results.append(
            CheckResult(
                check="Cancel/replace (edit_order emulated)",
                passed=edit_passed,
                notes="PASS requires replaceResult=true; otherwise treat as not supported with the given params/mode.",
                evidence=edit_evidence,
            )
        )

        # Secondary check: edit/cancelReplace without reduceOnly (useful in hedge mode).
        edit2_passed = False
        edit2_evidence: Dict[str, Any] = {}
        try:
            base_trigger = last * 1.12
            base_price = last * 1.11
            p = {"triggerPrice": base_trigger, "workingType": "MARK_PRICE"}  # no reduceOnly
            o1 = exchange.create_order(symbol, "limit", "sell", amount, base_price, p)
            matrix_log.write_event("edit2.base_order", {"order": o1})
            new_trigger = last * 1.13
            new_price = last * 1.115
            p2 = {"triggerPrice": new_trigger, "workingType": "MARK_PRICE"}  # no reduceOnly
            o2 = exchange.edit_order(o1.get("id"), symbol, "limit", "sell", amount, new_price, p2)
            matrix_log.write_event("edit2.response", {"order": o2})
            try:
                exchange.cancel_order(o2.get("id") or o1.get("id"), symbol)
            except Exception:
                pass
            info = (o2 or {}).get("info") or {}
            replace_ok = _as_boolish(info.get("replaceResult"))
            edit2_passed = bool(replace_ok)
            edit2_evidence = {"base": o1, "edited": o2, "replaceResult": info.get("replaceResult"), "replaceMsg": info.get("replaceMsg")}
        except Exception as e:
            edit2_evidence = {"error": repr(e)}
            matrix_log.write_event("edit2.failed", edit2_evidence)
        results.append(
            CheckResult(
                check="Cancel/replace without reduceOnly (edit_order emulated)",
                passed=edit2_passed,
                notes="In hedge mode, BingX may reject reduceOnly; this checks whether cancelReplace works without it.",
                evidence=edit2_evidence,
            )
        )

    finally:
        # Ensure cleanup
        try:
            cancel_all_open_orders(exchange, symbol, matrix_log)
            close_any_position(exchange, symbol, matrix_log)
        finally:
            matrix_log.write_event("session.end", {"results_count": len(results)})
            matrix_log.close()

    # Write summary artifacts
    summary = {
        "ts": utc_now(),
        "symbol": symbol,
        "sandbox": True,
        "position_mode": position_mode,
        "scenario": args.scenario,
        "results": [asdict(r) for r in results],
        "jsonl_path": matrix_log.jsonl_path,
    }
    # Write a unique per-run summary file to preserve artifacts across scenarios.
    summary_name = f"bingx_vst_matrix_summary_{args.scenario}_{matrix_log.run_tag}.json"
    summary_path = os.path.join(args.out_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(redacted(summary), f, ensure_ascii=False, indent=2)

    # Also keep a stable "latest" pointer file for convenience.
    latest_path = os.path.join(args.out_dir, "bingx_vst_matrix_summary_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump({"latest": summary_name, "ts": summary["ts"], "scenario": args.scenario}, f, ensure_ascii=False, indent=2)

    # Console output (minimal)
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {latest_path}")
    print(f"Wrote: {matrix_log.jsonl_path}")
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'} | {r.check} | {r.notes}")

    # Non-zero exit on failures
    failed = [r for r in results if not r.passed]
    return 2 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
