"""
Advanced Position Management System.
Comprehensive position lifecycle management.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
import math
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from core.logger import get_current_run_id

# Triple-fallback import strategy for maximum compatibility:
# 1. Direct utils import (when src/ is on sys.path)
# 2. Absolute src.utils import (when repo root is on sys.path)
# 3. Relative import (when imported as package module)
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import (
        calculate_unrealized_pnl,
        calculate_realized_pnl,
        calculate_pnl_percentage
    )
except ModuleNotFoundError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import (
            calculate_unrealized_pnl,
            calculate_realized_pnl,
            calculate_pnl_percentage
        )
    except ModuleNotFoundError as e:
        # Option 3: Relative import (package context)
        if e.name in ('src', 'src.utils', 'src.utils.pnl_calculator'):
            from ..utils.pnl_calculator import (
                calculate_unrealized_pnl,
                calculate_realized_pnl,
                calculate_pnl_percentage
            )
        else:
            # Unknown module missing, re-raise
            raise

logger = logging.getLogger(__name__)


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _read_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


_SHUTDOWN_CLOSE_MAX_ATTEMPTS = max(1, _read_int_env("POSITION_CLOSE_RETRY_LIMIT", 2))
_SHUTDOWN_CLOSE_BACKOFF = max(0.0, _read_float_env("POSITION_CLOSE_RETRY_BACKOFF", 2.0))


class PositionStateStore:
    """
    File-backed snapshot store for open positions.

    CRITICAL: Uses atomic write to prevent corruption on crashes:
      - write to positions_snapshot.tmp
      - flush + fsync
      - os.replace -> positions_snapshot.json

    The store is intentionally minimal (open positions only) and JSON-only.
    """

    SNAPSHOT_VERSION = 1

    def __init__(self, snapshot_path: Path, logger: Optional[logging.Logger] = None) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.tmp_path = self.snapshot_path.with_suffix(".tmp")
        self.backup_path = self.snapshot_path.with_suffix(".bak.json")
        self._logger = logger or logging.getLogger(__name__)

    def load(self) -> Dict[str, Any]:
        """Load snapshot file; returns an empty payload if missing/corrupt."""
        if not self.snapshot_path.exists():
            return {"version": self.SNAPSHOT_VERSION, "positions": {}}

        try:
            with self.snapshot_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                raise ValueError("snapshot payload not a dict")
            version = payload.get("version", self.SNAPSHOT_VERSION)
            positions = payload.get("positions", {})
            if not isinstance(positions, dict):
                positions = {}
            return {"version": version, "positions": positions}
        except Exception as exc:
            self._logger.warning("Failed to load positions snapshot (%s): %s", self.snapshot_path, exc)
            return {"version": self.SNAPSHOT_VERSION, "positions": {}}

    def save(self, positions: Dict[str, Dict[str, Any]]) -> bool:
        """Persist open positions snapshot with atomic replace semantics."""
        try:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._logger.warning("Failed to create snapshot directory %s: %s", self.snapshot_path.parent, exc)
            return False

        payload = {
            "version": self.SNAPSHOT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "positions": {str(pid): self._jsonify(pos) for pid, pos in (positions or {}).items()},
        }

        data = None
        try:
            data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as exc:
            self._logger.warning("Failed to serialize positions snapshot payload: %s", exc)
            return False

        # Best-effort backup of the last known-good snapshot.
        if self.snapshot_path.exists():
            try:
                shutil.copyfile(self.snapshot_path, self.backup_path)
            except Exception:
                pass

        # Primary atomic write: deterministic tmp filename (positions_snapshot.tmp).
        try:
            self._atomic_write(self.tmp_path, self.snapshot_path, data)
            return True
        except Exception as exc:
            # Fallback: unique tmp file in same directory (still atomic replace).
            self._logger.warning("Primary snapshot atomic write failed (%s): %s", self.tmp_path, exc)
            tmp_name = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    delete=False,
                    dir=str(self.snapshot_path.parent),
                    prefix=self.snapshot_path.stem + ".",
                    suffix=".tmp",
                ) as tmp_stream:
                    tmp_stream.write(data)
                    tmp_stream.flush()
                    os.fsync(tmp_stream.fileno())
                    tmp_name = tmp_stream.name
                os.replace(tmp_name, self.snapshot_path)
                return True
            except Exception as exc2:
                self._logger.error("Fallback snapshot write failed: %s", exc2, exc_info=True)
                try:
                    if tmp_name and os.path.exists(tmp_name):
                        os.remove(tmp_name)
                except Exception:
                    pass
                return False

    @staticmethod
    def _atomic_write(tmp_path: Path, final_path: Path, data: str) -> None:
        tmp_path = Path(tmp_path)
        final_path = Path(final_path)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tmp_path.open("w", encoding="utf-8") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, final_path)
        finally:
            # If replace failed, try to clean up the tmp file (best effort).
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def _jsonify(cls, obj: Any) -> Any:
        """Best-effort conversion of common runtime objects to JSON-safe values."""
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, datetime):
            return obj.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                try:
                    key = str(k)
                except Exception:
                    key = repr(k)
                out[key] = cls._jsonify(v)
            return out
        if isinstance(obj, (list, tuple, set)):
            return [cls._jsonify(v) for v in obj]
        # Fallback: stringify unknown objects (keeps snapshot writes robust).
        try:
            return str(obj)
        except Exception:
            return repr(obj)

class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = 'open'
    CLOSED = 'closed'
    PARTIALLY_CLOSED = 'partially_closed'
    PENDING_CLOSE = 'pending_close'


class PositionManagerPnlProvider:
    """PnL view backed directly by AdvancedPositionManager positions."""

    def __init__(self, position_manager: "AdvancedPositionManager"):
        self.position_manager = position_manager

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_unrealized_pnl_pct(self, position: Dict[str, Any]) -> Optional[float]:
        """
        Extract and normalize unrealized PnL percentage from a position snapshot.

        Normalization: `_pct` fields coming from PositionManager computations are stored
        as human-readable percentages (e.g., 0.44 for +0.44%). RiskManager thresholds
        expect fractional form (e.g., 0.0044). This helper converts percent-style fields
        into fractional form while preserving already-normalized values.
        """
        metrics = position.get("metrics") if isinstance(position.get("metrics"), dict) else {}
        pnl_obj = position.get("pnl") if isinstance(position.get("pnl"), dict) else {}

        def _normalize(value: Optional[float], source: str) -> Optional[float]:
            if value is None or not math.isfinite(value):
                return None
            # Fields named `pnl_pct` (or computed here) are percent-based -> convert to fraction
            if source in {"pnl_pct", "pnl.pct", "computed"}:
                return value / 100.0
            # For explicit unrealized_pnl_pct, accept fractional inputs directly; if unusually large, treat as percent.
            if abs(value) > 10:
                return value / 100.0
            return value

        candidates = [
            ("unrealized_pnl_pct", position.get("unrealized_pnl_pct")),
            ("metrics.unrealized_pnl_pct", metrics.get("unrealized_pnl_pct")),
            ("pnl_pct", position.get("pnl_pct")),
            ("pnl.pct", pnl_obj.get("pct")),
        ]
        for source, raw_val in candidates:
            candidate = self._safe_float(raw_val)
            normalized = _normalize(candidate, source) if candidate is not None else None
            if normalized is not None:
                return normalized

        # Fallback: derive from unrealized PnL using entry/amount
        unrealized = self._safe_float(position.get("unrealized_pnl"))
        entry_price = self._safe_float(position.get("entry_price"))
        amount = self._safe_float(position.get("amount"))
        if unrealized is None or entry_price is None or amount is None:
            return None
        if entry_price <= 0 or amount <= 0:
            return None

        computed_pct = self._safe_float(
            calculate_pnl_percentage(unrealized, entry_price, amount)
        )
        return _normalize(computed_pct, "computed") if computed_pct is not None else None

    def get_positions_for_symbol(
        self,
        symbol: str,
        strategy_name: Optional[str] = None,
        side: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not symbol:
            return []
        positions_store = getattr(self.position_manager, "positions", {}) or {}
        results: List[Dict[str, Any]] = []
        for pos_id, pos in positions_store.items():
            if pos.get("symbol") != symbol:
                continue
            if strategy_name and pos.get("strategy_name") and pos.get("strategy_name") != strategy_name:
                continue
            if side and pos.get("side") and pos.get("side") != side:
                continue
            snapshot = dict(pos)
            snapshot.setdefault("position_id", pos_id)
            normalized_pct = self._extract_unrealized_pnl_pct(snapshot)
            if normalized_pct is not None:
                snapshot["unrealized_pnl_pct"] = normalized_pct
                metrics_source = snapshot.get("metrics")
                metrics = dict(metrics_source) if isinstance(metrics_source, dict) else {}
                metrics.setdefault("unrealized_pnl_pct", normalized_pct)
                snapshot["metrics"] = metrics
            results.append(snapshot)
        return results


class ExitReason(Enum):
    """Position exit reason enumeration."""
    TAKE_PROFIT = 'take_profit'
    STOP_LOSS = 'stop_loss'
    TRAILING_STOP = 'trailing_stop'
    TIME_EXIT = 'time_exit'
    SIGNAL_EXIT = 'signal_exit'
    MANUAL = 'manual'
    EMERGENCY = 'emergency'
    SHUTDOWN = 'shutdown'  # *** YENİ: Kapanış için yeni neden ***


class AdvancedPositionManager:
    """Comprehensive position lifecycle management."""
    
    LONG_SIDES = {'long', 'buy'}
    SHORT_SIDES = {'short', 'sell'}

    def __init__(self, risk_manager, order_manager, websocket_manager=None, portfolio_manager=None):
        """
        Initialize position manager.
        
        Args:
            risk_manager: RiskManager instance.
            order_manager: SmartOrderManager instance for executing close orders.
            websocket_manager: WebSocketManager for fetching live prices.
            portfolio_manager: PortfolioManager instance (optional, can be set later).
        """
        self.risk_manager = risk_manager
        self.order_manager = order_manager # DOĞRUDAN BAĞIMLILIK
        self.ws_manager = websocket_manager
        self.portfolio_manager = portfolio_manager  # Link to PortfolioManager for trade counting
        
        # Position tracking
        self.positions = {}  # position_id -> position_data
        self.pnl_tracker = {}  # position_id -> pnl_history
        self.closed_positions = []

        # Trade history logging
        self.repo_root = Path.cwd()
        self.trade_history_dir = self.repo_root / 'logs'
        self.trade_history_path = self.trade_history_dir / 'trade_history.jsonl'
        self._ensure_trade_history_path()

        # Position snapshot persistence (open positions only)
        self.state_store = PositionStateStore(self.repo_root / "data" / "positions_snapshot.json", logger=logger)
        self._last_snapshot_save_ts: float = 0.0
        try:
            self._snapshot_throttle_s = float(os.getenv("POSITIONS_SNAPSHOT_THROTTLE_S", 5.0) or 5.0)
        except (TypeError, ValueError):
            self._snapshot_throttle_s = 5.0
        if self._snapshot_throttle_s < 0:
            self._snapshot_throttle_s = 0.0
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None

        # Dispatch notifier wiring (StrategyCoordinator -> LiveTradingEngine bridge wake-up)
        self._dispatch_notifier: Optional[Callable[[], Awaitable[Any]]] = None
        
        logger.info("AdvancedPositionManager initialized")

    def _ensure_trade_history_path(self):
        try:
            self.trade_history_dir.mkdir(parents=True, exist_ok=True)
            self.trade_history_path.touch(exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to prepare trade history file: %s", exc)

    def _save_positions_snapshot(self, force: bool = False) -> bool:
        """Persist the open positions snapshot (throttled unless force=True)."""
        if not getattr(self, "state_store", None):
            return False
        now = time.time()
        throttle = float(getattr(self, "_snapshot_throttle_s", 0.0) or 0.0)
        if not force and throttle > 0 and (now - float(getattr(self, "_last_snapshot_save_ts", 0.0) or 0.0)) < throttle:
            return False
        ok = False
        try:
            ok = bool(self.state_store.save(self.positions))
        except Exception as exc:
            logger.warning("Positions snapshot save failed: %s", exc)
            ok = False
        if ok:
            self._last_snapshot_save_ts = now
        return ok

    async def restore_positions_from_snapshot(
        self,
        *,
        exchange_clients: Optional[Dict[str, Any]] = None,
        reconcile_with_exchange: bool = False,
    ) -> Dict[str, Any]:
        """
        Restore open positions from local snapshot.

        Backward compatible:
        - If snapshot is missing/corrupt, does nothing.

        Optional (LIVE mode only):
        - If reconcile_with_exchange=True, removes snapshot positions that are not open on the exchange.
        """
        results = {
            "restored": 0,
            "skipped": 0,
            "stale_removed": 0,
            "errors": [],
        }
        store = getattr(self, "state_store", None)
        if not store:
            return results

        payload = store.load()
        raw_positions = payload.get("positions", {}) if isinstance(payload, dict) else {}
        if not isinstance(raw_positions, dict) or not raw_positions:
            return results

        restored_positions: Dict[str, Dict[str, Any]] = {}
        for key, raw in raw_positions.items():
            if not isinstance(raw, dict):
                results["skipped"] += 1
                continue
            position_id = raw.get("position_id") or key
            if not position_id:
                results["skipped"] += 1
                continue

            try:
                normalized = self._normalize_restored_position(dict(raw), position_id=str(position_id))
            except Exception as exc:
                results["errors"].append(f"normalize_failed:{position_id}:{exc}")
                results["skipped"] += 1
                continue

            restored_positions[str(position_id)] = normalized

        if not restored_positions:
            return results

        # Install restored positions into in-memory stores and register with risk/portfolio for gating.
        for pid, position in restored_positions.items():
            self.positions[pid] = position
            if pid not in self.pnl_tracker:
                entry_price = float(position.get("entry_price") or 0.0)
                current_price = float(position.get("current_price") or entry_price or 0.0)
                self.pnl_tracker[pid] = [{
                    "timestamp": datetime.now(timezone.utc),
                    "price": current_price,
                    "unrealized_pnl": 0.0,
                    "pnl_pct": 0.0,
                }]

            try:
                if getattr(self, "risk_manager", None) and hasattr(self.risk_manager, "register_position"):
                    self.risk_manager.register_position(pid, dict(position))
            except Exception as exc:
                results["errors"].append(f"risk_register_failed:{pid}:{exc}")

            try:
                if getattr(self, "portfolio_manager", None) and hasattr(self.portfolio_manager, "register_position"):
                    self.portfolio_manager.register_position(pid, dict(position))
            except Exception as exc:
                results["errors"].append(f"portfolio_register_failed:{pid}:{exc}")

            results["restored"] += 1

        if reconcile_with_exchange and exchange_clients:
            try:
                reconcile = await self._reconcile_snapshot_with_exchange(exchange_clients)
                results["stale_removed"] = reconcile.get("stale_removed", 0)
            except Exception as exc:
                results["errors"].append(f"reconcile_failed:{exc}")

        # Persist snapshot after restore/reconcile (keeps disk in sync with memory).
        self._save_positions_snapshot(force=True)
        logger.info(
            "Positions restored from snapshot: restored=%s skipped=%s stale_removed=%s",
            results["restored"],
            results["skipped"],
            results["stale_removed"],
        )
        return results

    def _normalize_restored_position(self, position: Dict[str, Any], *, position_id: str) -> Dict[str, Any]:
        """Ensure restored positions contain required defaults and types."""
        position.setdefault("position_id", position_id)
        position.setdefault("status", PositionStatus.OPEN.value)

        # Datetime restoration (best-effort)
        opened_at = position.get("opened_at")
        if isinstance(opened_at, str):
            parsed = self._parse_datetime(opened_at)
            if parsed:
                position["opened_at"] = parsed
        elif opened_at is None:
            entry_iso = position.get("entry_time_iso") or position.get("opened_at_iso")
            if isinstance(entry_iso, str):
                parsed = self._parse_datetime(entry_iso)
                if parsed:
                    position["opened_at"] = parsed

        entry_price = self._safe_float(position.get("entry_price"), 0.0) or 0.0
        current_price = self._safe_float(position.get("current_price"), entry_price) or entry_price
        position["entry_price"] = entry_price
        position["current_price"] = current_price

        # Trailing defaults (fields consumed by monitor_position_pnl + exit checks)
        position.setdefault("trailing_stop_enabled", False)
        position.setdefault("trailing_stop_distance", 0.02)
        position.setdefault("trailing_stop_activation_threshold", 0.0)
        position.setdefault("trailing_stop_activated", False)

        # Price excursion defaults
        position.setdefault("highest_price", current_price or entry_price)
        position.setdefault("lowest_price", current_price or entry_price)
        position.setdefault("max_adverse_excursion", 0.0)
        position.setdefault("max_favorable_excursion", 0.0)

        # Exit levels defaults (best-effort; unknown restored positions may not have these)
        position.setdefault("stop_loss", 0.0)
        position.setdefault("take_profit", 0.0)

        # Keep execution config (per-position) if present
        execution_cfg = position.get("execution")
        if execution_cfg is not None and not isinstance(execution_cfg, dict):
            position.pop("execution", None)

        return position

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            # Handle Zulu suffix
            normalized = value.replace("Z", "+00:00") if isinstance(value, str) else value
            return datetime.fromisoformat(normalized)
        except Exception:
            return None

    async def _reconcile_snapshot_with_exchange(self, exchange_clients: Dict[str, Any]) -> Dict[str, Any]:
        """
        LIVE-mode reconciliation:
        - If a restored snapshot position is not found on the exchange, remove it from local tracking.

        IMPORTANT: This does NOT adjust equity/PnL (unknown realized outcome).
        """
        if not self.positions:
            return {"stale_removed": 0}
        if not exchange_clients or not isinstance(exchange_clients, dict):
            return {"stale_removed": 0}

        exchange_view = await self._fetch_exchange_open_positions(exchange_clients)
        if not exchange_view.get("ok"):
            return {"stale_removed": 0, "error": exchange_view.get("error")}

        open_index = exchange_view.get("index") or {}
        stale_removed = 0

        for position_id, position in list(self.positions.items()):
            exchange = str(position.get("exchange") or "").lower()
            symbol_raw = str(position.get("symbol") or "")
            raw_side = str(position.get("side") or "").lower()
            amount = self._safe_float(position.get("amount"), 0.0) or 0.0

            if not exchange or not symbol_raw or amount <= 0:
                continue

            # Normalize common symbol/side formats to reduce false "stale" removals.
            symbol_variants = {symbol_raw}
            try:
                sym_norm = symbol_raw.replace("-", "/")
                symbol_variants.add(sym_norm)
                if ":" in sym_norm:
                    symbol_variants.add(sym_norm.split(":", 1)[0])
                if "USDT" in sym_norm and ":" not in sym_norm and "/" in sym_norm:
                    symbol_variants.add(sym_norm + ":USDT")
                # Handle compact symbols like BTCUSDT -> BTC/USDT(:USDT)
                if "/" not in sym_norm and sym_norm.endswith("USDT") and len(sym_norm) > 4:
                    base = sym_norm[:-4]
                    symbol_variants.add(f"{base}/USDT")
                    symbol_variants.add(f"{base}/USDT:USDT")
            except Exception:
                symbol_variants = {symbol_raw}

            side_norm = raw_side
            if raw_side in self.LONG_SIDES:
                side_norm = "long"
            elif raw_side in self.SHORT_SIDES:
                side_norm = "short"
            side_variants = {s for s in {raw_side, side_norm} if s}

            matched = False
            for sym in symbol_variants:
                if not sym:
                    continue
                for side in side_variants:
                    key = (exchange, sym, side)
                    candidates = open_index.get(key) or []
                    if self._match_amount(amount, candidates):
                        matched = True
                        break
                if matched:
                    break

            if matched:
                continue

            self._archive_missing_exchange_position(position_id, reason="reconciled_missing_exchange")
            stale_removed += 1

        if stale_removed:
            self._save_positions_snapshot(force=True)

        return {"stale_removed": stale_removed}

    async def _fetch_exchange_open_positions(self, exchange_clients: Dict[str, Any]) -> Dict[str, Any]:
        """Build a lightweight index of open positions from exchanges for reconciliation."""
        index: Dict[Tuple[str, str, str], List[float]] = {}
        try:
            for ex_name, client in exchange_clients.items():
                ex = str(ex_name or "").lower()
                if not client:
                    continue

                if ex == "bingx" and hasattr(client, "get_bingx_positions"):
                    response = await asyncio.to_thread(client.get_bingx_positions)
                    data = response.get("data", []) if isinstance(response, dict) else []
                    for pos_data in data or []:
                        try:
                            raw_symbol = pos_data.get("symbol")
                            amt = float(pos_data.get("positionAmt", 0))
                            if not raw_symbol or amt == 0:
                                continue
                            symbol = raw_symbol.replace("-", "/")
                            if "USDT" in symbol and ":" not in symbol:
                                symbol += ":USDT"
                            side = "long" if amt > 0 else "short"
                            key = (ex, symbol, side)
                            index.setdefault(key, []).append(abs(float(amt)))
                        except Exception:
                            continue
                    continue

                if hasattr(client, "fetch_positions"):
                    positions = await asyncio.to_thread(client.fetch_positions)
                    for pos in positions or []:
                        try:
                            symbol = pos.get("symbol")
                            if not symbol:
                                continue
                            amt = pos.get("contracts", pos.get("amount", 0))
                            amt = float(amt or 0)
                            if amt == 0:
                                continue
                            side = (pos.get("side") or "").lower()
                            if side not in ("long", "short"):
                                side = "long" if amt > 0 else "short"
                            key = (ex, str(symbol), side)
                            index.setdefault(key, []).append(abs(float(amt)))
                        except Exception:
                            continue
        except Exception as exc:
            return {"ok": False, "error": str(exc), "index": {}}

        return {"ok": True, "index": index}

    @staticmethod
    def _match_amount(local_amount: float, candidates: List[float], rel_tol: float = 0.05, abs_tol: float = 1e-12) -> bool:
        if local_amount <= 0:
            return False
        for cand in candidates or []:
            try:
                cand_val = float(cand)
            except (TypeError, ValueError):
                continue
            if cand_val <= 0:
                continue
            if abs(local_amount - cand_val) <= max(abs_tol, local_amount * rel_tol):
                return True
        return False

    def _archive_missing_exchange_position(self, position_id: str, reason: str) -> None:
        """Remove a stale position without affecting portfolio equity (unknown realized PnL)."""
        position = self.positions.pop(position_id, None)
        if not position:
            return

        try:
            position["status"] = PositionStatus.CLOSED.value
            position["closed_at"] = datetime.now(timezone.utc)
            position["exit_reason"] = reason
            position.setdefault("realized_pnl", 0.0)
            self.closed_positions.append(position)
        except Exception:
            pass

        try:
            if getattr(self, "risk_manager", None) and hasattr(self.risk_manager, "active_positions"):
                self.risk_manager.active_positions.pop(position_id, None)
        except Exception:
            pass

        try:
            if getattr(self, "portfolio_manager", None) and hasattr(self.portfolio_manager, "active_positions"):
                self.portfolio_manager.active_positions.pop(position_id, None)
        except Exception:
            pass

        self.pnl_tracker.pop(position_id, None)
        logger.warning("Archived stale position missing on exchange: %s (%s)", position_id, reason)

    def set_dispatch_notifier(self, notifier: Optional[Callable[[], Awaitable[Any]]]) -> None:
        """Register async callback that wakes the coordinator dispatch loop."""
        self._dispatch_notifier = notifier

    def _schedule_dispatch_nudge(self) -> None:
        if not self._dispatch_notifier:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Dispatch notifier skipped; no running loop detected")
            return

        try:
            result = self._dispatch_notifier()
        except Exception as exc:
            logger.debug("Dispatch notifier callable failed: %s", exc)
            return

        if asyncio.iscoroutine(result):
            loop.create_task(result)

    def _generate_trade_id(self) -> str:
        return uuid.uuid4().hex[:8]

    @staticmethod
    def _safe_float(value: Any, fallback: Optional[float] = None) -> Optional[float]:
        try:
            return float(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _isoformat_z(dt: Optional[datetime]) -> Optional[str]:
        if not dt:
            return None
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _extract_entry_metadata(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        metadata = signal.get('metadata') if isinstance(signal, dict) else {}
        regime_info = metadata.get('regime') if isinstance(metadata, dict) else None
        regime_label = None
        if isinstance(regime_info, dict):
            regime_label = regime_info.get('trend') or regime_info.get('momentum') or regime_info.get('volatility')
        elif regime_info:
            regime_label = str(regime_info)

        ml_metadata = signal.get('ml_metadata') if signal else {}
        if not isinstance(ml_metadata, dict):
            ml_metadata = {}

        entry_indicators = signal.get('entry_indicators') if signal else {}
        if not isinstance(entry_indicators, dict):
            entry_indicators = {}

        return {
            'rsi_at_entry': self._safe_float(entry_indicators.get('rsi'))
                             or self._safe_float(signal.get('rsi'))
                             or self._safe_float(signal.get('entry_rsi')),
            'regime_at_entry': regime_label or signal.get('regime'),
            'regime_confidence': self._safe_float(ml_metadata.get('regime_confidence')),
            'ml_price_score': self._safe_float(ml_metadata.get('consensus')),
            'ml_price_score_normalized': self._safe_float(ml_metadata.get('ml_price_score_normalized')),
            'ml_price_direction': ml_metadata.get('price_direction'),
            'ml_regime': ml_metadata.get('regime'),
            'ml_price_confidence': self._safe_float(ml_metadata.get('price_confidence')),
            'ppo_action': signal.get('ppo_action'),
            'ml_position_modifier': self._safe_float(signal.get('ml_position_modifier')),
            'quality_score': self._safe_float(signal.get('quality_score')),
            'quality_breakdown': signal.get('quality_breakdown'),
            'regime_data': regime_info if isinstance(regime_info, dict) else None,
            'entry_indicators': entry_indicators,
            'ml_metadata': ml_metadata
        }

    def _extract_price_field(self, payload: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            if not isinstance(payload, dict):
                continue
            value = self._safe_float(payload.get(key))
            if value and value > 0:
                return value
        return None

    def _normalize_pct(self, raw_value: Optional[float]) -> Optional[float]:
        if raw_value is None:
            return None
        try:
            pct = float(raw_value)
        except (TypeError, ValueError):
            return None
        if pct <= 0:
            return None
        return pct / 100.0 if pct > 1 else pct

    def _derive_take_profit(self, signal: Dict[str, Any], entry_price: float,
                            stop_loss: Optional[float], is_long: bool) -> float:
        explicit_tp = self._extract_price_field(signal, ['target', 'take_profit', 'tp_price'])
        if explicit_tp:
            return explicit_tp

        tp_pct = self._normalize_pct(
            signal.get('tp_pct') or signal.get('take_profit_pct') or signal.get('target_pct')
        )
        if tp_pct:
            return entry_price * (1 + tp_pct) if is_long else entry_price * (1 - tp_pct)

        atr_value = self._safe_float(signal.get('atr') or signal.get('atr_value'))
        tp_atr_mult = self._safe_float(signal.get('tp_atr_mult'))
        if atr_value and tp_atr_mult:
            distance = atr_value * tp_atr_mult
            return entry_price + distance if is_long else entry_price - distance

        risk_limits = getattr(self.risk_manager, 'risk_limits_dataclass', None)
        take_profit_ratio = getattr(risk_limits, 'take_profit_ratio', None)
        if take_profit_ratio and stop_loss:
            distance = abs(entry_price - stop_loss) * take_profit_ratio
            candidate = entry_price + distance if is_long else entry_price - distance
            if candidate > 0:
                return candidate

        default_tp_pct = 0.05
        return entry_price * (1 + default_tp_pct) if is_long else entry_price * (1 - default_tp_pct)

    def _derive_exit_levels(self, signal: Dict[str, Any], entry_price: float) -> Tuple[float, float]:
        side = (signal.get('side') or 'long').lower()
        is_long = side not in self.SHORT_SIDES

        stop_loss = self._extract_price_field(signal, ['stop', 'stop_loss', 'stop_price'])
        if not stop_loss:
            calc_stop = getattr(self.risk_manager, '_calculate_stop_loss_from_signal', None)
            if callable(calc_stop):
                try:
                    stop_loss = self._safe_float(calc_stop(signal, entry_price))
                except Exception:
                    stop_loss = None

        if not stop_loss or stop_loss <= 0:
            fallback_pct = getattr(
                getattr(self.risk_manager, 'risk_limits_dataclass', None),
                'stop_loss_pct',
                0.02
            ) or 0.02
            stop_loss = entry_price * (1 - fallback_pct) if is_long else entry_price * (1 + fallback_pct)

        take_profit = self._derive_take_profit(signal, entry_price, stop_loss, is_long)

        if is_long and stop_loss >= entry_price:
            stop_loss = entry_price * 0.99
        elif not is_long and stop_loss <= entry_price:
            stop_loss = entry_price * 1.01

        if is_long and take_profit <= entry_price:
            take_profit = entry_price * 1.02
        elif not is_long and take_profit >= entry_price:
            take_profit = entry_price * 0.98

        signal['stop'] = stop_loss
        signal['target'] = take_profit
        return stop_loss, take_profit

    def _append_trade_history(self, payload: Dict[str, Any]) -> None:
        try:
            logger.debug("Appending trade payload to %s", self.trade_history_path)
            with self.trade_history_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(payload, ensure_ascii=False))
                stream.write('\n')
        except Exception as exc:
            logger.warning("Failed to append trade history log: %s", exc)

    # *** UPDATED: Tüm açık pozisyonları kapatmak için (exchange_clients dependency injection ile) ***
    async def close_all_positions(self, exchange_clients: Optional[Dict] = None, reason: str = ExitReason.SHUTDOWN.value) -> Dict[str, Any]:
        """
        Force-close all open positions using the injected exchange clients.
        
        Args:
            exchange_clients: Dictionary of live exchange client instances (CRITICAL for shutdown)
            reason: Reason for closing positions
            
        Returns:
            Dictionary with closure results
        """
        if not self.positions:
            logger.info("No open positions to close.")
            return {'success': True, 'closed_count': 0, 'errors': []}

        logger.warning(f"🚨 Initiating closure of all {len(self.positions)} open positions. Reason: {reason}")
        
        closed_count = 0
        errors = []
        
        position_ids_to_close = list(self.positions.keys())

        for position_id in position_ids_to_close:
            position = self.positions.get(position_id)
            if not position:
                continue

            symbol = position['symbol']
            side = position['side']
            amount = position['amount']
            exchange = position['exchange'] # Pozisyon bilgisinden borsayı al
            
            close_side = 'buy' if side.lower() in ['short', 'sell'] else 'sell'
            
            try:
                # --- CRITICAL FIX: Use injected exchange_clients ---
                if not self.order_manager:
                    logger.error(f"Cannot close {symbol}: OrderManager is not available.")
                    errors.append({'position_id': position_id, 'reason': 'OrderManager not found'})
                    continue

                logger.info(f"Submitting market order to close {position_id}: {close_side} {amount} {symbol}")
                
                # Create close order request and submit to OrderManager
                close_order_request = {
                    'symbol': symbol,
                    'side': close_side,
                    'amount': amount,
                    'exchange': exchange,  # Include exchange info in order
                }
                
                max_attempts = _SHUTDOWN_CLOSE_MAX_ATTEMPTS
                attempt_success = False
                last_error = 'Execution failed'

                for attempt in range(1, max_attempts + 1):
                    execution_result = await self.order_manager.place_order(
                        close_order_request,
                        execution_algo='market',
                        exchange_clients=exchange_clients
                    )

                    if execution_result and execution_result.get('success'):
                        exit_price = execution_result.get('avg_price')
                        await self.close_position(position_id, exit_price, exit_reason=reason)
                        closed_count += 1
                        attempt_success = True
                        break

                    last_error = execution_result.get('reason', 'Execution failed')
                    logger.error(
                        f"Failed to close position {position_id} (attempt {attempt}/{max_attempts}): {last_error}"
                    )

                    if attempt < max_attempts and _SHUTDOWN_CLOSE_BACKOFF > 0:
                        wait_seconds = _SHUTDOWN_CLOSE_BACKOFF * attempt
                        logger.warning(
                            f"Retrying closure for {position_id} in {wait_seconds:.1f}s (attempt {attempt + 1}/{max_attempts})"
                        )
                        await asyncio.sleep(wait_seconds)

                if not attempt_success:
                    errors.append({'position_id': position_id, 'reason': last_error})

            except Exception as e:
                logger.error(f"Critical error closing position {position_id}: {e}", exc_info=True)
                errors.append({'position_id': position_id, 'reason': str(e)})

        logger.info(f"✅ Position closure summary: Closed={closed_count}, Errors={len(errors)}")
        return {'success': len(errors) == 0, 'closed_count': closed_count, 'errors': errors}
    
    async def open_position(self, signal: Dict, execution_result: Dict) -> Dict[str, Any]:
        """
        Initialize new position with full tracking.
        
        Args:
            signal: Trading signal that triggered the position
            execution_result: Order execution result
            
        Returns:
            Position initialization result
        """
        try:
            if not execution_result.get('success'):
                return {
                    'success': False,
                    'reason': 'Execution failed',
                    'position_id': None
                }
            
            # Generate position ID
            position_id = f"pos_{signal.get('symbol', 'UNKNOWN')}_{int(datetime.now(timezone.utc).timestamp())}"
            run_id = get_current_run_id()
            
            # Extract position details
            symbol = signal.get('symbol')
            side = signal.get('side', 'long')
            timeframe = signal.get('timeframe') or signal.get('tf')
            entry_price = execution_result.get('avg_price', 0)
            amount = execution_result.get('filled_amount', 0)
            
            # Calculate stop-loss and take-profit levels with directional awareness
            stop_loss, take_profit = self._derive_exit_levels(signal, entry_price)
            
            # CRITICAL FIX: Get exchange from execution_result if not in signal
            # This ensures we always have a valid exchange name for position closure during shutdown
            exchange = signal.get('exchange')
            if not exchange or exchange == 'unknown':
                # Try to get from execution_result order object
                order_obj = execution_result.get('order', {})
                exchange = order_obj.get('exchange')
                
                # If still missing or unknown, log critical warning
                if not exchange or exchange == 'unknown':
                    # This should never happen with the LiveTradingEngine fix, but log for debugging
                    logger.warning(
                        f"⚠️ Position {position_id}: Exchange is 'unknown' - shutdown position closure may fail! "
                        f"Signal: {signal.get('symbol')}, Strategy: {signal.get('strategy')}"
                    )
                    # Set to unknown but continue - better to track the position than fail creation
                    exchange = 'unknown'
            
            # Create position record
            entry_meta = self._extract_entry_metadata(signal)
            strategy_name = signal.get('strategy_name') or signal.get('strategy') or 'unknown'
            dca_metadata = signal.get('dca_metadata') if isinstance(signal.get('dca_metadata'), dict) else {}
            scale_profile = signal.get('scale_profile') or dca_metadata.get('profile')
            risk_per_unit = abs(entry_price - stop_loss)
            risk_usd = risk_per_unit * amount if amount and amount > 0 else 0.0
            opened_at = datetime.now(timezone.utc)
            opened_at_iso = self._isoformat_z(opened_at)
            position = {
                'position_id': position_id,
                'trade_id': self._generate_trade_id(),
                'run_id': run_id,
                'symbol': symbol,
                'side': side,
                'timeframe': timeframe,
                'entry_price': entry_price,
                'current_price': entry_price,
                'amount': amount,
                'size': amount,
                'position_size': amount,
                'initial_amount': amount,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': PositionStatus.OPEN.value,
                'opened_at': opened_at,
                'entry_time_iso': opened_at_iso,
                'open_timestamp': time.time(),
                'strategy': strategy_name,
                'strategy_name': strategy_name,
                'exchange': exchange,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'trailing_stop_enabled': False,
                'trailing_stop_distance': 0.02,
                'trailing_stop_activation_threshold': 0.0,
                'trailing_stop_activated': False,
                'highest_price': entry_price if side == 'long' else entry_price,
                'lowest_price': entry_price if side == 'short' else entry_price,
                'max_adverse_excursion': 0.0,
                'max_favorable_excursion': 0.0,
                'max_adverse_excursion_pct': 0.0,
                'max_favorable_excursion_pct': 0.0,
                'entry_metadata': entry_meta,
                'risk_usd': risk_usd,
                'risk_amount': risk_usd,
                'position_notional': entry_price * amount if amount else 0.0,
                'rsi_at_entry': entry_meta.get('rsi_at_entry'),
                'regime_at_entry': entry_meta.get('regime_at_entry'),
                'regime_confidence': entry_meta.get('regime_confidence'),
                'ml_price_score': entry_meta.get('ml_price_score'),
                'ml_price_score_normalized': entry_meta.get('ml_price_score_normalized'),
                'ml_price_direction': entry_meta.get('ml_price_direction'),
                'ml_regime': entry_meta.get('ml_regime'),
                'ppo_action': entry_meta.get('ppo_action'),
                'ml_position_modifier': entry_meta.get('ml_position_modifier'),
                'quality_score': entry_meta.get('quality_score'),
                'exit_reason': None,
                'scale_profile': scale_profile,
                'dca_metadata': dca_metadata or None,
                'volume_bucket_at_entry': signal.get('volume_bucket'),
                'volume_strength_at_entry': signal.get('volume_strength'),
                'volume_ctx_source': signal.get('volume_ctx_source'),
                'momentum_strength_at_entry': signal.get('momentum_strength'),
            }
            
            # Register with risk manager
            self.risk_manager.register_position(position_id, position)

            # Also register with portfolio manager so exposure/concurrency stats stay accurate
            if self.portfolio_manager and hasattr(self.portfolio_manager, 'register_position'):
                try:
                    self.portfolio_manager.register_position(position_id, dict(position))
                except Exception as pm_err:
                    logger.error(
                        f"⚠️ Failed to register position {position_id} with PortfolioManager: {pm_err}",
                        exc_info=True
                    )
            
            # Store position
            self.positions[position_id] = position
            
            # Initialize P&L tracker
            self.pnl_tracker[position_id] = [{
                'timestamp': datetime.now(timezone.utc),
                'price': entry_price,
                'unrealized_pnl': 0.0,
                'pnl_pct': 0.0
            }]
            
            # Increment daily trade counter
            if self.portfolio_manager and hasattr(self.portfolio_manager, 'increment_trade_count'):
                self.portfolio_manager.increment_trade_count()
            else:
                logger.warning("⚠️ Could not increment trade count: PortfolioManager not linked or method missing")
            
            logger.info(f"Position opened: {position_id} - {symbol} {side} {amount} @ {entry_price:.4f}")
            logger.info(f"  Stop-loss: {stop_loss:.4f}, Take-profit: {take_profit:.4f}")

            # Persist snapshot for restart safety (open positions only)
            self._save_positions_snapshot(force=True)
            
            return {
                'success': True,
                'position_id': position_id,
                'position': position
            }
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return {
                'success': False,
                'reason': str(e),
                'position_id': None
            }
    
    async def monitor_position_pnl(self, position_id: str, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Real-time P&L monitoring and alerting.
        
        Args:
            position_id: Position identifier
            current_price: Current market price (optional, will fetch if not provided)
            
        Returns:
            P&L monitoring result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Update current price
            if current_price:
                position['current_price'] = current_price
            
            # Calculate unrealized P&L
            entry_price = position['entry_price']
            amount = position['amount']
            side = position['side']
            
            unrealized_pnl = calculate_unrealized_pnl(side, entry_price, position['current_price'], amount)
            
            position['unrealized_pnl'] = unrealized_pnl
            
            # Calculate P&L percentage
            pnl_pct = calculate_pnl_percentage(unrealized_pnl, entry_price, amount)
            position['pnl_pct'] = pnl_pct
            normalized_unrealized_pct = (pnl_pct / 100.0) if pnl_pct is not None else None
            if normalized_unrealized_pct is not None:
                position['unrealized_pnl_pct'] = normalized_unrealized_pct
                metrics = position.get('metrics') if isinstance(position.get('metrics'), dict) else {}
                metrics = dict(metrics) if metrics else {}
                metrics['unrealized_pnl_pct'] = normalized_unrealized_pct
                position['metrics'] = metrics
            
            # Update max adverse/favorable excursion
            if unrealized_pnl < 0 and abs(unrealized_pnl) > abs(position['max_adverse_excursion']):
                position['max_adverse_excursion'] = unrealized_pnl
                if abs(pnl_pct) > abs(position.get('max_adverse_excursion_pct', 0.0)):
                    position['max_adverse_excursion_pct'] = pnl_pct
            elif unrealized_pnl > 0 and unrealized_pnl > position['max_favorable_excursion']:
                position['max_favorable_excursion'] = unrealized_pnl
                if pnl_pct > position.get('max_favorable_excursion_pct', 0.0):
                    position['max_favorable_excursion_pct'] = pnl_pct
            
            # Update highest/lowest price for trailing stop
            is_long = side in self.LONG_SIDES
            if is_long:
                if position['current_price'] > position['highest_price']:
                    position['highest_price'] = position['current_price']
            else:  # short
                if position['current_price'] < position['lowest_price']:
                    position['lowest_price'] = position['current_price']

            stop_updated = False
            if position.get('trailing_stop_enabled'):
                trailing_distance = self._safe_float(position.get('trailing_stop_distance'), 0.02) or 0.02
                was_active = bool(position.get('trailing_stop_activated', False))
                is_active = self._is_trailing_stop_active(position, position['current_price'], entry_price, is_long)
                if is_active and not was_active:
                    logger.info(
                        "TRAILING_ACTIVATED %s symbol=%s side=%s entry=%.2f price=%.2f "
                        "activation_threshold=%.4f trail_dist=%.4f",
                        position_id,
                        position.get('symbol', 'UNKNOWN'),
                        side,
                        entry_price,
                        position['current_price'],
                        position.get('trailing_stop_activation_threshold', 0.0) or 0.0,
                        trailing_distance
                    )

                if is_active:
                    config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
                    trailing_cfg = config.get('position_management', {}).get('trailing_stop', {})
                    min_step_bps = self._safe_float(trailing_cfg.get('min_trail_step_bps'), 0.0) or 0.0
                    min_update_interval_s = self._safe_float(trailing_cfg.get('min_trail_update_interval_s'), 0.0) or 0.0
                    if min_step_bps < 0:
                        min_step_bps = 0.0
                    if min_update_interval_s < 0:
                        min_update_interval_s = 0.0
                    min_step_fraction = (min_step_bps / 10000.0) if min_step_bps > 0 else 0.0
                    now_ts = time.time()
                    last_update_ts = position.get('last_trail_update_ts', 0.0) or 0.0

                    if is_long:
                        trailing_stop_level = position['highest_price'] * (1 - trailing_distance)
                        if trailing_stop_level > position['stop_loss']:
                            update_allowed = True
                            if min_update_interval_s > 0 and (now_ts - last_update_ts) < min_update_interval_s:
                                update_allowed = False
                            if min_step_fraction > 0 and position['stop_loss'] > 0:
                                step_fraction = (trailing_stop_level - position['stop_loss']) / position['stop_loss']
                                if step_fraction < min_step_fraction:
                                    update_allowed = False

                            if update_allowed:
                                old_stop = position['stop_loss']
                                position['stop_loss'] = trailing_stop_level
                                position['last_trail_update_ts'] = now_ts
                                stop_updated = True
                                trail_log_interval_s = 30
                                last_log_ts = position.get('last_trail_log_ts', 0.0) or 0.0
                                if (now_ts - last_log_ts) >= trail_log_interval_s:
                                    logger.info(
                                        "TRAILING_STOP_UPDATE %s symbol=%s side=%s old_stop=%.4f new_stop=%.4f "
                                        "trail_dist=%.4f highest=%.2f lowest=%.2f entry=%.2f price=%.2f",
                                        position_id,
                                        position.get('symbol', 'UNKNOWN'),
                                        side,
                                        old_stop,
                                        trailing_stop_level,
                                        trailing_distance,
                                        position['highest_price'],
                                        position['lowest_price'],
                                        entry_price,
                                        position['current_price']
                                    )
                                    position['last_trail_log_ts'] = now_ts
                    else:
                        trailing_stop_level = position['lowest_price'] * (1 + trailing_distance)
                        if trailing_stop_level < position['stop_loss']:
                            update_allowed = True
                            if min_update_interval_s > 0 and (now_ts - last_update_ts) < min_update_interval_s:
                                update_allowed = False
                            if min_step_fraction > 0 and position['stop_loss'] > 0:
                                step_fraction = (position['stop_loss'] - trailing_stop_level) / position['stop_loss']
                                if step_fraction < min_step_fraction:
                                    update_allowed = False

                            if update_allowed:
                                old_stop = position['stop_loss']
                                position['stop_loss'] = trailing_stop_level
                                position['last_trail_update_ts'] = now_ts
                                stop_updated = True
                                trail_log_interval_s = 30
                                last_log_ts = position.get('last_trail_log_ts', 0.0) or 0.0
                                if (now_ts - last_log_ts) >= trail_log_interval_s:
                                    logger.info(
                                        "TRAILING_STOP_UPDATE %s symbol=%s side=%s old_stop=%.4f new_stop=%.4f "
                                        "trail_dist=%.4f highest=%.2f lowest=%.2f entry=%.2f price=%.2f",
                                        position_id,
                                        position.get('symbol', 'UNKNOWN'),
                                        side,
                                        old_stop,
                                        trailing_stop_level,
                                        trailing_distance,
                                        position['highest_price'],
                                        position['lowest_price'],
                                        entry_price,
                                        position['current_price']
                                    )
                                    position['last_trail_log_ts'] = now_ts

            # Persist snapshot only when stop-loss was updated by trailing logic (throttled).
            if stop_updated:
                self._save_positions_snapshot(force=False)
            
            # Record P&L snapshot
            self.pnl_tracker[position_id].append({
                'timestamp': datetime.now(timezone.utc),
                'price': position['current_price'],
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct
            })
            
            # Check for exit conditions
            exit_signal = await self._check_exit_conditions(position)
            
            return {
                'success': True,
                'position_id': position_id,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'current_price': position['current_price'],
                'exit_signal': exit_signal
            }
            
        except Exception as e:
            logger.error(f"Error monitoring position P&L: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _check_exit_conditions(self, position: Dict) -> Optional[Dict[str, Any]]:
        """Check if position should be exited based on conditions."""
        current_price = position['current_price']
        side = position['side']
        
        # Check stop-loss
        if side in ['long', 'buy']:
            if current_price <= position['stop_loss']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.STOP_LOSS.value,
                    'price': current_price
                }
            # Check take-profit
            if current_price >= position['take_profit']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.TAKE_PROFIT.value,
                    'price': current_price
                }
        else:  # short
            if current_price >= position['stop_loss']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.STOP_LOSS.value,
                    'price': current_price
                }
            if current_price <= position['take_profit']:
                return {
                    'should_exit': True,
                    'reason': ExitReason.TAKE_PROFIT.value,
                    'price': current_price
                }
        
        return None
    
    async def close_position(self, position_id: str, exit_price: float, 
                           exit_reason: str = ExitReason.MANUAL.value) -> Dict[str, Any]:
        """
        Close position and finalize P&L.
        
        Args:
            position_id: Position identifier
            exit_price: Exit execution price
            exit_reason: Reason for exit
            
        Returns:
            Position close result
        """
        try:
            if position_id not in self.positions:
                return {'success': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            
            # Calculate final P&L
            entry_price = position['entry_price']
            amount = position['amount']
            side = position['side']
            symbol = position.get('symbol', 'UNKNOWN')
             
            realized_pnl = calculate_realized_pnl(side, entry_price, exit_price, amount)

            # "Stop in profit" normalization: if a stop-loss exit realizes profit, treat it as a trailing stop
            # so it is counted under Trailing Stop in session reporting.
            if exit_reason == ExitReason.STOP_LOSS.value and realized_pnl > 0:
                exit_reason = ExitReason.TRAILING_STOP.value
             
            position['realized_pnl'] = realized_pnl
            position['exit_price'] = exit_price
            position['exit_reason'] = exit_reason
            position['status'] = PositionStatus.CLOSED.value
            position['closed_at'] = datetime.now(timezone.utc)
            
            # Calculate return percentage
            position_value = entry_price * amount
            return_pct = (realized_pnl / position_value) * 100 if position_value > 0 else 0
            position['return_pct'] = return_pct
            
            # Close with risk manager
            self.risk_manager.close_position(position_id, exit_price, realized_pnl)

            # Keep PortfolioManager in sync for exposure/drawdown metrics
            if self.portfolio_manager and hasattr(self.portfolio_manager, 'close_position'):
                try:
                    self.portfolio_manager.close_position(position_id, exit_price, realized_pnl)
                except Exception as pm_err:
                    logger.error(
                        f"⚠️ Failed to close position {position_id} via PortfolioManager: {pm_err}",
                        exc_info=True
                    )
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[position_id]

            # Persist snapshot removal for restart safety
            self._save_positions_snapshot(force=True)
            
            # ===== RL FEEDBACK LOOP (NEW) =====
            # Provide feedback to RL agent if available
            if hasattr(self, 'rl_agent') and self.rl_agent is not None:
                try:
                    # Calculate reward from PnL
                    reward = self._calculate_rl_reward(realized_pnl, return_pct, exit_reason)
                    
                    # Get entry and exit states if available
                    entry_state = position.get('entry_state')
                    current_state = position.get('current_state')
                    
                    # Only provide feedback if we have state information
                    if entry_state is not None and current_state is not None:
                        # Map the action (buy/hold/sell)
                        action_map = {'buy': 0, 'hold': 1, 'sell': 2}
                        action = action_map.get(side, 1)  # Default to hold if unknown
                        
                        # Provide experience to RL agent
                        metrics = self.rl_agent.learn_from_experience(
                            state=entry_state,
                            action=action,
                            reward=reward,
                            next_state=current_state,
                            done=True  # Trade is complete
                        )
                        
                        logger.info(
                            f"🧠 [RL-FEEDBACK] {symbol}: Reward={reward:.4f}, "
                            f"Loss={metrics.get('loss', 0):.4f}, "
                            f"Q-value={metrics.get('q_value', 0):.4f}"
                        )
                    else:
                        logger.debug(f"🧠 [RL-FEEDBACK] {symbol}: No state data available for learning")
                except Exception as e:
                    logger.warning(f"🧠 [RL-FEEDBACK] Failed to provide feedback to RL agent: {e}")
            
            # *** DÜZELTME: Hatalı f-string düzeltildi ***
            exit_emoji = '🛑' if exit_reason == ExitReason.STOP_LOSS.value else \
                         '🎯' if exit_reason == ExitReason.TAKE_PROFIT.value else \
                         '🚦' if exit_reason == ExitReason.TRAILING_STOP.value else \
                         '🚪' if exit_reason == ExitReason.SHUTDOWN.value else '🔄'

            exit_type = (
                'STOP-LOSS-HIT' if exit_reason == ExitReason.STOP_LOSS.value else
                'TAKE-PROFIT-HIT' if exit_reason == ExitReason.TAKE_PROFIT.value else
                'TRAILING-STOP-HIT' if exit_reason == ExitReason.TRAILING_STOP.value else
                exit_reason.upper().replace('_', '-')
            )
            
            logger.info(
                f"{exit_emoji} [{exit_type}]\n"
                f"   Symbol: {symbol}\n"
                f"   Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}\n"
                f"   P&L: ${realized_pnl:.2f} ({return_pct:+.2f}%)\n"
                f"   Reason: {exit_reason.upper().replace('_', '-')}"
            )
            exit_time = position.get('closed_at')
            open_time = position.get('opened_at')
            entry_time_iso = self._isoformat_z(open_time)
            exit_time_iso = self._isoformat_z(exit_time)
            duration_seconds = ((exit_time - open_time).total_seconds()
                                 if exit_time and open_time else 0.0)
            duration_min = round(duration_seconds / 60, 1)
            risk_usd = position.get('risk_usd', 0.0)
            rr_achieved = round(realized_pnl / risk_usd, 2) if risk_usd else None
            mfe_pct = position.get('max_favorable_excursion_pct', 0.0)
            mae_pct = position.get('max_adverse_excursion_pct', 0.0)
            entry_meta = position.get('entry_metadata', {}) or {}
            regime_data = entry_meta.get('regime_data') or {}
            if not isinstance(regime_data, dict):
                regime_data = {}
            payload = {
                'event': 'TRADE_CLOSED',
                'timestamp': exit_time_iso or self._isoformat_z(datetime.now(timezone.utc)),
                'run_id': position.get('run_id') or get_current_run_id(),
                'trade_id': position.get('trade_id'),
                'position_id': position_id,
                'symbol': position.get('symbol'),
                'timeframe': position.get('timeframe'),
                'side': position.get('side', '').upper(),
                'strategy': position.get('strategy_name') or position.get('strategy'),
                'strategy_name': position.get('strategy_name') or position.get('strategy'),
                'entry_price': round(position.get('entry_price', 0.0), 4),
                'entry_time': entry_time_iso,
                'exit_price': round(exit_price, 4),
                'exit_time': exit_time_iso,
                'exit_reason': exit_reason,
                'position_size': position.get('position_size') or position.get('size') or position.get('amount'),
                'pnl_usd': round(realized_pnl, 4),
                'realized_pnl_usd': round(realized_pnl, 4),
                'realized_pnl_usdt': round(realized_pnl, 4),
                'pnl_pct': round(return_pct, 3),
                'rr': rr_achieved,
                'rr_achieved': rr_achieved,
                'duration_min': duration_min,
                'rsi_at_entry': position.get('rsi_at_entry'),
                'regime_at_entry': position.get('regime_at_entry'),
                'regime_conf': position.get('regime_confidence'),
                'ml_price_score': position.get('ml_price_score'),
                'ml_price_score_normalized': position.get('ml_price_score_normalized'),
                'ml_price_direction': position.get('ml_price_direction'),
                'ml_regime': position.get('ml_regime'),
                'ppo_action': position.get('ppo_action'),
                'ml_position_modifier': position.get('ml_position_modifier'),
                'quality_score': position.get('quality_score'),
                'quality_breakdown': position.get('entry_metadata', {}).get('quality_breakdown'),
                'mfe_pct': round(mfe_pct, 3),
                'mae_pct': round(mae_pct, 3),
                'volume_bucket_at_entry': position.get('volume_bucket_at_entry'),
                'volume_strength_at_entry': position.get('volume_strength_at_entry'),
                'volume_ctx_source': position.get('volume_ctx_source'),
                'momentum_strength_at_entry': position.get('momentum_strength_at_entry'),
                'entry_metadata': {
                    'entry_indicators': entry_meta.get('entry_indicators'),
                    'ml_metadata': entry_meta.get('ml_metadata'),
                    'regime_data': {
                        k: (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
                        for k, v in regime_data.items()
                    }
                }
            }
            logger.debug("TRADE_CLOSED payload prepared for position %s", position_id)
            logger.info("TRADE_CLOSED %s", json.dumps(payload))
            self._append_trade_history(payload)

            self._schedule_dispatch_nudge()

            return {
                'success': True,
                'position_id': position_id,
                'realized_pnl': realized_pnl,
                'return_pct': return_pct,
                'exit_reason': exit_reason
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def _get_current_price_from_ws(self, symbol: str) -> Optional[float]:
        """
        Get current price from WebSocket with API fallback.
        Phase 3.4 - Issue #100: Exit Monitoring
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None
        """
        try:
            # Try WebSocket first
            if self.ws_manager:
                try:
                    # WebSocket'ten fiyat almayı dene
                    # ÖNEMLİ: Bu, WebSocketManager'da get_latest_ticker gibi bir metodun
                    # var olduğunu varsayar. Eğer yoksa, collector'dan direkt alınabilir.
                    ticker = None
                    if hasattr(self.ws_manager, 'get_latest_ticker'):
                         ticker = self.ws_manager.get_latest_ticker(symbol)
                    elif hasattr(self.ws_manager, 'collector'):
                         # Collector'dan almayı dene
                         ticker = self.ws_manager.collector.get_latest_data(symbol, 'ticker')

                    if ticker and 'last' in ticker:
                        return float(ticker['last'])
                except Exception as ws_error:
                    logger.debug(f"WebSocket price fetch failed for {symbol}: {ws_error}")
            
            # Fallback to API - safely check if exchange_clients exists
            if hasattr(self.portfolio_manager, 'exchange_clients') and self.portfolio_manager.exchange_clients:
                for ex_name, client in self.portfolio_manager.exchange_clients.items():
                    try:
                        ticker = await asyncio.to_thread(client.fetch_ticker, symbol)
                        last_price = ticker.get('last', ticker.get('close', 0))
                        if last_price > 0:
                            return float(last_price)
                    except Exception as api_error:
                        logger.debug(f"API price fetch failed for {symbol} on {ex_name}: {api_error}")
            
            logger.warning(f"Could not fetch current price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _check_stop_loss_hit(self, position: Dict, current_price: float) -> bool:
        """Check if stop-loss is hit."""
        if current_price <= 0:
            logger.warning(f"_check_stop_loss_hit called with invalid current_price={current_price} for position_id={position.get('id', 'unknown')}. Data quality issue.")
            return False
        
        side = position['side']
        stop_loss = position['stop_loss']
        
        if side in ['long', 'buy']:
            return current_price <= stop_loss
        else:  # short
            return current_price >= stop_loss
    
    def _check_take_profit_hit(self, position: Dict, current_price: float) -> bool:
        """Check if take-profit is hit."""
        if current_price <= 0:
            return False
        
        side = position['side']
        take_profit = position['take_profit']
        
        if side in ['long', 'buy']:
            return current_price >= take_profit
        else:  # short
            return current_price <= take_profit
    
    def _check_timeout_exit(self, position: Dict) -> bool:
        """Check if position should exit due to timeout."""
        # Get configuration
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        position_config = config.get('position_management', {}).get('time_based_exit', {})
        max_duration = position_config.get('max_position_duration', 3600)
        
        opened_at = position.get('opened_at')
        if not opened_at:
            return False
        
        current_time = datetime.now(timezone.utc)
        duration = (current_time - opened_at).total_seconds()
        
        return duration >= max_duration

    def _is_trailing_stop_active(self, position: Dict, current_price: float, entry_price: float, is_long: bool) -> bool:
        activation_threshold = self._safe_float(position.get('trailing_stop_activation_threshold'), 0.0)
        if activation_threshold is None:
            activation_threshold = 0.0
        if activation_threshold <= 0:
            position['trailing_stop_activated'] = True
            return True
        if position.get('trailing_stop_activated'):
            return True
        if entry_price <= 0 or current_price <= 0:
            return False
        activation_price = entry_price * (1 + activation_threshold) if is_long else entry_price * (1 - activation_threshold)
        if (is_long and current_price >= activation_price) or (not is_long and current_price <= activation_price):
            position['trailing_stop_activated'] = True
            return True
        return False
    
    async def manage_position_exits(self, position_id: str) -> Dict[str, Any]:
        """
        Check if position should exit based on stop loss, take profit, or trailing stop.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Dict with should_exit (bool), exit_reason (str), exit_price (float)
        """
        try:
            if position_id not in self.positions:
                return {'should_exit': False, 'reason': 'Position not found'}
            
            position = self.positions[position_id]
            current_price = position.get('exit_price', position.get('current_price', 0))
            
            if current_price <= 0:
                return {'should_exit': False, 'reason': 'No current price'}
            
            side = position.get('side', 'long')
            stop_loss = position.get('stop_loss', 0)
            take_profit = position.get('take_profit', 0)
            entry_price = position.get('entry_price', 0)
            is_long = side in self.LONG_SIDES

            config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
            exit_guardrails = config.get('position_management', {}).get('exit_guardrails', {})
            eps = self._safe_float(exit_guardrails.get('eps'), 0.0) or 0.0
            if eps < 0:
                eps = 0.0

            trailing_enabled = bool(position.get('trailing_stop_enabled', False))
            trailing_active = bool(position.get('trailing_stop_activated', False))
            profit_protecting = False
            if trailing_enabled and trailing_active and entry_price > 0 and stop_loss > 0:
                if is_long and stop_loss > entry_price:
                    profit_protecting = True
                elif not is_long and stop_loss < entry_price:
                    profit_protecting = True

            trailing_context = ""
            if profit_protecting:
                trail_dist = position.get('trailing_stop_distance', 0.0) or 0.0
                highest_price = position.get('highest_price', entry_price)
                lowest_price = position.get('lowest_price', entry_price)
                trailing_context = (
                    f"\n   stop_mode=trailing trailing_active=true trail_dist={trail_dist:.4f}\n"
                    f"   highest={highest_price:.2f} lowest={lowest_price:.2f} entry={entry_price:.2f} price={current_price:.2f}"
                )
            
            # If trailing-stop is enabled, evaluate it BEFORE the generic stop-loss check so we can
            # classify the exit as `trailing_stop` (and have it counted correctly in summaries).
            if trailing_enabled:
                trailing_distance = self._safe_float(position.get('trailing_stop_distance'), 0.02) or 0.02
                if trailing_distance < 0:
                    trailing_distance = 0.0
                if is_long:
                    highest_price = position.get('highest_price', entry_price)
                    if current_price > highest_price:
                        position['highest_price'] = current_price
                        highest_price = current_price
                    trailing_stop_level = highest_price * (1 - trailing_distance)
                    if self._is_trailing_stop_active(position, current_price, entry_price, True) and current_price <= trailing_stop_level:
                        return {
                            'should_exit': True,
                            'exit_reason': ExitReason.TRAILING_STOP.value,
                            'exit_price': current_price
                        }
                else:  # short position
                    lowest_price = position.get('lowest_price', entry_price)
                    if current_price < lowest_price:
                        position['lowest_price'] = current_price
                        lowest_price = current_price
                    trailing_stop_level = lowest_price * (1 + trailing_distance)
                    if self._is_trailing_stop_active(position, current_price, entry_price, False) and current_price >= trailing_stop_level:
                        return {
                            'should_exit': True,
                            'exit_reason': ExitReason.TRAILING_STOP.value,
                            'exit_price': current_price
                        }

            # Check stop loss
            if side in ['long', 'buy']:
                stop_threshold = stop_loss * (1 - eps) if stop_loss > 0 else stop_loss
                if stop_loss > 0 and current_price <= stop_threshold:
                    logger.warning(
                        f"🛑 [STOP-LOSS-HIT] {position_id}\n"
                        f"   Current Price: ${current_price:.2f}\n"
                        f"   Stop Loss: ${stop_loss:.2f}\n"
                        f"   PnL: {((current_price - entry_price) / entry_price * 100):+.2f}%"
                        f"{trailing_context}"
                    )
                    return {
                        'should_exit': True,
                        'exit_reason': ExitReason.STOP_LOSS.value,
                        'exit_price': current_price
                    }
            else:  # short
                stop_threshold = stop_loss * (1 + eps) if stop_loss > 0 else stop_loss
                if stop_loss > 0 and current_price >= stop_threshold:
                    logger.warning(
                        f"🛑 [STOP-LOSS-HIT] {position_id}\n"
                        f"   Current Price: ${current_price:.2f}\n"
                        f"   Stop Loss: ${stop_loss:.2f}\n"
                        f"   PnL: {((entry_price - current_price) / entry_price * 100):+.2f}%"
                        f"{trailing_context}"
                    )
                    return {
                        'should_exit': True,
                        'exit_reason': ExitReason.STOP_LOSS.value,
                        'exit_price': current_price
                    }
            
            # Check take profit
            if side in ['long', 'buy']:
                if take_profit > 0 and current_price >= take_profit:
                    logger.info(
                        f"🎯 [TAKE-PROFIT-HIT] {position_id}\n"
                        f"   Current Price: ${current_price:.2f}\n"
                        f"   Take Profit: ${take_profit:.2f}\n"
                        f"   Profit: {((current_price - entry_price) / entry_price * 100):+.2f}%"
                    )
                    return {
                        'should_exit': True,
                        'exit_reason': ExitReason.TAKE_PROFIT.value,
                        'exit_price': current_price
                    }
            else:  # short
                if take_profit > 0 and current_price <= take_profit:
                    logger.info(
                        f"🎯 [TAKE-PROFIT-HIT] {position_id}\n"
                        f"   Current Price: ${current_price:.2f}\n"
                        f"   Take Profit: ${take_profit:.2f}\n"
                        f"   Profit: {((entry_price - current_price) / entry_price * 100):+.2f}%"
                    )
                    return {
                        'should_exit': True,
                        'exit_reason': ExitReason.TAKE_PROFIT.value,
                        'exit_price': current_price
                    }
            
            # Check trailing stop (if enabled)
            if position.get('trailing_stop_enabled', False):
                trailing_distance = self._safe_float(position.get('trailing_stop_distance'), 0.02) or 0.02
                if trailing_distance < 0:
                    trailing_distance = 0.0
                if is_long:
                    highest_price = position.get('highest_price', entry_price)
                    if current_price > highest_price:
                        position['highest_price'] = current_price
                        highest_price = current_price

                    trailing_stop_level = highest_price * (1 - trailing_distance)

                    if self._is_trailing_stop_active(position, current_price, entry_price, True):
                        if current_price <= trailing_stop_level:
                            logger.info(
                                f"📉 [TRAILING-STOP-HIT] {position_id}\n"
                                f"   Highest Price: ${highest_price:.2f}\n"
                                f"   Current Price: ${current_price:.2f}\n"
                                f"   Trailing Stop: ${trailing_stop_level:.2f}"
                            )
                            return {
                                'should_exit': True,
                                'exit_reason': ExitReason.TRAILING_STOP.value,
                                'exit_price': current_price
                            }
                else:  # short position
                    lowest_price = position.get('lowest_price', entry_price)
                    if current_price < lowest_price:
                        position['lowest_price'] = current_price
                        lowest_price = current_price

                    trailing_stop_level = lowest_price * (1 + trailing_distance)

                    if self._is_trailing_stop_active(position, current_price, entry_price, False):
                        if current_price >= trailing_stop_level:
                            logger.info(
                                f"📈 [TRAILING-STOP-HIT] {position_id}\n"
                                f"   Lowest Price: ${lowest_price:.2f}\n"
                                f"   Current Price: ${current_price:.2f}\n"
                                f"   Trailing Stop: ${trailing_stop_level:.2f}"
                            )
                            return {
                                'should_exit': True,
                                'exit_reason': ExitReason.TRAILING_STOP.value,
                                'exit_price': current_price
                            }
            
            return {'should_exit': False, 'reason': 'No exit conditions met'}
        
        except Exception as e:
            logger.error(f"Error checking exit for {position_id}: {e}")
            return {'should_exit': False, 'reason': f'Error: {str(e)}'}
    
    def calculate_position_metrics(self, position_id: str) -> Dict[str, Any]:
        """
        Calculate comprehensive position metrics.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position metrics
        """
        try:
            # Check both active and closed positions
            position = self.positions.get(position_id)
            if not position:
                # Check closed positions
                position = next((p for p in self.closed_positions if p['position_id'] == position_id), None)
            
            if not position:
                return {'success': False, 'reason': 'Position not found'}
            
            # Basic metrics
            metrics = {
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'current_price': position.get('current_price', position.get('exit_price', 0)),
                'amount': position['amount'],
                'status': position['status'],
            }
            
            # P&L metrics
            if position['status'] == PositionStatus.CLOSED.value:
                metrics['realized_pnl'] = position['realized_pnl']
                metrics['return_pct'] = position['return_pct']
                metrics['exit_reason'] = position['exit_reason']
                
                # Calculate holding period
                if 'opened_at' in position and 'closed_at' in position:
                    holding_period = position['closed_at'] - position['opened_at']
                    metrics['holding_period_seconds'] = holding_period.total_seconds()
                    metrics['holding_period_hours'] = holding_period.total_seconds() / 3600
            else:
                metrics['unrealized_pnl'] = position['unrealized_pnl']
                pnl_pct = calculate_pnl_percentage(position['unrealized_pnl'], position['entry_price'], position['amount'])
                metrics['unrealized_pnl_pct'] = pnl_pct
            
            # Risk metrics
            metrics['max_adverse_excursion'] = position['max_adverse_excursion']
            metrics['max_favorable_excursion'] = position['max_favorable_excursion']
            metrics['stop_loss'] = position['stop_loss']
            metrics['take_profit'] = position['take_profit']
            
            # Calculate risk-reward ratio
            if position['side'] in ['long', 'buy']:
                risk = position['entry_price'] - position['stop_loss']
                reward = position['take_profit'] - position['entry_price']
            else:
                risk = position['stop_loss'] - position['entry_price']
                reward = position['entry_price'] - position['take_profit']
            
            if risk > 0:
                metrics['risk_reward_ratio'] = reward / risk
            else:
                metrics['risk_reward_ratio'] = 0
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating position metrics: {e}")
            return {'success': False, 'reason': str(e)}
    
    def enable_trailing_stop(
        self,
        position_id: str,
        trailing_distance: float = 0.02,
        activation_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Enable trailing stop-loss for a position.
        
        Args:
            position_id: Position identifier
            trailing_distance: Trailing distance as decimal (e.g., 0.02 for 2%)
            activation_threshold: Profit threshold before trailing activates (e.g., 0.003 for 0.3%)
            
        Returns:
            Operation result
        """
        return self.configure_trailing_stop(
            position_id,
            enabled=True,
            delta_pct=trailing_distance,
            activation_threshold_pct=activation_threshold,
        )

    def attach_execution_config(self, position_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attach resolved per-position execution configuration and persist snapshot.

        Expected shape (best-effort):
          {
            "profile": str|None,
            "trailing_stop": {"enabled": bool, "delta_pct": float, "activation_threshold_pct": float},
            "dca": {"enabled": bool, ...}
          }
        """
        try:
            if position_id not in self.positions:
                return {"success": False, "reason": "Position not found"}
            if not isinstance(config, dict):
                return {"success": False, "reason": "Execution config must be a dict"}

            position = self.positions[position_id]
            position["execution"] = dict(config)
            self._save_positions_snapshot(force=True)
            return {"success": True, "position_id": position_id}
        except Exception as exc:
            logger.error("Error attaching execution config: %s", exc, exc_info=True)
            return {"success": False, "reason": str(exc)}

    def configure_trailing_stop(
        self,
        position_id: str,
        *,
        enabled: bool,
        delta_pct: float,
        activation_threshold_pct: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Configure trailing stop-loss for a position and persist snapshot.

        Args:
            position_id: Position identifier
            enabled: Enable/disable trailing stop
            delta_pct: Trailing distance as decimal (e.g., 0.02 for 2%)
            activation_threshold_pct: Profit threshold before trailing activates (e.g., 0.003 for 0.3%)
        """
        try:
            if position_id not in self.positions:
                return {"success": False, "reason": "Position not found"}

            position = self.positions[position_id]
            enabled_flag = bool(enabled)
            normalized_distance = self._safe_float(delta_pct, 0.02) or 0.02
            normalized_threshold = self._safe_float(activation_threshold_pct, 0.0) or 0.0
            if normalized_distance < 0:
                normalized_distance = 0.0
            if normalized_threshold < 0:
                normalized_threshold = 0.0

            position["trailing_stop_enabled"] = enabled_flag
            position["trailing_stop_distance"] = normalized_distance
            position["trailing_stop_activation_threshold"] = normalized_threshold
            position["trailing_stop_activated"] = bool(enabled_flag and normalized_threshold <= 0)

            if enabled_flag:
                logger.info(
                    "Trailing stop configured for %s: enabled=true dist=%.4f activation=%.4f",
                    position_id,
                    normalized_distance,
                    normalized_threshold,
                )
            else:
                logger.info("Trailing stop configured for %s: enabled=false", position_id)

            self._save_positions_snapshot(force=True)
            return {"success": True, "position_id": position_id}
        except Exception as exc:
            logger.error("Error configuring trailing stop: %s", exc, exc_info=True)
            return {"success": False, "reason": str(exc)}
    
    def get_all_positions(self) -> Dict[str, Any]:
        """Get all active positions."""
        return {
            'active_positions': list(self.positions.values()),
            'count': len(self.positions)
        }
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        active_count = len(self.positions)
        closed_count = len(self.closed_positions)
        
        # Calculate total P&L
        total_unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.values())
        total_realized_pnl = sum(p['realized_pnl'] for p in self.closed_positions)
        
        return {
            'active_positions': active_count,
            'closed_positions': closed_count,
            'total_positions': active_count + closed_count,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl
        }
    
    def get_exit_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive exit statistics for closed positions.
        Issue #134: Validate exit logic with session summaries.
        Issue #136: Fix KeyError by always including all exit stat keys.
        
        Returns:
            Dictionary with exit statistics including counts by type and win/loss breakdown
        """
        if not self.closed_positions:
            return {
                'total_exits': 0,
                'exits_by_reason': {},
                'stop_loss_count': 0,
                'take_profit_count': 0,
                'trailing_stop_count': 0,
                'manual_close_count': 0,
                'liquidation_count': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_win_pnl': 0.0,
                'total_loss_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        # Count exits by reason
        exits_by_reason = {}
        winning_trades = 0
        losing_trades = 0
        total_win_pnl = 0.0
        total_loss_pnl = 0.0
        
        for position in self.closed_positions:
            exit_reason = position.get('exit_reason', 'unknown')
            exits_by_reason[exit_reason] = exits_by_reason.get(exit_reason, 0) + 1
            
            realized_pnl = position.get('realized_pnl', 0)
            if realized_pnl > 0:
                winning_trades += 1
                total_win_pnl += realized_pnl
            else:
                losing_trades += 1
                total_loss_pnl += realized_pnl
        
        total_exits = len(self.closed_positions)
        win_rate = (winning_trades / total_exits * 100) if total_exits > 0 else 0.0
        avg_win = (total_win_pnl / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (total_loss_pnl / losing_trades) if losing_trades > 0 else 0.0
        
        return {
            'total_exits': total_exits,
            'exits_by_reason': exits_by_reason,
            'stop_loss_count': exits_by_reason.get('stop_loss', 0),
            'take_profit_count': exits_by_reason.get('take_profit', 0),
            'trailing_stop_count': exits_by_reason.get('trailing_stop', 0),
            'manual_close_count': exits_by_reason.get('manual', 0),
            'liquidation_count': exits_by_reason.get('emergency', 0),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_win_pnl + total_loss_pnl,
            'total_win_pnl': total_win_pnl,
            'total_loss_pnl': total_loss_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def log_exit_summary(self):
        """
        Log comprehensive exit summary for the session.
        Issue #134: Enhanced exit logging for validation.
        Issue #136: Use safe dictionary access to prevent KeyError.
        """
        stats = self.get_exit_statistics()
        
        logger.info("\n" + "="*70)
        logger.info("📊 EXIT SUMMARY - Session Statistics")
        logger.info("="*70)
        logger.info(f"Total Exits: {stats.get('total_exits', 0)}")
        logger.info(f"\nExits by Reason:")
        logger.info(f"  🛑 Stop Loss:     {stats.get('stop_loss_count', 0)}")
        logger.info(f"  🎯 Take Profit:   {stats.get('take_profit_count', 0)}")
        logger.info(f"  🚦 Trailing Stop: {stats.get('trailing_stop_count', 0)}")
        
        for reason, count in stats.get('exits_by_reason', {}).items():
            if reason not in ['stop_loss', 'take_profit', 'trailing_stop']:
                logger.info(f"  🔄 {reason.replace('_', ' ').title()}: {count}")
        
        logger.info(f"\nWin/Loss Breakdown:")
        logger.info(f"  ✅ Winning Trades: {stats.get('winning_trades', 0)}")
        logger.info(f"  ❌ Losing Trades:  {stats.get('losing_trades', 0)}")
        logger.info(f"  📈 Win Rate:       {stats.get('win_rate', 0.0):.2f}%")
        
        logger.info(f"\nP&L Summary:")
        logger.info(f"  Total P&L:    ${stats.get('total_pnl', 0.0):+.2f}")
        logger.info(f"  Total Wins:   ${stats.get('total_win_pnl', 0.0):+.2f}")
        logger.info(f"  Total Losses: ${stats.get('total_loss_pnl', 0.0):+.2f}")
        logger.info(f"  Avg Win:      ${stats.get('avg_win', 0.0):+.2f}")
        logger.info(f"  Avg Loss:     ${stats.get('avg_loss', 0.0):+.2f}")
        logger.info("="*70 + "\n")
        self.log_individual_trade_history()

    def log_individual_trade_history(self, limit: int = 20):
        trades = self.closed_positions[-limit:]
        if not trades:
            logger.info("No closed trades available for individual history table.")
            return

        header = "=" * 120
        logger.info(header)
        logger.info(f" INDIVIDUAL TRADE HISTORY (Last {len(trades)})")
        logger.info(header)
        logger.info(
            "ID       STRATEGY         SIDE   ENTRY       EXIT        P&L USD     P&L %   R:R   REASON         DUR(min) REGIME    CONF"
        )
        logger.info("-" * 120)

        stats = self.get_exit_statistics()

        for trade in trades:
            trade_id = trade.get('trade_id') or trade.get('position_id', 'n/a')
            strategy = trade.get('strategy_name') or trade.get('strategy', 'unknown')
            side = (trade.get('side') or 'unknown').upper()[:6]
            entry_price = trade.get('entry_price', 0.0)
            exit_price = trade.get('exit_price', 0.0)
            pnl = trade.get('realized_pnl', 0.0)
            pnl_pct = trade.get('return_pct') if trade.get('return_pct') is not None else 0.0
            risk_usd = trade.get('risk_usd', 0.0)
            rr = round(pnl / risk_usd, 2) if risk_usd else None
            rr_display = f"{rr:.2f}" if rr is not None else "N/A"
            exit_reason = (trade.get('exit_reason') or 'unknown').upper()
            opened = trade.get('opened_at')
            closed = trade.get('closed_at')
            duration = ((closed - opened).total_seconds() / 60) if opened and closed else 0.0
            regime = trade.get('regime_at_entry') or 'neutral'
            regime_conf = trade.get('regime_confidence') if trade.get('regime_confidence') is not None else 0.0

            logger.info(
                f"{trade_id:<10} {strategy:<16} {side:<6} {entry_price:>10.2f} {exit_price:>10.2f} "
                f"{pnl:>10.4f} {pnl_pct:>+7.2f}% {rr_display:>5} {exit_reason:<14} {duration:>9.1f} {regime:<8} "
                f"{(regime_conf if regime_conf is not None else 0.0):>5.2f}"
            )

        logger.info("-" * 120)
        logger.info(
            f"TOTAL P&L: {stats.get('total_pnl', 0.0):+.2f} USDT  |  Win Rate: {stats.get('win_rate', 0.0):.1f}%  |  "
            f"Avg Win: {stats.get('avg_win', 0.0):+.2f}  |  Avg Loss: {stats.get('avg_loss', 0.0):+.2f}"
        )
        logger.info(header)
    
    async def start_exit_monitoring(self):
        """
        Start continuous exit monitoring loop.
        Phase 3.4 - Issue #100: Position Exit Monitoring
        
        Monitors all active positions every 5 seconds for exit conditions.
        """
        if self.monitoring_active:
            logger.warning("Exit monitoring already active")
            return
        
        # Get configuration
        config = self.portfolio_manager.cfg if hasattr(self.portfolio_manager, 'cfg') else {}
        exit_config = config.get('position_management', {}).get('exit_monitoring', {})
        
        if not exit_config.get('enabled', True):
            logger.info("Exit monitoring disabled in configuration")
            return
        
        check_frequency = exit_config.get('check_frequency', 5)
        
        self.monitoring_active = True
        logger.info(f"Starting exit monitoring (check every {check_frequency}s)")
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Check all active positions
                    position_ids = list(self.positions.keys())
                    
                    for position_id in position_ids:
                        try:
                            result = await self.manage_position_exits(position_id)
                            
                            if result.get('should_exit'):
                                exit_reason = result.get('exit_reason', 'unknown')
                                logger.info(f"Position {position_id} triggering exit: {exit_reason}")
                                
                                # Execute actual close
                                close_result = await self.execute_close_position(position_id, exit_reason)
                                
                                if close_result.get('success'):
                                    logger.info(f"✅ Position {position_id} closed successfully via monitoring")
                                else:
                                    logger.error(f"❌ Failed to close position {position_id} via monitoring: {close_result.get('reason')}")
                        
                        except Exception as e:
                            logger.error(f"Error checking position {position_id}: {e}")
                    
                    # Wait before next check
                    await asyncio.sleep(check_frequency)
                    
                except Exception as e:
                    logger.error(f"Error in exit monitoring loop: {e}")
                    await asyncio.sleep(check_frequency)
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(monitoring_loop())
        logger.info("Exit monitoring task started")
    
    async def stop_exit_monitoring(self):
        """Stop exit monitoring loop."""
        if not self.monitoring_active:
            logger.warning("Exit monitoring not active")
            return
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Exit monitoring stopped")
    
    async def execute_close_position(self, position_id: str, reason: str) -> Dict[str, Any]:
        """
        Execute market close for a specific position.
        
        Args:
            position_id: Position identifier
            reason: Reason for closing
            
        Returns:
            Close result
        """
        if position_id not in self.positions:
            return {'success': False, 'reason': 'Position not found'}
            
        position = self.positions[position_id]
        symbol = position['symbol']
        amount = position['amount']
        side = position['side']
        close_side = 'sell' if side in ['long', 'buy'] else 'buy'
        exchange = position.get('exchange')
        
        logger.info(f"Executing close for {position_id} ({reason})")
        
        try:
            # Place market close order
            order_req = {
                'symbol': symbol,
                'side': close_side,
                'amount': amount,
                'type': 'market',
                'exchange': exchange,
                'params': {'reduceOnly': True}
            }
            
            # Use order manager
            if self.order_manager:
                # Pass exchange_clients if available in portfolio_manager
                exchange_clients = getattr(self.portfolio_manager, 'exchange_clients', None)
                
                result = await self.order_manager.place_order(
                    order_req, 
                    execution_algo='market',
                    exchange_clients=exchange_clients
                )
                
                if result.get('success'):
                    exit_price = result.get('avg_price', position.get('current_price'))
                    return await self.close_position(position_id, exit_price, reason)
                else:
                    return {'success': False, 'reason': f"Order failed: {result.get('reason')}"}
            else:
                # Simulation / No order manager
                exit_price = position.get('current_price')
                return await self.close_position(position_id, exit_price, reason)
                
        except Exception as e:
            logger.error(f"Failed to execute close for {position_id}: {e}")
            return {'success': False, 'reason': str(e)}
    
    async def sync_positions(self, exchange_clients: Dict[str, Any]):
        """
        Sync open positions from exchanges on startup.
        Issue #168: Position Sync
        """
        logger.info("🔄 Syncing positions from exchanges...")
        
        for ex_name, client in exchange_clients.items():
            try:
                if ex_name == 'bingx':
                    # Use BingX specific method
                    if hasattr(client, 'get_bingx_positions'):
                        response = client.get_bingx_positions()
                        if response.get('code') == 0:
                            positions_data = response.get('data', [])
                            for pos_data in positions_data:
                                await self._import_bingx_position(pos_data, client)
                    else:
                        # Fallback to CCXT fetch_positions if available
                        if hasattr(client, 'fetch_positions'):
                            positions = await asyncio.to_thread(client.fetch_positions)
                            for pos in positions:
                                await self._import_ccxt_position(pos, ex_name)
                else:
                    # Generic CCXT sync
                    if hasattr(client, 'fetch_positions'):
                        positions = await asyncio.to_thread(client.fetch_positions)
                        for pos in positions:
                            await self._import_ccxt_position(pos, ex_name)
                            
            except Exception as e:
                logger.error(f"Failed to sync positions from {ex_name}: {e}")
        
        logger.info(f"✅ Position sync complete. Active positions: {len(self.positions)}")

    async def _import_bingx_position(self, pos_data: Dict, client: Any):
        """Import raw BingX position data."""
        try:
            # BingX format: symbol="BTC-USDT", positionAmt="0.001", ...
            raw_symbol = pos_data.get('symbol')
            if not raw_symbol:
                return

            amount = float(pos_data.get('positionAmt', 0))
            
            if amount == 0:
                return
            
            # Convert symbol
            symbol = raw_symbol.replace('-', '/')
            if 'USDT' in symbol and ':' not in symbol:
                symbol += ':USDT'
            
            position_id = f"imported_{raw_symbol}_{int(time.time())}"
            side = 'long' if amount > 0 else 'short'
            entry_price = float(pos_data.get('avgPrice', 0) or pos_data.get('entryPrice', 0))
            
            # Create position record
            position = {
                'position_id': position_id,
                'trade_id': self._generate_trade_id(),
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': entry_price, # Will be updated by monitor
                'amount': abs(amount),
                'size': abs(amount),
                'initial_amount': abs(amount),
                'stop_loss': 0.0, # Unknown
                'take_profit': 0.0, # Unknown
                'status': PositionStatus.OPEN.value,
                'opened_at': datetime.now(timezone.utc), # Unknown, use now
                'entry_time_iso': datetime.now(timezone.utc).isoformat(),
                'open_timestamp': time.time(),
                'strategy': 'manual_or_imported',
                'exchange': 'bingx',
                'unrealized_pnl': float(pos_data.get('unrealizedProfit', 0)),
                'realized_pnl': 0.0,
                'trailing_stop_enabled': False,
                'imported': True
            }
            
            # Register
            self.positions[position_id] = position
            self.risk_manager.register_position(position_id, position)
            if self.portfolio_manager:
                self.portfolio_manager.register_position(position_id, position)
                
            logger.info(f"📥 Imported position: {symbol} {side} {abs(amount)}")
            
        except Exception as e:
            logger.error(f"Error importing BingX position: {e}")

    async def _import_ccxt_position(self, pos: Dict, exchange_name: str):
        """Import CCXT unified position data."""
        try:
            amount = float(pos.get('contracts', 0) or pos.get('amount', 0))
            if amount == 0:
                return
                
            symbol = pos.get('symbol')
            side = pos.get('side') # 'long' or 'short'
            if not side:
                side = 'long' if amount > 0 else 'short'
            
            position_id = f"imported_{symbol}_{int(time.time())}"
            entry_price = float(pos.get('entryPrice', 0))
            
            position = {
                'position_id': position_id,
                'trade_id': self._generate_trade_id(),
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'current_price': entry_price,
                'amount': abs(amount),
                'size': abs(amount),
                'initial_amount': abs(amount),
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'status': PositionStatus.OPEN.value,
                'opened_at': datetime.now(timezone.utc),
                'entry_time_iso': datetime.now(timezone.utc).isoformat(),
                'open_timestamp': time.time(),
                'strategy': 'manual_or_imported',
                'exchange': exchange_name,
                'unrealized_pnl': float(pos.get('unrealizedPnl', 0) or 0),
                'realized_pnl': 0.0,
                'trailing_stop_enabled': False,
                'imported': True
            }
            
            self.positions[position_id] = position
            self.risk_manager.register_position(position_id, position)
            if self.portfolio_manager:
                self.portfolio_manager.register_position(position_id, position)
                
            logger.info(f"📥 Imported position: {symbol} {side} {abs(amount)}")
            
        except Exception as e:
            logger.error(f"Error importing CCXT position: {e}")

    def _calculate_rl_reward(self, pnl: float, return_pct: float, exit_reason: str) -> float:
        """Calculate reward for RL agent based on trade outcome."""
        reward = return_pct / 10.0  # Base reward
        
        # Modifiers
        if exit_reason == ExitReason.TAKE_PROFIT.value:
            reward += 0.2  # Bonus for TP
        elif exit_reason == ExitReason.STOP_LOSS.value:
            reward -= 0.1  # Small penalty for SL
        
        return max(-2.0, min(2.0, reward))  # Clipped
