import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from src.core.signal_intents import INTENT_SCALE_IN


class DCAWatcher:
    """
    Minimal DCA scale-in signal generator (v1).

    - Disabled by default (controlled via cfg.dca.enabled).
    - Uses existing position/price data; no queue changes.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        position_manager: Any,
        market_data_pipeline: Any = None,
        portfolio_manager: Any = None,
        price_fetcher=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg or {}
        self.dca_cfg = (self.cfg.get("dca") or {}) if isinstance(self.cfg, dict) else {}
        self.position_manager = position_manager
        self.portfolio_manager = portfolio_manager
        self.market_data_pipeline = market_data_pipeline
        self.price_fetcher = price_fetcher
        self.logger = logger or logging.getLogger(__name__)
        strategy_cfg = self.dca_cfg.get("strategy", {}) if isinstance(self.dca_cfg, dict) else {}
        cooldown = float(strategy_cfg.get("cooldown_seconds", 0) or 0)
        # Poll faster than cooldown but never hammer
        self.poll_interval = max(5.0, min(60.0, cooldown / 3 if cooldown else 15.0))
        self._state: Dict[str, Dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.dca_cfg.get("enabled", False))

    async def run_once(self) -> List[Dict[str, Any]]:
        """Generate DCA scale-in signals based on current positions."""
        if not self.enabled:
            return []

        grouped = self._group_positions_by_symbol()
        signals: List[Dict[str, Any]] = []

        for symbol, positions in grouped.items():
            if not positions:
                continue
            state = self._ensure_state(symbol, positions)
            if not state:
                continue

            maybe_signal = await self._check_dca_trigger(symbol, positions, state)
            if maybe_signal:
                signals.append(maybe_signal)
        return signals

    def _group_positions_by_symbol(self) -> Dict[str, List[Dict[str, Any]]]:
        results: Dict[str, List[Dict[str, Any]]] = {}
        try:
            positions = getattr(self.position_manager, "positions", {}) or {}
            for pos in positions.values():
                symbol = pos.get("symbol")
                if not symbol:
                    continue
                results.setdefault(symbol, []).append(pos)
        except Exception:
            self.logger.debug("DCA watcher could not read positions from PositionManager", exc_info=True)
        return results

    def _ensure_state(self, symbol: str, positions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Establish plan state per symbol using the earliest non-DCA position as anchor.
        """
        base_positions = [p for p in positions if not self._is_dca_position(p)]
        if not base_positions:
            return None

        base = sorted(base_positions, key=lambda p: p.get("entry_time") or p.get("opened_at") or 0)[0]
        anchor_price = base.get("entry_price") or base.get("entry")
        side = (base.get("side") or "").lower()

        try:
            anchor_price = float(anchor_price)
        except (TypeError, ValueError):
            return None
        if anchor_price is None or anchor_price <= 0:
            return None

        notional = self._extract_notional(base)
        if notional <= 0:
            return None

        state = self._state.get(symbol, {})
        if not state:
            state = {
                "plan_id": uuid.uuid4().hex[:10],
                "anchor_price": anchor_price,
                "base_notional": notional,
                "direction": side or base.get("direction") or "long",
                "triggered_layers": [],
                "last_trigger_ts": 0.0,
                "stop_loss": base.get("stop_loss"),
                "take_profit": base.get("take_profit"),
                "timeframe": base.get("timeframe") or base.get("tf"),
                "strategy_name": base.get("strategy_name") or base.get("strategy") or "unknown",
            }
        else:
            # Keep anchor stable for consistency in v1
            state.setdefault("anchor_price", anchor_price)
            state.setdefault("base_notional", notional)
            state.setdefault("direction", side or "long")
        self._state[symbol] = state
        return state

    async def _check_dca_trigger(
        self,
        symbol: str,
        positions: List[Dict[str, Any]],
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        strategy_cfg = self.dca_cfg.get("strategy", {}) if isinstance(self.dca_cfg, dict) else {}
        max_layers_cfg = strategy_cfg.get("max_layers", 0)
        try:
            max_layers_cfg = int(max_layers_cfg)
        except (TypeError, ValueError):
            max_layers_cfg = 0
        allowed_dca_layers = max(0, max_layers_cfg - 1) if max_layers_cfg else max_layers_cfg

        current_layers = len([p for p in positions if self._is_dca_position(p)])
        if allowed_dca_layers and current_layers >= allowed_dca_layers:
            return None

        current_price = await self._get_current_price(symbol)
        if current_price is None:
            return None

        anchor_price = state.get("anchor_price")
        direction = (state.get("direction") or "long").lower()
        price_drop_pct = self._compute_adverse_move(anchor_price, current_price, direction)
        if price_drop_pct is None:
            return None

        step_pct = float(strategy_cfg.get("step_pct", 0.0) or 0.0)
        if step_pct <= 0:
            return None
        next_layer_index = current_layers + 1  # 1-based for DCA layers
        required_drop = step_pct * next_layer_index
        if price_drop_pct < required_drop:
            return None

        cooldown_seconds = float(strategy_cfg.get("cooldown_seconds", 0) or 0)
        now = time.time()
        last_ts = state.get("last_trigger_ts", 0.0)
        if cooldown_seconds and (now - last_ts) < cooldown_seconds:
            return None

        return self._create_dca_signal(
            symbol=symbol,
            current_price=current_price,
            anchor_price=anchor_price,
            direction=direction,
            layer_index=next_layer_index,
            price_drop_pct=price_drop_pct,
            state=state,
            positions=positions,
        )

    def _create_dca_signal(
        self,
        symbol: str,
        current_price: float,
        anchor_price: float,
        direction: str,
        layer_index: int,
        price_drop_pct: float,
        state: Dict[str, Any],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        strategy_cfg = self.dca_cfg.get("strategy", {}) if isinstance(self.dca_cfg, dict) else {}
        weights = strategy_cfg.get("position_weights") or [1.0]
        try:
            weight = float(weights[min(layer_index - 1, len(weights) - 1)])
        except Exception:
            weight = 1.0
        min_volume = float(strategy_cfg.get("min_volume_usdt", 0) or 0)
        base_notional = float(state.get("base_notional") or 0.0)
        volume_usdt = max(base_notional * weight, min_volume)
        amount = volume_usdt / current_price if current_price > 0 else 0.0

        timestamp = time.time()
        reason = f"DCA layer {layer_index} price drop {price_drop_pct:.2%}"
        strategy_name = "dca_watcher"
        base_strategy = state.get("strategy_name")
        if base_strategy and base_strategy != "unknown":
            strategy_name = f"{base_strategy}_dca"

        signal = {
            "symbol": symbol,
            "intent": INTENT_SCALE_IN,
            "scale_profile": "dca",
            "side": direction,
            "entry": current_price,
            "price": current_price,
            "reason": reason,
            "strategy_name": strategy_name,
            "timeframe": state.get("timeframe"),
            "amount": amount,
            "notional": volume_usdt,
            "stop": state.get("stop_loss"),
            "target": state.get("take_profit"),
            "dca_metadata": {
                "profile": "dca",
                "plan_id": state.get("plan_id"),
                "layer_index": layer_index,
                "anchor_price": anchor_price,
                "price_drop_pct": price_drop_pct,
                "volume_usdt": volume_usdt,
                "step_pct": strategy_cfg.get("step_pct"),
                "volume_weight": weight,
                "trigger_type": "price_drop",
                "force_entry": True,
            },
        }

        state.setdefault("triggered_layers", []).append(
            {"layer": layer_index, "price": current_price, "timestamp": timestamp, "volume_usdt": volume_usdt}
        )
        state["last_trigger_ts"] = timestamp

        self.logger.info(
            "DCA signal generated | sym=%s layer=%d drop=%.2f%% vol=%.2f anchor=%.4f",
            symbol,
            layer_index,
            price_drop_pct * 100,
            volume_usdt,
            anchor_price,
        )
        return signal

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        if self.price_fetcher:
            try:
                return await self._maybe_await(self.price_fetcher(symbol))
            except Exception:
                self.logger.debug("Price fetcher failed for %s", symbol, exc_info=True)

        if self.market_data_pipeline and hasattr(self.market_data_pipeline, "get_latest_price"):
            try:
                return await self.market_data_pipeline.get_latest_price(symbol, timeframe="1m")
            except Exception:
                self.logger.debug("MarketDataPipeline price fetch failed for %s", symbol, exc_info=True)

        if self.position_manager and hasattr(self.position_manager, "_get_current_price_from_ws"):
            try:
                return await self.position_manager._get_current_price_from_ws(symbol)
            except Exception:
                self.logger.debug("PositionManager WS price fetch failed for %s", symbol, exc_info=True)
        return None

    @staticmethod
    def _extract_notional(position: Dict[str, Any]) -> float:
        if not position:
            return 0.0
        notional = position.get("notional") or position.get("position_notional")
        if notional is None:
            try:
                entry = float(position.get("entry_price") or position.get("entry") or 0.0)
                qty = float(position.get("amount") or position.get("size") or 0.0)
                notional = entry * qty
            except Exception:
                notional = 0.0
        try:
            return float(notional or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_dca_position(position: Dict[str, Any]) -> bool:
        if not isinstance(position, dict):
            return False
        meta = position.get("dca_metadata") or {}
        profile = position.get("scale_profile") or meta.get("profile")
        return profile == "dca"

    @staticmethod
    def _compute_adverse_move(anchor_price: Optional[float], current_price: Optional[float], direction: str) -> Optional[float]:
        try:
            anchor = float(anchor_price)
            current = float(current_price)
        except (TypeError, ValueError):
            return None
        if anchor <= 0 or current <= 0:
            return None
        if direction in ("short", "sell"):
            return max(0.0, (current - anchor) / anchor)
        return max(0.0, (anchor - current) / anchor)

    @staticmethod
    async def _maybe_await(value):
        if hasattr(value, "__await__"):
            return await value
        return value
