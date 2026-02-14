from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Literal, Tuple

PYDANTIC_AVAILABLE = False

try:  # Prefer Pydantic v2 (the project currently vendors a minimal YAML parser).
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

    PYDANTIC_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may be absent in some sandboxes
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[override]
        return default

    class ValidationError(Exception):  # type: ignore[no-redef]
        pass

    def field_validator(*_: Any, **__: Any):  # type: ignore[override]
        def decorator(fn):
            return fn

        return decorator

    def model_validator(*_: Any, **__: Any):  # type: ignore[override]
        def decorator(fn):
            return fn

        return decorator


class ConfigSafetyError(ValueError):
    """Raised when configuration violates fail-fast safety invariants."""


def _is_valid_canary_symbol_token(token: str) -> bool:
    """Conservative symbol token validation for rollout canary lists."""
    if token == "*":
        return True
    if not token:
        return False
    if any(ch.isspace() for ch in token):
        return False
    return True


def _validate_promote_override_rollout(config_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    strategies = config_data.get("strategies")
    if not isinstance(strategies, dict):
        return errors

    mr_cfg = strategies.get("mean_reversion")
    if mr_cfg is None:
        return errors
    if not isinstance(mr_cfg, dict):
        errors.append(("strategies.mean_reversion", "must be a mapping/object"))
        return errors

    fw_cfg = mr_cfg.get("fast_watch")
    if fw_cfg is None:
        return errors
    if not isinstance(fw_cfg, dict):
        errors.append(("strategies.mean_reversion.fast_watch", "must be a mapping/object"))
        return errors

    po_cfg = fw_cfg.get("promote_override")
    if po_cfg is None:
        return errors
    if not isinstance(po_cfg, dict):
        errors.append(("strategies.mean_reversion.fast_watch.promote_override", "must be a mapping/object"))
        return errors

    mode = po_cfg.get("mode")
    if mode is not None:
        if not isinstance(mode, str):
            errors.append(
                (
                    "strategies.mean_reversion.fast_watch.promote_override.mode",
                    "must be one of: observe|enforce|off|disabled",
                )
            )
        else:
            mode_norm = mode.strip().lower()
            allowed_modes = {"observe", "enforce", "off", "disabled"}
            if mode_norm not in allowed_modes:
                errors.append(
                    (
                        "strategies.mean_reversion.fast_watch.promote_override.mode",
                        f"invalid value '{mode}'; allowed: observe|enforce|off|disabled",
                    )
                )

    canary_symbols = po_cfg.get("canary_symbols")
    canary_path = "strategies.mean_reversion.fast_watch.promote_override.canary_symbols"
    if canary_symbols is None:
        return errors
    if isinstance(canary_symbols, str):
        raw = str(canary_symbols)
        parts = [p.strip() for p in raw.split(",")]
        tokens = [p for p in parts if p]
        if raw.strip() and not tokens:
            errors.append((canary_path, "CSV string must contain at least one non-empty symbol token"))
        for idx, token in enumerate(tokens):
            if not _is_valid_canary_symbol_token(token):
                errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
        return errors
    if isinstance(canary_symbols, (list, tuple, set)):
        for idx, token in enumerate(canary_symbols):
            if not isinstance(token, str):
                errors.append((f"{canary_path}[{idx}]", "must be a string symbol token"))
                continue
            token_norm = token.strip()
            if not _is_valid_canary_symbol_token(token_norm):
                errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
        return errors

    errors.append((canary_path, "must be list[str] or CSV string"))
    return errors


def _validate_rsi_zone_router(config_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    strategies = config_data.get("strategies")
    if not isinstance(strategies, dict):
        return errors

    router_cfg = strategies.get("rsi_zone_router")
    if router_cfg is None:
        return errors
    if not isinstance(router_cfg, dict):
        errors.append(("strategies.rsi_zone_router", "must be a mapping/object"))
        return errors

    source_cfg = router_cfg.get("source")
    if source_cfg is not None and not isinstance(source_cfg, dict):
        errors.append(("strategies.rsi_zone_router.source", "must be a mapping/object"))
    elif isinstance(source_cfg, dict):
        mode = source_cfg.get("mode")
        if mode is not None:
            if not isinstance(mode, str):
                errors.append(("strategies.rsi_zone_router.source.mode", "must be one of: slow_only|consensus"))
            else:
                mode_norm = mode.strip().lower()
                if mode_norm not in {"slow_only", "consensus"}:
                    errors.append(
                        (
                            "strategies.rsi_zone_router.source.mode",
                            f"invalid value '{mode}'; allowed: slow_only|consensus",
                        )
                    )

    thresholds_cfg = router_cfg.get("thresholds")
    if thresholds_cfg is not None and not isinstance(thresholds_cfg, dict):
        errors.append(("strategies.rsi_zone_router.thresholds", "must be a mapping/object"))
    elif isinstance(thresholds_cfg, dict):
        for key in ("ob_floor", "ob_cap", "str_floor", "str_cap", "min_gap"):
            raw = thresholds_cfg.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except Exception:
                errors.append((f"strategies.rsi_zone_router.thresholds.{key}", "must be a finite number"))
                continue
            if value != value or value in (float("inf"), float("-inf")):
                errors.append((f"strategies.rsi_zone_router.thresholds.{key}", "must be a finite number"))
        raw_min_gap = thresholds_cfg.get("min_gap")
        if raw_min_gap is not None:
            try:
                min_gap = float(raw_min_gap)
                if min_gap <= 0:
                    errors.append(("strategies.rsi_zone_router.thresholds.min_gap", "must be > 0"))
            except Exception:
                pass

    transition_cfg = router_cfg.get("transition")
    if transition_cfg is not None and not isinstance(transition_cfg, dict):
        errors.append(("strategies.rsi_zone_router.transition", "must be a mapping/object"))
    elif isinstance(transition_cfg, dict):
        raw_width = transition_cfg.get("width")
        if raw_width is not None:
            try:
                width = float(raw_width)
                if width < 0:
                    errors.append(("strategies.rsi_zone_router.transition.width", "must be >= 0"))
                elif width != width or width in (float("inf"), float("-inf")):
                    errors.append(("strategies.rsi_zone_router.transition.width", "must be a finite number"))
            except Exception:
                errors.append(("strategies.rsi_zone_router.transition.width", "must be a finite number"))

    return errors


def _validate_level_zone_router(config_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    strategies = config_data.get("strategies")
    if not isinstance(strategies, dict):
        return errors

    router_cfg = strategies.get("level_zone_router")
    if router_cfg is None:
        return errors
    if not isinstance(router_cfg, dict):
        errors.append(("strategies.level_zone_router", "must be a mapping/object"))
        return errors

    source_cfg = router_cfg.get("source")
    if source_cfg is not None and not isinstance(source_cfg, dict):
        errors.append(("strategies.level_zone_router.source", "must be a mapping/object"))
    elif isinstance(source_cfg, dict):
        mode = source_cfg.get("mode")
        if mode is not None:
            if not isinstance(mode, str):
                errors.append(("strategies.level_zone_router.source.mode", "must be one of: single_tf|consensus"))
            else:
                mode_norm = mode.strip().lower()
                if mode_norm not in {"single_tf", "consensus"}:
                    errors.append(
                        (
                            "strategies.level_zone_router.source.mode",
                            f"invalid value '{mode}'; allowed: single_tf|consensus",
                        )
                    )
        tfs = source_cfg.get("timeframes")
        if tfs is not None:
            if isinstance(tfs, str):
                if not [part.strip() for part in tfs.split(",") if part.strip()]:
                    errors.append(("strategies.level_zone_router.source.timeframes", "must contain at least one timeframe"))
            elif isinstance(tfs, (list, tuple)):
                if not [str(item).strip() for item in tfs if str(item).strip()]:
                    errors.append(("strategies.level_zone_router.source.timeframes", "must contain at least one timeframe"))
            else:
                errors.append(("strategies.level_zone_router.source.timeframes", "must be list[str] or CSV string"))

    levels_cfg = router_cfg.get("levels")
    if levels_cfg is not None and not isinstance(levels_cfg, dict):
        errors.append(("strategies.level_zone_router.levels", "must be a mapping/object"))
    elif isinstance(levels_cfg, dict):
        float_keys = ("band_pct", "smc_cluster_pct", "touch_proximity_bps")
        int_keys = ("pivot_left", "pivot_right", "lookback_bars", "min_cluster_n", "kmin", "kmax")
        for key in float_keys:
            raw = levels_cfg.get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
                if not math.isfinite(val):
                    raise ValueError
                if val <= 0:
                    errors.append((f"strategies.level_zone_router.levels.{key}", "must be > 0"))
            except Exception:
                errors.append((f"strategies.level_zone_router.levels.{key}", "must be a finite number"))
        for key in int_keys:
            raw = levels_cfg.get(key)
            if raw is None:
                continue
            try:
                val = int(raw)
                if val <= 0:
                    errors.append((f"strategies.level_zone_router.levels.{key}", "must be > 0"))
            except Exception:
                errors.append((f"strategies.level_zone_router.levels.{key}", "must be an integer > 0"))

    zones_cfg = router_cfg.get("zones")
    if zones_cfg is not None and not isinstance(zones_cfg, dict):
        errors.append(("strategies.level_zone_router.zones", "must be a mapping/object"))
    elif isinstance(zones_cfg, dict):
        for key in ("near_level_bps", "decision_zone_low", "decision_zone_high"):
            raw = zones_cfg.get(key)
            if raw is None:
                continue
            try:
                val = float(raw)
                if not math.isfinite(val):
                    raise ValueError
            except Exception:
                errors.append((f"strategies.level_zone_router.zones.{key}", "must be a finite number"))
        low_raw = zones_cfg.get("decision_zone_low")
        high_raw = zones_cfg.get("decision_zone_high")
        try:
            if low_raw is not None:
                low = float(low_raw)
                if low < 0 or low > 1:
                    errors.append(("strategies.level_zone_router.zones.decision_zone_low", "must be in [0, 1]"))
        except Exception:
            pass
        try:
            if high_raw is not None:
                high = float(high_raw)
                if high < 0 or high > 1:
                    errors.append(("strategies.level_zone_router.zones.decision_zone_high", "must be in [0, 1]"))
        except Exception:
            pass
        try:
            if low_raw is not None and high_raw is not None and float(low_raw) > float(high_raw):
                errors.append(("strategies.level_zone_router.zones", "decision_zone_low must be <= decision_zone_high"))
        except Exception:
            pass

    breakout_cfg = router_cfg.get("breakout")
    if breakout_cfg is not None and not isinstance(breakout_cfg, dict):
        errors.append(("strategies.level_zone_router.breakout", "must be a mapping/object"))
    elif isinstance(breakout_cfg, dict):
        raw_bars = breakout_cfg.get("min_close_bars")
        if raw_bars is not None:
            try:
                bars = int(raw_bars)
                if bars <= 0:
                    errors.append(("strategies.level_zone_router.breakout.min_close_bars", "must be > 0"))
            except Exception:
                errors.append(("strategies.level_zone_router.breakout.min_close_bars", "must be an integer > 0"))
        raw_mult = breakout_cfg.get("min_volume_mult")
        if raw_mult is not None:
            try:
                mult = float(raw_mult)
                if (not math.isfinite(mult)) or mult <= 0:
                    errors.append(("strategies.level_zone_router.breakout.min_volume_mult", "must be > 0"))
            except Exception:
                errors.append(("strategies.level_zone_router.breakout.min_volume_mult", "must be a finite number"))

    rollout_cfg = router_cfg.get("rollout")
    if rollout_cfg is not None and not isinstance(rollout_cfg, dict):
        errors.append(("strategies.level_zone_router.rollout", "must be a mapping/object"))
    elif isinstance(rollout_cfg, dict):
        mode = rollout_cfg.get("mode")
        if mode is not None:
            if not isinstance(mode, str):
                errors.append(("strategies.level_zone_router.rollout.mode", "must be one of: enforce|observe|off|disabled"))
            else:
                mode_norm = mode.strip().lower()
                if mode_norm not in {"enforce", "observe", "off", "disabled"}:
                    errors.append(
                        (
                            "strategies.level_zone_router.rollout.mode",
                            f"invalid value '{mode}'; allowed: enforce|observe|off|disabled",
                        )
                    )
        canary_symbols = rollout_cfg.get("canary_symbols")
        canary_path = "strategies.level_zone_router.rollout.canary_symbols"
        if canary_symbols is not None:
            if isinstance(canary_symbols, str):
                tokens = [part.strip() for part in canary_symbols.split(",") if part.strip()]
                if canary_symbols.strip() and not tokens:
                    errors.append((canary_path, "CSV string must contain at least one non-empty symbol token"))
                for idx, token in enumerate(tokens):
                    if not _is_valid_canary_symbol_token(token):
                        errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
            elif isinstance(canary_symbols, (list, tuple, set)):
                for idx, token in enumerate(canary_symbols):
                    if not isinstance(token, str):
                        errors.append((f"{canary_path}[{idx}]", "must be a string symbol token"))
                        continue
                    token_norm = token.strip()
                    if not _is_valid_canary_symbol_token(token_norm):
                        errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
            else:
                errors.append((canary_path, "must be list[str] or CSV string"))

    return errors


def _validate_directional_bias_config(config_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    errors: List[Tuple[str, str]] = []
    signals = config_data.get("signals")
    if not isinstance(signals, dict):
        return errors

    cfg = signals.get("directional_bias")
    if cfg is None:
        return errors
    if not isinstance(cfg, dict):
        errors.append(("signals.directional_bias", "must be a mapping/object"))
        return errors

    mode = cfg.get("mode")
    if mode is not None:
        if not isinstance(mode, str):
            errors.append(("signals.directional_bias.mode", "must be one of: quality_adjust_only|off|disabled"))
        else:
            mode_norm = mode.strip().lower()
            if mode_norm not in {"quality_adjust_only", "off", "disabled"}:
                errors.append(
                    (
                        "signals.directional_bias.mode",
                        f"invalid value '{mode}'; allowed: quality_adjust_only|off|disabled",
                    )
                )

    for key in ("weight", "max_quality_delta", "at_level_penalty"):
        raw = cfg.get(key)
        if raw is None:
            continue
        try:
            val = float(raw)
            if not math.isfinite(val):
                raise ValueError
            if val < 0:
                errors.append((f"signals.directional_bias.{key}", "must be >= 0"))
        except Exception:
            errors.append((f"signals.directional_bias.{key}", "must be a finite number"))

    raw_max_delta = cfg.get("max_quality_delta")
    if raw_max_delta is not None:
        try:
            max_delta = float(raw_max_delta)
            if max_delta > 1.0:
                errors.append(("signals.directional_bias.max_quality_delta", "must be <= 1.0"))
        except Exception:
            pass

    rollout_cfg = cfg.get("rollout")
    if rollout_cfg is not None and not isinstance(rollout_cfg, dict):
        errors.append(("signals.directional_bias.rollout", "must be a mapping/object"))
    elif isinstance(rollout_cfg, dict):
        mode = rollout_cfg.get("mode")
        if mode is not None:
            if not isinstance(mode, str):
                errors.append(("signals.directional_bias.rollout.mode", "must be one of: enforce|observe|off|disabled"))
            else:
                mode_norm = mode.strip().lower()
                if mode_norm not in {"enforce", "observe", "off", "disabled"}:
                    errors.append(
                        (
                            "signals.directional_bias.rollout.mode",
                            f"invalid value '{mode}'; allowed: enforce|observe|off|disabled",
                        )
                    )
        canary_symbols = rollout_cfg.get("canary_symbols")
        canary_path = "signals.directional_bias.rollout.canary_symbols"
        if canary_symbols is not None:
            if isinstance(canary_symbols, str):
                tokens = [part.strip() for part in canary_symbols.split(",") if part.strip()]
                if canary_symbols.strip() and not tokens:
                    errors.append((canary_path, "CSV string must contain at least one non-empty symbol token"))
                for idx, token in enumerate(tokens):
                    if not _is_valid_canary_symbol_token(token):
                        errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
            elif isinstance(canary_symbols, (list, tuple, set)):
                for idx, token in enumerate(canary_symbols):
                    if not isinstance(token, str):
                        errors.append((f"{canary_path}[{idx}]", "must be a string symbol token"))
                        continue
                    token_norm = token.strip()
                    if not _is_valid_canary_symbol_token(token_norm):
                        errors.append((f"{canary_path}[{idx}]", f"invalid symbol token '{token}'"))
            else:
                errors.append((canary_path, "must be list[str] or CSV string"))

    return errors


def _iter_pct_violations(config: Any) -> List[Tuple[str, str]]:
    violations: List[Tuple[str, str]] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = f"{path}.{key}" if path else str(key)

                # NOTE: `risk.max_notional_pct_per_trade` is treated as a raw multiplier (e.g., 10.0 == 10x),
                # despite the `_pct` suffix. Exempt it from the generic 0-1 invariant enforcement.
                if key == "max_notional_pct_per_trade":
                    walk(value, child)
                    continue

                if isinstance(key, str) and key.endswith("_pct"):
                    if isinstance(value, bool) or value is None:
                        violations.append((child, "must be a number in [0, 1] (decimal fraction)"))
                    elif isinstance(value, (int, float)):
                        numeric = float(value)
                        if numeric < 0.0 or numeric > 1.0:
                            violations.append((child, f"must be in [0, 1] (decimal fraction), got {numeric}"))
                    else:
                        violations.append((child, "must be a number in [0, 1] (decimal fraction)"))

                # Normalize quality_threshold semantics to 0-1 only (no 0-100).
                if key == "quality_threshold":
                    if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
                        violations.append((child, "must be a number in [0, 1]"))
                    else:
                        numeric = float(value)
                        if numeric < 0.0 or numeric > 1.0:
                            violations.append((child, f"must be in [0, 1], got {numeric}"))

                walk(value, child)
        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                walk(value, f"{path}[{idx}]")

    walk(config, "")
    return violations


def validate_config_safety(config_data: Dict[str, Any]) -> None:
    """Fail-fast validation for critical config invariants (works without Pydantic)."""
    errors: List[Tuple[str, str]] = []

    risk = config_data.get("risk")
    if not isinstance(risk, dict):
        errors.append(("risk", "must be a mapping/object"))
    else:
        if "max_position_size" in risk:
            try:
                max_pos = float(risk.get("max_position_size"))
            except Exception:
                errors.append(("risk.max_position_size", "must be a number in [0, 1]"))
            else:
                if max_pos < 0.0 or max_pos > 1.0:
                    errors.append(("risk.max_position_size", f"must be in [0, 1], got {max_pos}"))
        if "class_limits" in risk:
            errors.append(("risk.class_limits", "is removed/unused; delete this block"))
        if "max_notional_per_trade" in risk:
            errors.append((
                "risk.max_notional_per_trade",
                "is removed; use risk.max_notional_pct_per_trade (multiplier-based cap) instead",
            ))
        if "max_position_size_pct" in risk:
            errors.append((
                "risk.max_position_size_pct",
                "is ignored by RiskConfiguration; rename to risk.max_position_size",
            ))

    universe = config_data.get("universe")
    if not isinstance(universe, dict):
        errors.append(("universe", "must be a mapping/object"))
    else:
        prefetch = universe.get("prefetch")
        if not isinstance(prefetch, dict):
            errors.append(("universe.prefetch", "must be a mapping/object"))
        else:
            count = prefetch.get("startup_candle_count", 0)
            try:
                count_int = int(count)
            except Exception:
                count_int = 0
            if count_int < 2000:
                errors.append((
                    "universe.prefetch.startup_candle_count",
                    f"too low ({count_int}); must be >= 2000 (VWAP lookback=1440 + buffer)",
                ))

    errors.extend(_iter_pct_violations(config_data))
    errors.extend(_validate_promote_override_rollout(config_data))
    errors.extend(_validate_rsi_zone_router(config_data))
    errors.extend(_validate_level_zone_router(config_data))
    errors.extend(_validate_directional_bias_config(config_data))

    if errors:
        rendered = "\n".join(f"- {path}: {msg}" for path, msg in errors)
        raise ConfigSafetyError(f"Configuration validation failed:\n{rendered}")


# ==============================================================================
# Pydantic schema (optional)
# ==============================================================================


class DynamicScaling(BaseModel):
    enabled: bool = True
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="0-1 arasında olmalı")
    min_unrealized_pnl_pct: float
    max_additional_positions: int

    model_config = {"extra": "ignore"}


class ConcurrentLimits(BaseModel):
    max_open_positions: int
    max_positions_per_symbol: int
    max_total_risk_pct: float = Field(..., le=1.0)  # %100'den büyük olamaz (fraction)
    correlation_bucket_threshold: float
    dynamic_scaling: DynamicScaling

    model_config = {"extra": "ignore"}


class RiskConfiguration(BaseModel):
    size_planner_enabled: bool
    equity_usd: float = Field(..., gt=0)
    per_trade_risk_pct: float = Field(..., le=0.05, description="Trade başı risk %5'i geçemez")
    daily_loss_limit_pct: float = Field(..., le=0.20, description="Günlük zarar limiti %20'yi geçemez")
    risk_usd_cap: Optional[float] = Field(
        default=None,
        description="Optional absolute USD risk ceiling per trade (None disables the cap).",
    )

    max_position_size: float = Field(default=0.25, le=1.0)
    max_notional_pct_per_trade: float = Field(
        ...,
        le=50.0,
        description="NOTE: Treated as a raw leverage multiplier (e.g., 10.0 = 10x equity), despite the '_pct' suffix. Do not normalize.",
    )
    position_size_policy: Literal["clip", "reject"]

    concurrent_limits: ConcurrentLimits

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def _normalize_and_forbid(cls, values: Any):
        if not isinstance(values, dict):
            return values

        # Hard-forbid dead / misleading blocks.
        if "class_limits" in values:
            raise ValueError("🚨 GÜVENLİK HATASI: 'class_limits' bloğu kaldırıldı. Lütfen config dosyasından silin.")
        if "max_notional_per_trade" in values:
            raise ValueError("🚨 GÜVENLİK HATASI: 'max_notional_per_trade' kaldırıldı. Lütfen 'max_notional_pct_per_trade' kullanın.")

        # Naming mismatch: accept legacy key, but normalize into the canonical key.
        if "max_position_size" not in values and "max_position_size_pct" in values:
            values["max_position_size"] = values.get("max_position_size_pct")

        # Enforce canonical key in production configs (legacy key should not be present).
        if "max_position_size_pct" in values:
            raise ValueError("🚨 KRİTİK: 'max_position_size_pct' ignored; rename to 'max_position_size'.")

        return values


class TradingSettings(BaseModel):
    order_type: Literal["market", "limit"]
    slippage_buffer: float

    model_config = {"extra": "ignore"}


class UniverseSettings(BaseModel):
    prefetch: Dict[str, Any]

    model_config = {"extra": "allow"}

    @field_validator("prefetch")
    def validate_startup_candles(cls, v: Any):
        if not isinstance(v, dict):
            raise ValueError("universe.prefetch must be an object")
        count = v.get("startup_candle_count", 0)
        try:
            count_int = int(count)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("startup_candle_count must be an integer") from exc
        if count_int < 2000:
            raise ValueError(
                f"🚨 KRİTİK HATA: startup_candle_count ({count_int}) çok düşük! "
                "VWAP (1440) hesaplanamaz. En az 2000 yapın."
            )
        return v


class MLConfiguration(BaseModel):
    enabled: bool

    model_config = {"extra": "allow"}


class BotConfiguration(BaseModel):
    trading: Optional[TradingSettings] = None
    execution: Optional[Dict[str, Any]] = None  # legacy block kept only for warnings/migration
    risk: RiskConfiguration
    universe: UniverseSettings
    ml: MLConfiguration

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    def _enforce_global_invariants(cls, values: Any):
        if isinstance(values, dict):
            # Global safety checks (also covers unknown blocks).
            validate_config_safety(values)
        return values

    @model_validator(mode="after")
    def enforce_execution_migration(self):
        if self.execution and isinstance(self.execution, dict) and self.execution.get("order_type") == "market":
            if self.execution.get("time_in_force") == "IOC":
                # Keep this as a non-fatal warning; the loader decides whether to fail-fast.
                print("⚠️ UYARI: Market emirlerinde IOC kullanımı risklidir. Lütfen temiz config kullanın.")
        return self


def validate_with_schema(config_data: Dict[str, Any]) -> None:
    """Validate using Pydantic when available; otherwise fall back to manual checks."""
    if not PYDANTIC_AVAILABLE:
        validate_config_safety(config_data)
        return

    # Validation happens during model construction via validators.
    BotConfiguration(**config_data)
