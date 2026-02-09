from __future__ import annotations

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
