from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)

VALID_MTF_MODES = ("off", "soft", "hard")
VALID_MISSING_POLICIES = ("skip", "reject")

DEFAULT_RSI_15M_MIN = 62.0
DEFAULT_RSI_1H_MAX = 60.0
DEFAULT_MIN_15M_CLOSE_OVER_EMA50_PCT = 0.0
DEFAULT_REQUIRE_1H_BEARISH_EMA_STACK = True
DEFAULT_MTF_MIN_BARS = {
    "rsi": 20,
    "ema21": 30,
    "ema50": 100,
    "ema200": 250,
}


@dataclass(frozen=True)
class MtfMinBars:
    rsi: int
    ema21: int
    ema50: int
    ema200: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "rsi": self.rsi,
            "ema21": self.ema21,
            "ema50": self.ema50,
            "ema200": self.ema200,
        }


@dataclass(frozen=True)
class MtfTimeframePolicy:
    mode: str
    on_missing: str
    missing_is_fatal: bool
    missing_policy: str
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    min_close_over_ema50_pct: float = 0.0
    require_bearish_ema_stack: bool = True


@dataclass(frozen=True)
class MtfConfirmationConfig:
    enabled: bool
    tf_15m: MtfTimeframePolicy
    tf_1h: MtfTimeframePolicy
    min_bars: MtfMinBars
    summary: str


def build_str_mtf_config(
    mtf_cfg: Dict[str, Any],
    *,
    strict: bool = True,
    log: Optional[logging.Logger] = None,
) -> MtfConfirmationConfig:
    if not isinstance(mtf_cfg, dict):
        raise ValueError("STR MTF config must be a dict.")

    log = log or logger

    enabled = mtf_cfg.get("enabled", False)
    legacy_enabled_present = "enabled" in mtf_cfg
    if legacy_enabled_present and ("15m_mode" not in mtf_cfg or "1h_mode" not in mtf_cfg):
        log.warning("Deprecated STR MTF key 'enabled' detected; use 15m_mode/1h_mode.")
    if not isinstance(enabled, bool):
        raise ValueError("STR MTF 'enabled' must be a bool.")

    mode_15m = _normalize_mode(mtf_cfg.get("15m_mode"), default="hard" if enabled else "off", key="15m_mode")
    mode_1h = _normalize_mode(mtf_cfg.get("1h_mode"), default="hard" if enabled else "off", key="1h_mode")

    if not enabled:
        if mtf_cfg.get("15m_mode") not in (None, "off"):
            log.warning("STR MTF 'enabled' is false; forcing 15m_mode=off.")
        if mtf_cfg.get("1h_mode") not in (None, "off"):
            log.warning("STR MTF 'enabled' is false; forcing 1h_mode=off.")
        mode_15m = "off"
        mode_1h = "off"

    missing_15m_is_fatal = _resolve_missing_fatal(
        mtf_cfg,
        new_key="missing_15m_is_fatal",
        legacy_key="require_15m",
        log=log,
    )
    missing_1h_is_fatal = _resolve_missing_fatal(
        mtf_cfg,
        new_key="missing_1h_is_fatal",
        legacy_key="require_1h",
        log=log,
    )

    on_missing_15m = _normalize_missing_policy(mtf_cfg.get("on_missing_15m", "skip"), key="on_missing_15m")
    on_missing_1h = _normalize_missing_policy(mtf_cfg.get("on_missing_1h", "skip"), key="on_missing_1h")

    rsi_15m_min = _require_float(
        mtf_cfg.get("rsi_15m_min", DEFAULT_RSI_15M_MIN),
        key="rsi_15m_min",
        minimum=0.0,
        maximum=100.0,
        strict=strict,
    )
    rsi_1h_max = _require_float(
        mtf_cfg.get("rsi_1h_max", DEFAULT_RSI_1H_MAX),
        key="rsi_1h_max",
        minimum=0.0,
        maximum=100.0,
        strict=strict,
    )

    min_ext_pct = _require_float(
        mtf_cfg.get("min_15m_close_over_ema50_pct", DEFAULT_MIN_15M_CLOSE_OVER_EMA50_PCT),
        key="min_15m_close_over_ema50_pct",
        minimum=0.0,
        maximum=None,
        strict=strict,
    )
    require_ema_stack = _require_bool(
        mtf_cfg.get("require_1h_bearish_ema_stack", DEFAULT_REQUIRE_1H_BEARISH_EMA_STACK),
        key="require_1h_bearish_ema_stack",
        strict=strict,
    )

    min_bars = MtfMinBars(
        rsi=_require_int(mtf_cfg.get("min_bars_rsi", DEFAULT_MTF_MIN_BARS["rsi"]), key="min_bars_rsi", minimum=0, strict=strict),
        ema21=_require_int(mtf_cfg.get("min_bars_ema21", DEFAULT_MTF_MIN_BARS["ema21"]), key="min_bars_ema21", minimum=0, strict=strict),
        ema50=_require_int(mtf_cfg.get("min_bars_ema50", DEFAULT_MTF_MIN_BARS["ema50"]), key="min_bars_ema50", minimum=0, strict=strict),
        ema200=_require_int(mtf_cfg.get("min_bars_ema200", DEFAULT_MTF_MIN_BARS["ema200"]), key="min_bars_ema200", minimum=0, strict=strict),
    )

    policy_15m = MtfTimeframePolicy(
        mode=mode_15m,
        on_missing=on_missing_15m,
        missing_is_fatal=missing_15m_is_fatal,
        missing_policy="reject" if missing_15m_is_fatal else on_missing_15m,
        rsi_min=rsi_15m_min,
        min_close_over_ema50_pct=min_ext_pct,
    )

    policy_1h = MtfTimeframePolicy(
        mode=mode_1h,
        on_missing=on_missing_1h,
        missing_is_fatal=missing_1h_is_fatal,
        missing_policy="reject" if missing_1h_is_fatal else on_missing_1h,
        rsi_max=rsi_1h_max,
        require_bearish_ema_stack=require_ema_stack,
    )

    summary = (
        f"15m_mode={policy_15m.mode}, 1h_mode={policy_1h.mode}; "
        "missing_* affects missing-data only; thresholds veto only in hard mode"
    )

    return MtfConfirmationConfig(
        enabled=enabled,
        tf_15m=policy_15m,
        tf_1h=policy_1h,
        min_bars=min_bars,
        summary=summary,
    )


def _normalize_mode(value: Any, *, default: str, key: str) -> str:
    if value is None:
        mode = default
    elif not isinstance(value, str):
        raise ValueError(f"STR MTF '{key}' must be one of {VALID_MTF_MODES}.")
    else:
        mode = value.strip().lower()

    if mode not in VALID_MTF_MODES:
        raise ValueError(f"STR MTF '{key}' must be one of {VALID_MTF_MODES}.")
    return mode


def _normalize_missing_policy(value: Any, *, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"STR MTF '{key}' must be one of {VALID_MISSING_POLICIES}.")
    policy = value.strip().lower()
    if policy not in VALID_MISSING_POLICIES:
        raise ValueError(f"STR MTF '{key}' must be one of {VALID_MISSING_POLICIES}.")
    return policy


def _resolve_missing_fatal(
    mtf_cfg: Dict[str, Any],
    *,
    new_key: str,
    legacy_key: str,
    log: logging.Logger,
) -> bool:
    legacy_present = legacy_key in mtf_cfg
    new_present = new_key in mtf_cfg

    if new_present:
        new_value = _require_bool(mtf_cfg[new_key], key=new_key, strict=True)
        if legacy_present:
            legacy_value = _require_bool(mtf_cfg[legacy_key], key=legacy_key, strict=True)
            if legacy_value != new_value:
                log.warning(
                    "STR MTF legacy '%s' differs from '%s'; using '%s'=%s.",
                    legacy_key,
                    new_key,
                    new_key,
                    new_value,
                )
        return new_value

    if legacy_present:
        log.warning("Deprecated STR MTF key '%s' detected; use '%s'.", legacy_key, new_key)
        return _require_bool(mtf_cfg[legacy_key], key=legacy_key, strict=True)

    return False


def _require_bool(value: Any, *, key: str, strict: bool) -> bool:
    if value is None or not isinstance(value, bool):
        raise ValueError(f"STR MTF '{key}' must be a bool.")
    return value


def _require_int(value: Any, *, key: str, minimum: Optional[int], strict: bool) -> int:
    if value is None or not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"STR MTF '{key}' must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"STR MTF '{key}' must be >= {minimum}.")
    return value


def _require_float(
    value: Any,
    *,
    key: str,
    minimum: Optional[float],
    maximum: Optional[float],
    strict: bool,
) -> float:
    if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"STR MTF '{key}' must be a float.")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"STR MTF '{key}' must be >= {minimum}.")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"STR MTF '{key}' must be <= {maximum}.")
    return numeric
