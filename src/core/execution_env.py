import os
from typing import Literal


TradingMode = Literal["paper", "live"]
ExecutionBackend = Literal["simulated", "ccxt"]
BingxEnv = Literal["prod", "vst"]
VstFullbotCanarySide = Literal["long", "short"]
ProdCanary0Side = Literal["long", "short"]

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _get_env_lower(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return (value or default).strip().lower()


def get_trading_mode() -> TradingMode:
    mode = _get_env_lower("TRADING_MODE", "paper")
    return "live" if mode == "live" else "paper"


def get_execution_backend() -> ExecutionBackend:
    backend = _get_env_lower("EXECUTION_BACKEND", "simulated")
    return "ccxt" if backend == "ccxt" else "simulated"


def get_bingx_env() -> BingxEnv:
    env = _get_env_lower("BINGX_ENV", "prod")
    return "vst" if env == "vst" else "prod"


def is_real_execution_enabled() -> bool:
    return get_trading_mode() == "live" and get_execution_backend() == "ccxt"


def require_explicit_bingx_env_if_real_execution() -> None:
    if not is_real_execution_enabled():
        return

    raw = os.getenv("BINGX_ENV")
    if raw is None or raw.strip().lower() not in {"prod", "vst"}:
        raise RuntimeError("Refusing real execution: set BINGX_ENV=prod|vst explicitly.")


def is_vst_fullbot_canary_enabled() -> bool:
    return env_flag("VST_FULLBOT_CANARY", False)


def is_vst_fullbot_canary_cleanup_enabled() -> bool:
    return env_flag("VST_FULLBOT_CANARY_ALLOW_CLEANUP", False)


def is_vst_fullbot_canary_force_market() -> bool:
    if not is_vst_fullbot_canary_enabled():
        return False
    raw = os.getenv("VST_FULLBOT_CANARY_FORCE_MARKET")
    if raw is None:
        return True
    return raw.strip().lower() in _TRUE_VALUES


def get_vst_fullbot_canary_max_closed_trades() -> int:
    value = env_int("VST_FULLBOT_CANARY_MAX_CLOSED_TRADES", 1)
    return max(1, value)


def get_vst_fullbot_canary_evidence_dir() -> str:
    return env_str("VST_FULLBOT_CANARY_EVIDENCE_DIR", "diagnostics/vst")


def get_vst_fullbot_canary_side() -> VstFullbotCanarySide:
    raw = _get_env_lower("VST_FULLBOT_CANARY_SIDE", "long")
    if raw in {"short", "sell"}:
        return "short"
    return "long"


def is_prod_canary_0_enabled() -> bool:
    return env_flag("PROD_CANARY_0", False)


def is_prod_canary_0_cleanup_enabled() -> bool:
    return env_flag("PROD_CANARY_0_ALLOW_CLEANUP", False)


def get_prod_canary_0_max_closed_trades() -> int:
    value = env_int("PROD_CANARY_0_MAX_CLOSED_TRADES", 1)
    return max(1, value)


def get_prod_canary_0_evidence_dir() -> str:
    return env_str("PROD_CANARY_0_EVIDENCE_DIR", "diagnostics/prod_canary")


def get_prod_canary_0_side() -> ProdCanary0Side:
    raw = _get_env_lower("PROD_CANARY_0_SIDE", "long")
    if raw in {"short", "sell"}:
        return "short"
    return "long"


def _format_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def format_mode_banner(exchange_clients: object | None = None) -> str:
    """
    Return a single-line startup banner summarizing runtime mode + routing.

    Intentionally excludes any credential-like values.
    """
    trading_mode = get_trading_mode()
    execution_backend = get_execution_backend()
    bingx_env = get_bingx_env()

    native_hard_stop_enabled = env_flag("BINGX_NATIVE_HARD_STOP_ENABLED", False)
    native_trailing_enabled = env_flag("BINGX_NATIVE_TRAILING_ON_ACTIVATION_ENABLED", False)

    ccxt_sandbox: object | None = None
    rest_base_url: object | None = None

    if isinstance(exchange_clients, dict):
        bingx_client = exchange_clients.get("bingx") or exchange_clients.get("BINGX")
        if bingx_client is not None:
            ex = getattr(bingx_client, "ex", None) or getattr(bingx_client, "exchange", None)
            ccxt_sandbox = bool(getattr(ex, "sandbox", False) or getattr(bingx_client, "bingx_env", None) == "vst")
            rest_base_url = getattr(bingx_client, "_bingx_rest_base_url", None)

    # Fallback to intended routing if no client is available yet.
    if rest_base_url in (None, ""):
        rest_base_url = "https://open-api-vst.bingx.com" if bingx_env == "vst" else "https://open-api.bingx.com"
    if ccxt_sandbox is None:
        ccxt_sandbox = (bingx_env == "vst")

    return (
        "[MODE-BANNER] "
        f"TRADING_MODE={trading_mode} EXECUTION_BACKEND={execution_backend} BINGX_ENV={bingx_env} | "
        f"CCXT_SANDBOX={_format_bool(bool(ccxt_sandbox))} REST_BASE_URL={rest_base_url} | "
        f"NATIVE_HARD_STOP={_format_bool(native_hard_stop_enabled)} "
        f"NATIVE_TRAILING={_format_bool(native_trailing_enabled)}"
    )
