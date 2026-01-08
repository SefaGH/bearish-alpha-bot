"""
BingX VST balance management (Demo-only) via POST /openApi/swap/v2/trade/getVst.

This endpoint MUST be called against the VST host only:
  https://open-api-vst.bingx.com

Signing requirements (per BingX docs / field validation):
  - Include timestamp (ms) and optional recvWindow
  - Sort params ASCII ascending and sign "k=v&..." (NO URL-encoding before signing)
  - signature = HMAC-SHA256(secret, payload)

Rate limit: 5 rps per UID. We enforce a simple in-process throttle.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


logger = logging.getLogger(__name__)


VST_HOST = "open-api-vst.bingx.com"
VST_BASE_URL = "https://open-api-vst.bingx.com"
GET_VST_PATH = "/openApi/swap/v2/trade/getVst"


class BingxVstBalanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VstBalanceResult:
    balance: float
    raw: Dict[str, Any]


def build_bingx_signature_payload(params: Dict[str, Any]) -> str:
    """
    Build the BingX signature payload:
      - Sort keys ASCII ascending
      - Join as k=v&...
      - Do NOT URL-encode values
    """
    items = []
    for key in sorted(params.keys(), key=lambda x: str(x)):
        value = params.get(key)
        if value is None:
            continue
        items.append(f"{key}={value}")
    return "&".join(items)


def hmac_sha256_hex(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    while value.endswith("/"):
        value = value[:-1]
    return value


def require_vst_base_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        raise ValueError("Missing BingX base URL for VST getVst.")
    lowered = normalized.lower()
    if VST_HOST not in lowered:
        raise ValueError(
            "Refusing to call BingX getVst on non-VST host. "
            f"Expected host={VST_HOST!r} base_url={normalized!r}"
        )
    if "open-api.bingx.com" in lowered and VST_HOST not in lowered:
        raise ValueError(
            "Refusing to call BingX getVst on production host. "
            f"Expected base_url={VST_BASE_URL!r} got={normalized!r}"
        )
    return normalized


class _Throttle:
    def __init__(self, max_rps: float) -> None:
        self._min_interval_s = 1.0 / float(max_rps) if max_rps > 0 else 0.0
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def sleep_if_needed(self) -> None:
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval_s - (now - self._last_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_ts = time.monotonic()


class BingxVstBalanceClient:
    """
    Minimal client for VST getVst:
      - query balance (no adjustType/amount)
      - top up (adjustType=0, amount=N)
    """

    _throttle = _Throttle(max_rps=5.0)

    def __init__(
        self,
        *,
        api_key: str,
        secret_key: str,
        base_url: str,
        recv_window_ms: int = 5000,
        timeout_s: float = 10.0,
    ) -> None:
        if not api_key or not secret_key:
            raise ValueError("Missing BingX api_key/secret_key for VST getVst.")
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = require_vst_base_url(base_url)

        try:
            self._recv_window_ms = int(recv_window_ms)
        except (TypeError, ValueError):
            self._recv_window_ms = 5000
        self._recv_window_ms = max(1000, min(10000, self._recv_window_ms))

        try:
            self._timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            self._timeout_s = 10.0
        self._timeout_s = max(1.0, self._timeout_s)

    def _timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def _post(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle.sleep_if_needed()

        request_params: Dict[str, Any] = dict(params or {})
        request_params["timestamp"] = self._timestamp_ms()
        request_params["recvWindow"] = self._recv_window_ms

        payload = build_bingx_signature_payload(request_params)
        signature = hmac_sha256_hex(self._secret_key, payload)
        request_params["signature"] = signature

        url = f"{self._base_url}{GET_VST_PATH}"
        headers = {
            "X-BX-APIKEY": self._api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        resp = requests.post(url, data=request_params, headers=headers, timeout=self._timeout_s)
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()

        code = data.get("code")
        if code in (0, "0", None):
            return data

        msg = str(data.get("msg") or "")
        if str(code) == "109500":
            raise BingxVstBalanceError(
                "BingX getVst returned code=109500 (commonly wrong domain or invalid params). "
                f"base_url={self._base_url!r} msg={msg[:200]!r}"
            )
        raise BingxVstBalanceError(f"BingX getVst failed code={code!r} msg={msg[:200]!r}")

    @staticmethod
    def extract_balance_value(response: Dict[str, Any]) -> float:
        """
        Best-effort extraction of the VST balance value from the API response.
        """
        if not isinstance(response, dict):
            raise BingxVstBalanceError("Invalid getVst response type (expected dict).")

        data = response.get("data")
        candidates = ("balance", "availableBalance", "available", "amount", "vstBalance", "vst")

        def _coerce(value: Any) -> Optional[float]:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if isinstance(data, dict):
            for key in candidates:
                if key in data:
                    parsed = _coerce(data.get(key))
                    if parsed is not None:
                        return parsed

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                for key in candidates:
                    if key in item:
                        parsed = _coerce(item.get(key))
                        if parsed is not None:
                            return parsed

        parsed = _coerce(data)
        if parsed is not None:
            return parsed

        raise BingxVstBalanceError("Unable to extract VST balance from getVst response.")

    def get_vst_balance(self) -> VstBalanceResult:
        response = self._post({})
        balance = self.extract_balance_value(response)
        return VstBalanceResult(balance=balance, raw=response)

    def apply_vst_topup(self, amount: float) -> Dict[str, Any]:
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            raise ValueError("Invalid top-up amount for getVst (expected number).") from None

        if amt <= 0:
            raise ValueError("Top-up amount must be > 0.")

        # BingX expects adjustType=0 for top-up, amount as numeric.
        # Keep amount formatting stable (no scientific notation).
        amount_str = str(int(amt)) if float(int(amt)) == amt else f"{amt:.8f}".rstrip("0").rstrip(".")
        response = self._post({"adjustType": 0, "amount": amount_str})
        return response


def create_vst_balance_client_from_ccxt_bingx_client(
    ccxt_client: Any,
    *,
    recv_window_ms: int,
    timeout_s: float = 10.0,
) -> BingxVstBalanceClient:
    """
    Convenience factory to build a VST balance client from our CcxtClient('bingx') wrapper.
    """
    base_url = getattr(ccxt_client, "_bingx_rest_base_url", None) or ""
    auth = getattr(ccxt_client, "bingx_auth", None)
    api_key = getattr(auth, "api_key", None) or ""
    secret_key = getattr(auth, "secret_key", None) or ""
    return BingxVstBalanceClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        recv_window_ms=recv_window_ms,
        timeout_s=timeout_s,
    )

