"""
Forensic probe utilities.

Goal: compare BingX VST vs PROD market-data streams (kline + ticker) to detect
whether VST API data is mirrored from production or isolated.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import websocket

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _decode_ws_message(message: Any) -> str:
    if isinstance(message, bytes):
        try:
            return gzip.decompress(message).decode("utf-8")
        except Exception:
            return message.decode("utf-8", errors="replace")
    if isinstance(message, str):
        return message
    return str(message)


@dataclass
class ProbeStats:
    connected: bool = False
    messages: int = 0
    subscription_confirms: int = 0
    parse_errors: int = 0
    last_error: Optional[str] = None
    last_data_ts_ms: Optional[int] = None
    last_recv_utc: Optional[str] = None


@dataclass
class ProbeState:
    label: str
    url: str
    symbol: str = "BTC-USDT"
    timeframe: str = "1m"
    duration_s: int = 60
    output_jsonl: Optional[Path] = None

    stats: ProbeStats = field(default_factory=ProbeStats)
    last_ticker: Dict[str, Optional[float]] = field(default_factory=dict)
    last_kline: Dict[str, Optional[float]] = field(default_factory=dict)
    kline_by_T: Dict[int, Dict[str, Optional[float]]] = field(default_factory=dict)
    ticker_samples: List[Dict[str, Any]] = field(default_factory=list)
    kline_samples: List[Dict[str, Any]] = field(default_factory=list)


class BingXWsProbeClient:
    def __init__(self, state: ProbeState):
        self.state = state
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._connected_event = threading.Event()
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._output_fp = None

    @property
    def connected(self) -> bool:
        return self._connected_event.is_set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        if self.state.output_jsonl:
            self.state.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._output_fp = self.state.output_jsonl.open("a", encoding="utf-8")

        self._ws = websocket.WebSocketApp(
            self.state.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        try:
            if self._ws:
                self._ws.close()
        finally:
            if self._thread:
                self._thread.join(timeout=timeout_s)
            if self._output_fp:
                self._output_fp.close()
                self._output_fp = None

    def wait_connected(self, timeout_s: float = 10.0) -> bool:
        return self._connected_event.wait(timeout=timeout_s)

    def _write_jsonl(self, payload: Dict[str, Any]) -> None:
        if not self._output_fp:
            return
        with self._write_lock:
            self._output_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._output_fp.flush()

    def _on_open(self, ws) -> None:
        self.state.stats.connected = True
        self._connected_event.set()
        logger.info("[probe:%s] connected url=%s", self.state.label, self.state.url)

        kline_type = f"{self.state.symbol}@kline_{self.state.timeframe}"
        ticker_type = f"{self.state.symbol}@ticker"
        for data_type in (kline_type, ticker_type):
            sub_msg = {"id": f"{self.state.label}:{data_type}", "reqType": "sub", "dataType": data_type}
            try:
                ws.send(json.dumps(sub_msg))
            except Exception as e:
                logger.warning("[probe:%s] failed to subscribe dataType=%s err=%s", self.state.label, data_type, e)

    def _on_message(self, ws, message) -> None:
        text = _decode_ws_message(message)
        if text == "Ping":
            try:
                ws.send("Pong")
            except Exception:
                pass
            return

        self.state.stats.messages += 1
        self.state.stats.last_recv_utc = _utc_now_iso()

        try:
            data = json.loads(text)
        except Exception:
            self.state.stats.parse_errors += 1
            return

        if "id" in data and "code" in data:
            self.state.stats.subscription_confirms += 1
            self._write_jsonl(
                {
                    "utc": _utc_now_iso(),
                    "label": self.state.label,
                    "url": self.state.url,
                    "type": "sub_confirm",
                    "payload": data,
                }
            )
            return

        data_type = data.get("dataType") or ""
        payload = data.get("data")
        self.state.stats.last_data_ts_ms = data.get("ts") or data.get("T") or data.get("E")

        if "@ticker" in data_type and isinstance(payload, dict):
            v = _safe_float(payload.get("v"))
            q = _safe_float(payload.get("q"))
            self.state.last_ticker = {"v": v, "q": q, "E": payload.get("E")}
            sample = {"utc": _utc_now_iso(), "label": self.state.label, "dataType": data_type, "v": v, "q": q, "E": payload.get("E")}
            self.state.ticker_samples.append(sample)
            self._write_jsonl({"type": "ticker", "url": self.state.url, **sample})
            return

        if "@kline" in data_type and isinstance(payload, list) and payload and isinstance(payload[0], dict):
            k = payload[0]
            T = k.get("T")
            v = _safe_float(k.get("v"))
            q = _safe_float(k.get("q"))  # usually absent on BingX, but keep for hypothesis testing
            self.state.last_kline = {"T": T, "v": v, "q": q}
            if isinstance(T, int):
                self.state.kline_by_T[T] = {"v": v, "q": q}
            sample = {"utc": _utc_now_iso(), "label": self.state.label, "dataType": data_type, "T": T, "v": v, "q": q}
            self.state.kline_samples.append(sample)
            self._write_jsonl({"type": "kline", "url": self.state.url, **sample})
            return

    def _on_error(self, ws, error) -> None:
        self.state.stats.last_error = str(error)
        logger.error("[probe:%s] websocket error url=%s err=%s", self.state.label, self.state.url, error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self.state.stats.connected = False
        logger.warning(
            "[probe:%s] closed url=%s code=%s msg=%s",
            self.state.label,
            self.state.url,
            close_status_code,
            close_msg,
        )


def _compare_latest_common_kline(vst: ProbeState, prod: ProbeState) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    common = set(vst.kline_by_T.keys()) & set(prod.kline_by_T.keys())
    if not common:
        return None, None, None
    latest_T = max(common)
    v_vst = vst.kline_by_T.get(latest_T, {}).get("v")
    v_prod = prod.kline_by_T.get(latest_T, {}).get("v")
    ratio = (v_vst / v_prod) if (v_vst is not None and v_prod) else None
    return latest_T, v_vst, ratio


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0.0):
        return None
    return a / b


def _kline_ratios_by_T(vst: ProbeState, prod: ProbeState) -> List[Dict[str, Any]]:
    common = sorted(set(vst.kline_by_T.keys()) & set(prod.kline_by_T.keys()))
    out: List[Dict[str, Any]] = []
    for T in common:
        v_vst = vst.kline_by_T.get(T, {}).get("v")
        v_prod = prod.kline_by_T.get(T, {}).get("v")
        out.append({"T": T, "vst_v": v_vst, "prod_v": v_prod, "ratio": _ratio(v_vst, v_prod)})
    return out


def _series_stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(min(numeric)), "max": float(max(numeric)), "mean": float(sum(numeric) / len(numeric))}


def run_dual_ws_probe(
    *,
    symbol: str = "BTC-USDT",
    timeframe: str = "1m",
    duration_s: int = 60,
    vst_url: str = "wss://open-api-vst.bingx.com/swap-market",
    prod_url: str = "wss://open-api-swap.bingx.com/swap-market",
    output_dir: str = "reports",
) -> Dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    base = f"forensic_probe_ws_compare_{symbol.replace('-', '')}_{ts}"
    out_vst = out_dir / f"{base}_VST.jsonl"
    out_prod = out_dir / f"{base}_PROD.jsonl"

    vst_candidates = [
        vst_url,
        "wss://vst-open-api-ws.bingx.com/swap-market",
    ]
    prod_candidates = [prod_url]

    def connect_first(label: str, urls: List[str], out_path: Path) -> BingXWsProbeClient:
        last_err = None
        for url in urls:
            state = ProbeState(label=label, url=url, symbol=symbol, timeframe=timeframe, duration_s=duration_s, output_jsonl=out_path)
            client = BingXWsProbeClient(state)
            client.start()
            if client.wait_connected(timeout_s=10.0):
                return client
            last_err = state.stats.last_error
            client.stop()
        raise RuntimeError(f"Failed to connect {label} WS. last_error={last_err} urls={urls}")

    vst = connect_first("VST", vst_candidates, out_vst)
    prod = connect_first("PROD", prod_candidates, out_prod)

    logger.info(
        "[probe] collecting duration_s=%s symbol=%s timeframe=%s out_vst=%s out_prod=%s",
        duration_s,
        symbol,
        timeframe,
        out_vst,
        out_prod,
    )
    time.sleep(duration_s)

    vst.stop()
    prod.stop()

    vst_state = vst.state
    prod_state = prod.state

    ticker_v_ratio = _ratio(vst_state.last_ticker.get("v"), prod_state.last_ticker.get("v"))
    ticker_q_ratio = _ratio(vst_state.last_ticker.get("q"), prod_state.last_ticker.get("q"))
    latest_T, kline_v_vst, kline_v_ratio = _compare_latest_common_kline(vst_state, prod_state)
    kline_ratios = _kline_ratios_by_T(vst_state, prod_state)
    vst_ticker_v_series = [s.get("v") for s in vst_state.ticker_samples]
    vst_ticker_q_series = [s.get("q") for s in vst_state.ticker_samples]
    prod_ticker_v_series = [s.get("v") for s in prod_state.ticker_samples]
    prod_ticker_q_series = [s.get("q") for s in prod_state.ticker_samples]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "duration_s": duration_s,
        "vst_url": vst_state.url,
        "prod_url": prod_state.url,
        "output_jsonl_vst": str(out_vst),
        "output_jsonl_prod": str(out_prod),
        "vst": {
            "stats": vst_state.stats.__dict__,
            "last_ticker": vst_state.last_ticker,
            "last_kline": vst_state.last_kline,
        },
        "prod": {
            "stats": prod_state.stats.__dict__,
            "last_ticker": prod_state.last_ticker,
            "last_kline": prod_state.last_kline,
        },
        "compare": {
            "ticker_v_ratio": ticker_v_ratio,
            "ticker_q_ratio": ticker_q_ratio,
            "ticker_v_stats_vst": _series_stats(vst_ticker_v_series),
            "ticker_q_stats_vst": _series_stats(vst_ticker_q_series),
            "ticker_v_stats_prod": _series_stats(prod_ticker_v_series),
            "ticker_q_stats_prod": _series_stats(prod_ticker_q_series),
            "kline_latest_common_T": latest_T,
            "kline_vst_v_at_T": kline_v_vst,
            "kline_v_ratio_at_T": kline_v_ratio,
            "kline_ratios_by_T": kline_ratios,
        },
    }
