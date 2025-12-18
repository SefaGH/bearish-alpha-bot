"""
One-time GEMMA health snapshot.

Collects quick metrics from the TorchScript adapter (float-first by default)
and optionally samples recent prediction logs to flag simple alerts:
- Class dominance (>90% of parsed predictions)
- High-confidence saturation (>80% with conf>0.95)
- Latency p95 from adapter (if available)

Usage:
    python scripts/gemma_health_snapshot.py --log-file logs/live_trading_latest.log
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Ensure project root is on sys.path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.ml.adapters.gemma.gemma_torchscript_adapter import GemmaTorchScriptAdapter
    from src.ml.manifest_manager import ManifestManager
except ImportError as exc:
    raise SystemExit(f"Failed to import GEMMA modules: {exc}")


def build_adapter(prefer_quantized: bool = False, force_quantized: bool = False) -> GemmaTorchScriptAdapter:
    """Instantiate the adapter with float-first defaults."""
    manifest_mgr = ManifestManager()
    manifest = manifest_mgr.load_manifest("artifacts/gemma/final")

    model_path = Path(manifest_mgr.get_model_path("gemma_price"))
    scaler_path = Path("artifacts/gemma/final") / manifest.get("price_scaler_path", "gemma_price_scaler.joblib")
    features_path = Path(manifest.get("active_features_path", "features/gemma/selected/gemma_price_selected_82.json"))

    adapter_cfg = {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "features_path": str(features_path),
        "feature_count": manifest.get("feature_count", 82),
        "feature_names": manifest.get("feature_names_ordered", []),
        "prefer_quantized": prefer_quantized,
        "force_quantized": force_quantized,
        "metrics_log_interval": 10,  # faster logging when run standalone
    }
    return GemmaTorchScriptAdapter(adapter_cfg)


def run_probe_predictions(adapter: GemmaTorchScriptAdapter) -> Dict[str, float]:
    """Run a few synthetic predictions to populate metrics."""
    features = adapter.feature_names or adapter.features or list(range(adapter.expected_feature_count))
    zero_features = {f: 0.0 for f in features}
    one_features = {f: 1.0 for f in features}
    rand_features = {f: float(np.random.randn()) for f in features}

    for name, feats in (("zeros", zero_features), ("ones", one_features), ("random", rand_features)):
        result = adapter.predict(feats)
        print(f"[probe] {name}: label={result.get('prediction_label')} conf={result.get('price_confidence'):.4f}")

    metrics = adapter.get_metrics()
    return {
        "conf_p50": metrics.get("recent_conf_p50", 0.0),
        "conf_p95": metrics.get("recent_conf_p95", 0.0),
        "p95_latency_ms": metrics.get("p95_inference_time_ms", 0.0),
        "class_counts": metrics.get("class_counts", {}),
    }


def parse_logs(log_path: Path) -> Tuple[Dict[str, int], List[float]]:
    """Parse prediction/confidence pairs from logs (supports multiple formats)."""
    class_counts: Dict[str, int] = {}
    confidences: List[float] = []
    patterns = [
        re.compile(r"Prediction:\s*(\w+).*?confidence[:=]\s*([0-9.]+)", re.IGNORECASE),
        re.compile(r"predicted_regime[:=]\s*['\"]?(\w+)['\"]?.*?confidence[:=]\s*([0-9.]+)", re.IGNORECASE),
        re.compile(r"ml_regime[\"']?\s*[:=]\s*['\"]?(\w+)['\"]?.*?regime_conf(?:idence)?[\"']?\s*[:=]\s*([0-9.]+)", re.IGNORECASE),
    ]
    if not log_path.exists():
        return class_counts, confidences
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            match = None
            for pat in patterns:
                match = pat.search(line)
                if match:
                    break
            if not match:
                continue
            label = match.group(1).lower()
            try:
                conf = float(match.group(2))
            except ValueError:
                continue
            class_counts[label] = class_counts.get(label, 0) + 1
            confidences.append(conf)
    return class_counts, confidences


def summarize_feature_vector(vec: np.ndarray, scaler: Optional[object] = None) -> Dict[str, float]:
    """Compute sparsity and z-score summary for a feature vector."""
    summary: Dict[str, float] = {}
    if vec.size == 0:
        return {"error": "empty_vector"}
    summary["length"] = int(len(vec))
    summary["zero_ratio"] = float(np.sum(vec == 0) / len(vec))
    summary["nan_ratio"] = float(np.sum(~np.isfinite(vec)) / len(vec))
    summary["min"] = float(np.nanmin(vec))
    summary["max"] = float(np.nanmax(vec))
    if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        z = (vec - scaler.mean_) / (scaler.scale_ + 1e-12)
        z = z[np.isfinite(z)]
        if z.size:
            summary["z_abs_p50"] = float(np.percentile(np.abs(z), 50))
            summary["z_abs_p95"] = float(np.percentile(np.abs(z), 95))
            summary["z_abs_max"] = float(np.max(np.abs(z)))
        else:
            summary["z_error"] = "non-finite z"
    return summary


def derive_alerts(class_counts: Dict[str, int], confidences: List[float]) -> List[str]:
    alerts: List[str] = []
    total = sum(class_counts.values())
    if total:
        dominant_label, dominant_cnt = max(class_counts.items(), key=lambda kv: kv[1])
        if dominant_cnt / total > 0.9:
            alerts.append(f"class dominance: {dominant_label} {dominant_cnt}/{total} (>90%)")
    if confidences:
        high_conf = sum(1 for c in confidences if c > 0.95)
        if high_conf / len(confidences) > 0.8:
            alerts.append(f"high confidence saturation: {high_conf}/{len(confidences)} (>80% >0.95)")
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="GEMMA health snapshot")
    parser.add_argument("--log-file", type=Path, default=Path("logs/live_trading_latest.log"), help="Prediction log to sample")
    parser.add_argument("--prefer-quantized", action="store_true", help="Prefer quantized model")
    parser.add_argument("--force-quantized", action="store_true", help="Force quantized model")
    parser.add_argument("--sample-live", action="store_true", help="Sample a live feature vector via price predictor (requires config + data access)")
    parser.add_argument("--live-features", action="store_true", help="Summarize one live feature vector (sparsity/z-scores) using feature pipeline and adapter scaler")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to fetch live features for (requires market data access)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe to fetch OHLCV for live features")
    args = parser.parse_args()

    adapter = build_adapter(prefer_quantized=args.prefer_quantized, force_quantized=args.force_quantized)
    probe_metrics = run_probe_predictions(adapter)

    class_counts, confidences = parse_logs(args.log_file)
    alerts = derive_alerts(class_counts, confidences)

    summary = {
        "probe_metrics": probe_metrics,
        "log_sample": {
            "class_counts": class_counts,
            "n_confidences": len(confidences),
            "conf_p50": statistics.median(confidences) if confidences else 0.0,
            "conf_p95": float(np.percentile(confidences, 95)) if len(confidences) > 1 else (confidences[0] if confidences else 0.0),
        },
        "alerts": alerts,
    }

    if args.sample_live:
        try:
            from config.live_trading_config import LiveTradingConfiguration
            from src.ml.feature_engineering import FeatureEngineeringPipeline
            from src.ml.price_predictor import AdvancedPricePredictionEngine
        except Exception as exc:
            summary["live_sample_error"] = f"Failed to import live deps: {exc}"
        else:
            try:
                cfg = LiveTradingConfiguration.load(log_summary=False)
                ml_cfg = cfg.get("ml", {})
                feat_pipe = FeatureEngineeringPipeline(cfg)
                price_engine = AdvancedPricePredictionEngine(None, feat_pipe, ml_cfg.get("price_prediction", {}))
                symbol = ml_cfg.get("symbols", ["BTC/USDT:USDT"])[0]
                # Attempt a single prediction (non-async helper)
                import asyncio

                loop = asyncio.get_event_loop()
                pred = loop.run_until_complete(price_engine._make_prediction_for_symbol(symbol))
                summary["live_prediction"] = {
                    "symbol": symbol,
                    "prediction": pred.get("adapter", {}).get("prediction_label") if pred else None,
                    "confidence": pred.get("adapter", {}).get("price_confidence") if pred else None,
                    "probabilities": pred.get("adapter", {}).get("probabilities") if pred else None,
                    "mode": pred.get("mode") if pred else None,
                }
            except Exception as exc:
                summary["live_sample_error"] = f"Live sample failed: {exc}"

    if args.live_features and "live_sample_error" not in summary:
        try:
            from config.live_trading_config import LiveTradingConfiguration
            from src.ml.feature_engineering import FeatureEngineeringPipeline
            import ccxt
            import pandas as pd
        except Exception as exc:
            summary["live_features_error"] = f"Failed to import live deps: {exc}"
        else:
            try:
                cfg = LiveTradingConfiguration.load(log_summary=False)
                feat_pipe = FeatureEngineeringPipeline(cfg)
                symbol = args.symbol or (cfg.get("ml", {}).get("symbols", ["BTC/USDT:USDT"]) or ["BTC/USDT:USDT"])[0]

                live_vec = None
                # Try cached features first if available
                try:
                    if hasattr(feat_pipe, "get_cached_features"):
                        fdf = feat_pipe.get_cached_features(symbol, mode="price")
                        if fdf is not None and not fdf.empty:
                            live_vec = fdf.tail(1).values[0]
                            summary["live_features_source"] = "cached"
                except Exception:
                    live_vec = None

                # Fallback: fetch fresh OHLCV via ccxt and extract features
                if live_vec is None:
                    try:
                        ex = ccxt.bingx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
                        data = ex.fetch_ohlcv(symbol.split(':')[0], timeframe=args.timeframe, limit=500)
                        if data:
                            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            feats = feat_pipe.extract_features(df, mode='price')
                            if not feats.empty:
                                live_vec = feats.tail(1).values[0]
                                summary["live_features_source"] = "ccxt_fetch"
                    except Exception as exc_fetch:
                        summary["live_features_fetch_error"] = str(exc_fetch)

                if live_vec is None:
                    live_vec = np.zeros(adapter.expected_feature_count, dtype=float)
                    summary["live_features_note"] = "Used zero vector fallback; no live features accessible"

                summary["live_features"] = summarize_feature_vector(live_vec, adapter.scaler)
            except Exception as exc:
                summary["live_features_error"] = f"Live features summary failed: {exc}"

    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
