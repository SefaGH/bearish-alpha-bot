#!/usr/bin/env python3
"""
diagnose_model_confidence.py
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, List
import importlib
from scripts.safe_torch_load import safe_torch_load # <-- YENİ İMPORT

import numpy as np

# Optional deps
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def load_samples(csv_path: str, limit: Optional[int] = None) -> np.ndarray:
    if pd is None:
        raise RuntimeError("pandas is required to read CSV files")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty")

    num_df = df.select_dtypes(include=["number"]).copy()
    for drop_col in ("label", "target", "y", "timestamp", "time", "datetime"):
        if drop_col in num_df.columns:
            num_df.drop(columns=[drop_col], inplace=True, errors="ignore")

    if num_df.empty:
        X = df.values
    else:
        X = num_df.values

    if limit is not None and limit > 0:
        X = X[:limit]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X.astype(np.float32, copy=False)


def try_load_model(model_path: Optional[str], model_class_import: Optional[str] = None) -> Dict[str, Any]:
    """
    Best-effort model loader.
    """
    if not model_path:
        return {"model": None, "note": "No model path provided"}
    if not os.path.exists(model_path):
        return {"model": None, "note": f"Model path does not exist: {model_path}"}

    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        if torch is None:
            return {"model": None, "note": "torch not available to load .pt/.pth"}
        try:
            # --- ÇÖZÜM: safe_torch_load kullanılıyor ---
            obj = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
            return {"model": obj, "note": "Loaded torch object via safe_torch_load"}
        except Exception as e:
            return {"model": None, "note": f"Failed safe_torch_load: {e}"}

    if ext in (".pkl", ".pickle"):
        try:
            if joblib is not None:
                obj = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, "rb") as f:
                    obj = pickle.load(f)
            return {"model": obj, "note": "Loaded via joblib/pickle"}
        except Exception as e:
            return {"model": None, "note": f"Failed pickle/joblib load: {e}"}

    if ext in (".yml", ".yaml"):
        if yaml is None:
            return {"model": None, "note": "PyYAML not available to parse YAML model config"}
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            keys_desc = list(y) if isinstance(y, dict) else str(type(y))
            return {"model": None, "note": f"Parsed YAML (not instantiating): keys={keys_desc}"}
        except Exception as e:
            return {"model": None, "note": f"Failed to parse YAML: {e}"}

    return {"model": None, "note": f"Unsupported model extension: {ext}"}


# --------------------------------------------------------------------------------------
# Core inference utility
# --------------------------------------------------------------------------------------
def _logits_to_probs(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, (torch.Tensor,)):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        with torch.no_grad():
            row_sums = x.sum(dim=1, keepdim=True)
            try:
                is_close_to_one = torch.allclose(row_sums, torch.ones_like(row_sums))
            except Exception:
                is_close_to_one = False 
            try_probs = bool(torch.all(x >= 0)) and is_close_to_one
            if try_probs:
                arr = x.detach().cpu().numpy()
                return arr
            if F is not None:
                return F.softmax(x, dim=1).cpu().numpy()
            t = x.detach().cpu()
            t = t - t.max(dim=1, keepdim=True)[0]
            e = torch.exp(t)
            p = e / e.sum(dim=1, keepdim=True)
            return p.cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.all(arr >= 0) and np.allclose(row_sums, 1.0, atol=1e-4):
        return arr
    arr = arr - arr.max(axis=1, keepdims=True)
    ex = np.exp(arr)
    return ex / ex.sum(axis=1, keepdims=True)


def compute_confidences_and_stats(model_obj: Any, X: np.ndarray, batch_size: int = 512) -> Dict[str, np.ndarray]:
    probs_list: List[np.ndarray] = []
    
    if isinstance(model_obj, dict):
        raise RuntimeError("Model object is a dictionary (checkpoint), not an instantiated model. Cannot run inference.")

    if hasattr(model_obj, "q_network") and getattr(model_obj, "q_network") is not None:
        if torch is None:
            raise RuntimeError("torch is required to use q_network")
        net = getattr(model_obj, "q_network")
        device = torch.device("cpu")
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
                out = net(xb)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs_chunk = _logits_to_probs(out)
                probs_list.append(probs_chunk)
    elif callable(model_obj):
        is_torch_module = False
        if torch is not None:
            try:
                is_torch_module = isinstance(model_obj, torch.nn.Module)
            except Exception:
                is_torch_module = False
        if is_torch_module:
            model_obj.eval()
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32))
                    out = model_obj(xb)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = _logits_to_probs(out)
                    probs_list.append(probs_chunk)
        else:
            if hasattr(model_obj, "predict_proba"):
                probs = np.asarray(model_obj.predict_proba(X))
                probs_list.append(probs)
            else:
                for i in range(0, len(X), batch_size):
                    Xb = X[i:i+batch_size]
                    try:
                        out = model_obj(Xb)
                    except Exception:
                        out = model_obj(Xb.astype(np.float32))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = _logits_to_probs(out)
                    probs_list.append(probs_chunk)
    elif hasattr(model_obj, "predict_proba"):
        probs = np.asarray(model_obj.predict_proba(X))
        probs_list.append(probs)
    else:
        raise RuntimeError("Model object is not callable, has no q_network, and has no predict_proba")

    if not probs_list:
        raise RuntimeError("No outputs produced by model during inference")
    probs = np.vstack(probs_list)
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T
    elif probs.shape[1] == 1:
        probs = np.hstack([1 - probs, probs])
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    return {"probs": probs, "confs": confs, "preds": preds}


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def build_report(X: np.ndarray,
                 result: Optional[Dict[str, np.ndarray]],
                 load_note: str,
                 error: Optional[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "samples_count": int(len(X)),
        "model_load_note": load_note,
    }
    if error is not None:
        report["inference_ok"] = False
        report["inference_error"] = str(error)
        return report
    probs = result["probs"]
    confs = result["confs"]
    preds = result["preds"]
    class_counts = {int(c): int((preds == c).sum()) for c in np.unique(preds)}
    avg_conf = float(confs.mean()) if len(confs) else 0.0
    report.update({
        "inference_ok": True,
        "n_classes": int(probs.shape[1]) if probs.ndim == 2 else 0,
        "avg_confidence": avg_conf,
        "class_distribution": class_counts,
        "probs_shape": list(probs.shape),
    })
    return report


def save_report(report: Dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Diagnose model confidences on a CSV of samples")
    p.add_argument("--csv", required=True, help="Path to samples CSV")
    p.add_argument("--model", default=None, help="Path to model file (optional)")
    p.add_argument("--limit", type=int, default=None, help="Optional row limit from CSV")
    p.add_argument("--out", default="report.json", help="Path to write JSON report")
    p.add_argument("--batch-size", type=int, default=512, help="Inference batch size")
    p.add_argument("--out-dir", default="diagnostics", help="Optional diagnostics output directory")
    
    p.add_argument("--model-class-import", default=None, help="Python import path for model class (e.g. src.agent.MyAgent)")
    
    return p.parse_args(argv)


def _write_diagnostics(out_dir: str, result: Dict[str, np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "probs_sample.npy"), result["probs"])
    confs = result["confs"]
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    hist = np.histogram(confs, bins=bins)[0].tolist()
    rel = {
        "bins": bins,
        "counts": hist,
        "avg_confidence": float(confs.mean()) if len(confs) else 0.0,
        "n": int(len(confs)),
    }
    with open(os.path.join(out_dir, "reliability.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    args = parse_args(argv)
    
    report_extras = {}
    inference_err = None

    # Load samples
    try:
        X = load_samples(args.csv, limit=args.limit)
    except Exception as e:
        err_report = {
            "samples_count": 0, "model_load_note": "skip (no model used)",
            "inference_ok": False, "inference_error": f"Failed to load samples: {e}",
        }
        save_report(err_report, args.out)
        print(f"[ERROR] {e}")
        return 2

    # Load model (optional)
    loaded = try_load_model(args.model, args.model_class_import)
    model_obj = loaded.get("model", None)
    load_note = loaded.get("note", "")

    # Run inference best-effort
    result = None
    if inference_err is None:
        if model_obj is None:
            inference_err = load_note 
        else:
            try:
                result = compute_confidences_and_stats(model_obj, X, batch_size=args.batch_size)
            except Exception as e:
                inference_err = str(e)

    report = build_report(X, result, load_note, inference_err)
    report.update(report_extras)
    save_report(report, args.out)

    # Always write diagnostics
    if args.out_dir:
        try:
            if report.get("inference_ok") and result is not None:
                _write_diagnostics(args.out_dir, result)
            else:
                os.makedirs(args.out_dir, exist_ok=True)
                np.save(os.path.join(args.out_dir, "X.npy"), X)
                with open(os.path.join(args.out_dir, "load_note.txt"), "w", encoding="utf-8") as f:
                    f.write(load_note or "")
        except Exception as e:
            print(f"[WARN] Failed to write diagnostics: {e}")

    # Console summary
    if report.get("inference_ok"):
        print(f"[OK] samples={report['samples_count']} | classes={report['n_classes']} | "
              f"avg_conf={report['avg_confidence']:.4f} | out={args.out}")
        if args.out_dir:
            print(f"[OK] diagnostics written to {args.out_dir}")
        return 0
    else:
        print(f"[WARN] Inference failed: {report.get('inference_error')} | out={args.out}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
