#!/usr/bin/env python3
"""
Diagnosis script for low-confidence predictions.

Usage examples (Python 3.11 required):
  python scripts/diagnose_model_confidence.py --model data/models/rl_agent_final.pth --samples sample_data/test_samples.csv --scaler data/models/scaler.pkl --state-import env.state_vector

Notes:
- The script tries multiple ways to load the model. If your repo provides a model class,
  set --model-class-import to the import path (e.g. models.rl_agent.RLAgent) so the script can
  recreate and load state_dict.
- Outputs: diagnostics/report.json and plots in diagnostics/
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import brier_score_loss
import math
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def _strip_quotes(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return str(s).strip().strip('"').strip("'")

def try_load_model(model_path: str, model_class_import: Optional[str]=None):
    """
    Robust model loader:
    - accepts scripted modules, saved nn.Modules, or state-dict checkpoints
    - if model_class_import points to a class that needs constructor args (e.g. TradingRLAgent),
      this function will try to infer sensible defaults from config/config.example.yaml
      and call the class's load_model(path) method if present.
    """
    # sanitize inputs (strip surrounding quotes)
    def _strip(s):
        return None if s is None else str(s).strip().strip('"').strip("'")
    model_path = _strip(model_path)
    model_class_import = _strip(model_class_import)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 1) Try scripted module first
    try:
        m = torch.jit.load(str(model_path), map_location="cpu")
        return {"type": "scripted", "model": m}
    except Exception:
        pass

    # 2) Try torch.load
    try:
        obj = torch.load(str(model_path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"torch.load failed: {e}")

    # 3) If obj is dict and model_class_import provided, try to construct class and load
    if isinstance(obj, dict) and model_class_import:
        # import class
        try:
            module_path, class_name = model_class_import.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            Klass = getattr(mod, class_name)
        except Exception as e:
            return {"type": "dict", "obj": obj, "note": f"failed to import {model_class_import}: {e}"}

        # attempt to instantiate intelligently
        try:
            # Try simple no-arg constructor first
            model = Klass()
        except TypeError as ctor_err:
            # Inspect signature to see required params
            import inspect, yaml
            sig = inspect.signature(Klass.__init__)
            params = list(sig.parameters.keys())[1:]  # drop 'self'
            # Try to read config/config.example.yaml or config/config.yaml for defaults
            cfg = {}
            for cfg_path in ("config/config.example.yaml", "config/config.yaml"):
                try:
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                        break
                except Exception:
                    cfg = {}
            ml_cfg = cfg.get("ml", {}) if isinstance(cfg, dict) else {}
            rl_cfg = ml_cfg.get("reinforcement_learning", {}) if isinstance(ml_cfg, dict) else {}
            # state_size guess: ml.features.feature_size or ml.feature_size or default 42
            state_size = None
            if isinstance(ml_cfg, dict):
                state_size = ml_cfg.get("features", {}).get("feature_size") or ml_cfg.get("feature_size")
            if state_size is None:
                state_size = 42
            # action_size guess: from rl_cfg or default 3
            action_size = rl_cfg.get("action_size") if isinstance(rl_cfg, dict) else None
            if action_size is None:
                action_size = 3

            # Build kwargs based on parameter names
            ctor_kwargs = {}
            if "state_size" in params:
                ctor_kwargs["state_size"] = int(state_size)
            if "action_size" in params:
                ctor_kwargs["action_size"] = int(action_size)
            if "config" in params:
                ctor_kwargs["config"] = rl_cfg if isinstance(rl_cfg, dict) else {}

            try:
                model = Klass(**ctor_kwargs)
            except Exception as e:
                return {"type": "dict", "obj": obj, "note": f"failed to construct {model_class_import} with inferred args: {e}"}

        # If model has load_model(path) prefer that (TradingRLAgent implements load_model)
        try:
            if hasattr(model, "load_model"):
                try:
                    model.load_model(str(model_path))
                    return {"type": "nn_module_loaded_via_class", "model": model}
                except Exception as e:
                    # proceed to try load_state_dict fallback
                    pass
            # Otherwise attempt load_state_dict if available
            if hasattr(model, "load_state_dict"):
                state = obj if "state_dict" not in obj else obj["state_dict"]
                model.load_state_dict(state)
                model.eval()
                return {"type": "nn_module", "model": model}
        except Exception as e:
            return {"type": "dict", "obj": obj, "note": f"constructed {model_class_import} but loading weights failed: {e}"}

        # fallback: return dict
        return {"type": "dict", "obj": obj, "note": "could not load into class; returned dict"}

    # 4) If obj is nn.Module saved directly
    if hasattr(obj, "eval"):
        obj.eval()
        return {"type": "nn_module_saved", "model": obj}

    # 5) Otherwise return dict
    return {"type": "dict", "obj": obj}

def load_scaler(scaler_path: Optional[str]):
    if not scaler_path:
        return None
    p = Path(scaler_path)
    if not p.exists():
        print(f"Scaler not found: {p}")
        return None
    try:
        return joblib.load(str(p))
    except Exception as e:
        print(f"joblib.load failed for scaler: {e}")
        # try pickle
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)

def compute_confidences_and_stats(model_obj, X: np.ndarray, batch_size=512):
    """
    Returns dict with confidences, predicted classes, probs.

    Behavior (deterministic):
    - If model_obj has attribute `q_network` and it's not None, use that network to compute logits -> softmax -> probs.
    - Else if model_obj is a PyTorch callable (nn.Module or scripted), call it with tensors -> softmax -> probs.
    - Else if model_obj has `predict_proba`, call that.
    - Otherwise raise a clear RuntimeError.

    Always returns:
      {"probs": probs, "confs": confs, "preds": preds}
    where probs is an (N, C) numpy array, confs is (N,) max-prob per sample, preds is (N,) argmax class indices.
    """
    import torch
    import numpy as _np
    try:
        import torch.nn.functional as F
    except Exception:
        F = None

    if X is None or len(X) == 0:
        raise RuntimeError("Empty input X given to compute_confidences_and_stats")

    probs_list = []

    # Helper: convert logits tensor -> probs numpy using softmax
    def tensor_logits_to_probs(tensor):
        nonlocal F
        if isinstance(tensor, torch.Tensor):
            if F is None:
                # softmax fallback using torch.exp / sum
                tensor = tensor.detach().cpu()
                e = torch.exp(tensor)
                s = e.sum(dim=1, keepdim=True)
                probs_t = e / s
                return probs_t.cpu().numpy()
            else:
                return F.softmax(tensor, dim=1).cpu().numpy()
        else:
            # numpy array fallback (logits)
            arr = _np.asarray(tensor)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # numerical stable softmax
            arr_max = arr.max(axis=1, keepdims=True)
            ex = _np.exp(arr - arr_max)
            return ex / ex.sum(axis=1, keepdims=True)

    # Case 1: RL agent with q_network attribute
    if hasattr(model_obj, "q_network") and getattr(model_obj, "q_network") is not None:
        net = getattr(model_obj, "q_network")
        device = torch.device("cpu")
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32)).to(device)
                out = net(xb)
                # If network returns tuple/list take first element
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs_chunk = tensor_logits_to_probs(out)
                probs_list.append(probs_chunk)
        probs = _np.vstack(probs_list)

    # Case 2: model_obj is callable (nn.Module or scripted)
    elif callable(model_obj):
        device = torch.device("cpu")
        # If it's a PyTorch module, make sure to use torch.no_grad
        try:
            is_torch_module = isinstance(model_obj, torch.nn.Module)
        except Exception:
            is_torch_module = False
        if is_torch_module:
            model_obj.eval()
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32)).to(device)
                    out = model_obj(xb)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = tensor_logits_to_probs(out)
                    probs_list.append(probs_chunk)
            probs = _np.vstack(probs_list)
        else:
            # Some scripted modules are callable but not subclass of nn.Module;
            # attempt to call and handle tensor or numpy outputs.
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32))
                    try:
                        out = model_obj(xb)
                    except Exception:
                        # try passing numpy
                        out = model_obj(X[i:i+batch_size])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = tensor_logits_to_probs(out)
                    probs_list.append(probs_chunk)
            probs = _np.vstack(probs_list)

    # Case 3: sklearn-like predict_proba
    elif hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(X)
        probs = _np.asarray(probs)
        if probs.ndim == 1:
            probs = _np.vstack([1 - probs, probs]).T

    else:
        raise RuntimeError("Model object is not callable, has no 'q_network', and has no 'predict_proba' method.")

    # Normalize binary/logit single-column outputs to two-column probs
    if probs.ndim == 1:
        probs = _np.vstack([1 - probs, probs]).T
    elif probs.shape[1] == 1:
        probs = _np.hstack([1 - probs, probs])

    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    return {"probs": probs, "confs": confs, "preds": preds}

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins=10):
    # probs: N x C, labels: N (int)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    return float(ece)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth or scripted model")
    ap.add_argument("--samples", required=False, help="CSV file of sample inputs for inference (rows of raw features)")
    ap.add_argument("--scaler", required=False, help="Path to scaler (joblib pickle)")
    ap.add_argument("--model-class-import", required=False, help="Optional: import path for model class to load state_dict, e.g. src.ml.reinforcement_learning.TradingRLAgent")
    ap.add_argument("--state-import", required=False, help="Optional: import path for state vector builder module, e.g. env.state_vector")
    ap.add_argument("--label-col", default="label", help="If samples CSV has a label column")
    args = ap.parse_args()

    args.model = _strip_quotes(args.model)
    args.samples = _strip_quotes(args.samples)
    args.scaler = _strip_quotes(args.scaler)
    args.state_import = _strip_quotes(args.state_import)
    args.model_class_import = _strip_quotes(args.model_class_import)
    args.label_col = _strip_quotes(args.label_col)

    out_dir = Path("diagnostics")
    out_dir.mkdir(exist_ok=True)

    print("=== Load model ===")
    loaded = try_load_model(args.model, args.model_class_import)
    with open(out_dir / "model_load.json", "w") as f:
        json.dump({"result_type": loaded.get("type"), "note": loaded.get("note","")}, f, indent=2)

    # Load sample data
    X = None
    y = None
    if args.samples:
        df = pd.read_csv(args.samples)
        if args.label_col in df.columns:
            y = df[args.label_col].to_numpy()
            df = df.drop(columns=[args.label_col])
        X = df.to_numpy(dtype=np.float32)
        print(f"Loaded samples: {X.shape}")
    else:
        print("No sample file provided. Skipping sample-based inference.")

    # Load scaler and transform
    scaler = load_scaler(args.scaler)
    if scaler is not None and X is not None:
        try:
            X = scaler.transform(X)
            print("Applied scaler.transform to X")
        except Exception as e:
            print(f"Scaler.transform failed: {e}")

    # If state_import provided, try to call a 'compute_state_vector' function
    if args.state_import and args.samples:
        try:
            module = importlib.import_module(args.state_import)
            if hasattr(module, "compute_state_vector"):
                print("Using compute_state_vector from", args.state_import)
                X_state = np.vstack([module.compute_state_vector(r) for r in df.to_dict(orient='records')])
                X = X_state.astype(np.float32)
                print("Built state vectors:", X.shape)
            else:
                print(f"No compute_state_vector in module {args.state_import}; skipping")
        except Exception as e:
            print(f"Import of {args.state_import} failed: {e}")

    report = {"model_load": loaded.get("type"), "notes": []}

    # Basic X stats and checks
    if X is not None:
        report["X_shape"] = X.shape
        report["X_nan_count"] = int(np.isnan(X).sum())
        report["X_inf_count"] = int(np.isinf(X).sum())
        report["X_mean"] = [float(x) for x in np.nanmean(X, axis=0).tolist()[:20]]
        report["X_std"] = [float(x) for x in np.nanstd(X, axis=0).tolist()[:20]]

    # Run inference if possible
    if X is not None:
        try:
            res = compute_confidences_and_stats(loaded.get("model") or loaded.get("obj"), X)
            probs = res["probs"]
            confs = res["confs"]
            preds = res["preds"]
            report["conf_mean"] = float(np.mean(confs))
            report["conf_median"] = float(np.median(confs))
            report["conf_std"] = float(np.std(confs))
            # class-wise
            n_classes = probs.shape[1]
            report["class_confidence_mean"] = {str(i): float(np.mean(probs[:,i])) for i in range(n_classes)}
            # Brier - requires one-hot labels
            if y is not None:
                # try to map y to 0..C-1 if needed
                if y.max() >= n_classes:
                    # attempt to map via unique
                    uniq = np.unique(y)
                    mapping = {v:i for i,v in enumerate(uniq)}
                    y_mapped = np.array([mapping[v] for v in y])
                else:
                    y_mapped = y
                # Brier per class averaged
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(y_mapped)), y_mapped.astype(int)] = 1
                brier = np.mean(np.sum((probs - onehot)**2, axis=1))
                report["brier_score"] = float(brier)
                ece = expected_calibration_error(probs, y_mapped, n_bins=10)
                report["ece"] = ece
            # Save histogram
            plt.figure(figsize=(6,4))
            plt.hist(confs, bins=50, range=(0,1))
            plt.title("Confidence histogram")
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(out_dir / "confidence_hist.png")
            plt.close()
            # reliability diagram (simple)
            bins = np.linspace(0,1,11)
            bin_centers = 0.5*(bins[:-1]+bins[1:])
            accuracies = []
            avg_conf = []
            for i in range(len(bins)-1):
                m = (confs > bins[i]) & (confs <= bins[i+1])
                if m.sum()==0:
                    accuracies.append(np.nan)
                    avg_conf.append(np.nan)
                    continue
                if y is None:
                    accuracies.append(None)
                else:
                    accuracies.append(float((preds[m] == y[m]).mean()))
                avg_conf.append(float(confs[m].mean()))
            with open(out_dir / "reliability.json","w") as f:
                json.dump({"bin_centers": bin_centers.tolist(), "accuracies": accuracies, "avg_conf": avg_conf}, f, indent=2)
            # Save probs sample
            np.save(out_dir / "probs_sample.npy", probs[:min(len(probs),2000)])
        except Exception as e:
            report["inference_error"] = str(e)

    # Save report
    with open(out_dir / "report.json","w") as f:
        json.dump(report, f, indent=2)
    print("Diagnostics written to diagnostics/")

if __name__ == "__main__":
    main()
