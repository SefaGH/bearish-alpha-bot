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
def try_load_model(model_path: str, model_class_import: Optional[str]=None):
    # Sanitize model_path in case workflow input included surrounding quotes
    model_path = str(model_path).strip().strip('"').strip("'")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    # Try torch.jit load first
    try:
        m = torch.jit.load(str(model_path), map_location='cpu')
        return {"type":"scripted", "model": m}
    except Exception:
        pass
    # Try torch.load
    try:
        obj = torch.load(str(model_path), map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"torch.load failed: {e}")
    # If obj is state_dict and model class provided, import and load
    if isinstance(obj, dict) and model_class_import:
        module_path, class_name = model_class_import.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        Klass = getattr(mod, class_name)
        model = Klass()
        try:
            state = obj if "state_dict" not in obj else obj["state_dict"]
            model.load_state_dict(state)
            model.eval()
            return {"type":"nn_module", "model": model}
        except Exception as e:
            return {"type":"dict", "obj": obj, "note": f"loaded dict but failed to load into {model_class_import}: {e}"}
    # If obj is nn.Module saved directly
    if hasattr(obj, "eval"):
        obj.eval()
        return {"type":"nn_module_saved", "model": obj}
    # Otherwise return dict
    return {"type":"dict", "obj": obj}

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
    Expects model_obj to be a PyTorch module or a callable that accepts numpy arrays.
    """
    probs_list = []
    model = model_obj
    # If module/callable that accepts tensors
    if hasattr(model, "forward") or hasattr(model, "__call__"):
        device = torch.device("cpu")
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
                out = model(xb)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                out = out.cpu().numpy()
                probs_list.append(out)
        probs = np.vstack(probs_list)
    else:
        # fallback if model object has predict_proba
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X)
        else:
            raise RuntimeError("Model object not callable and has no predict_proba.")
    # Normalize if needed
    if probs.ndim == 1:
        # maybe binary returning single logit -> convert via sigmoid
        probs = np.stack([1 - (1/(1+np.exp(probs))), 1/(1+np.exp(probs))], axis=1)
    elif probs.shape[1] == 1:
        probs = np.hstack([1 - probs, probs])
    # confidences = max prob per row
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
