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

def try_load_model(model_path: str, model_class_import: Optional[str] = None):
    """
    Robust loader for models and checkpoints.

    Behavior summary:
    - sanitize incoming strings
    - if path is scripted torch module -> torch.jit.load
    - attempt torch.load; handle:
       * pickled nn.Module saved directly
       * dict-style checkpoint with keys like 'state_dict', 'model_state_dict',
         'agent_state_dict', 'q_network_state_dict', or arbitrary nested state_dicts
    - if a model class import path is provided, import class and attempt to:
       * instantiate intelligently (inspect signature + fallback defaults from config)
       * call class.load_model(path) if available
       * map known state_dict keys into the instance (e.g., model.load_state_dict, model.q_network.load_state_dict)
    - return a dict describing result: {"type": <...>, "model": <model_or_obj>, "note": <optional>}
    """
    import inspect
    import yaml

    def _strip(s):
        return None if s is None else str(s).strip().strip('"').strip("'")

    model_path = _strip(model_path)
    model_class_import = _strip(model_class_import)
    

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 1) Try torch.jit scripted module
    try:
        scripted = torch.jit.load(str(model_path), map_location="cpu")
        scripted.eval()
        return {"type": "scripted", "model": scripted, "note": ""}
    except Exception:
        pass

    # 2) Try torch.load
    try:
        obj = torch.load(str(model_path), map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"torch.load failed: {e}")

    # If obj is already an nn.Module-like (saved via torch.save(module))
    if hasattr(obj, "eval"):
        try:
            obj.eval()
        except Exception:
            pass
        return {"type": "nn_module_saved", "model": obj, "note": ""}

    # Helper: try to find a reasonable state_dict inside a checkpoint dict
    def extract_state_dicts(checkpoint: dict):
        """
        Returns a dict of discovered state_dicts keyed by label, e.g.
        {'root': <state_dict>, 'q_network': <state_dict>, ...}
        """
        found = {}
        # common container keys
        common_keys = ['state_dict', 'model_state_dict', 'agent_state_dict']
        for k in common_keys:
            if k in checkpoint and isinstance(checkpoint[k], dict):
                found['root'] = checkpoint[k]
                break
        # direct q_network key
        for candidate in ['q_network_state_dict', 'qnet_state_dict', 'q_network', 'qnet']:
            if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
                found['q_network'] = checkpoint[candidate]
        # scan for any key that looks like *.state_dict or endswith 'state_dict'
        for k, v in checkpoint.items():
            if k.endswith('state_dict') and isinstance(v, dict):
                key_label = k[:-11] if k != 'state_dict' else 'root'
                if key_label == '':
                    key_label = 'root'
                found[key_label] = v
        # also, sometimes checkpoint is nested under 'model' or 'agent'
        for k in ('model', 'agent'):
            if k in checkpoint and isinstance(checkpoint[k], dict):
                for subk, subv in checkpoint[k].items():
                    if isinstance(subv, dict) and any(x in subk.lower() for x in ('state', 'state_dict')):
                        found[subk] = subv
        # If nothing found but checkpoint looks like a mapping of parameters (tensor values)
        # Heuristic: many keys look like 'layer.weight' => treat checkpoint itself as state_dict
        if not found:
            sample_keys = list(checkpoint.keys())[:10]
            if any('.' in str(k) for k in sample_keys):
                found['root'] = checkpoint
        return found

    # If torch.load returned a dict, analyze it
    if isinstance(obj, dict):
        state_dicts = extract_state_dicts(obj)

        # If no model class import provided, return the dict (but include discovered state_dict keys)
        if not model_class_import:
            note = f"checkpoint_dict_keys={list(obj.keys())} discovered_state_dict_keys={list(state_dicts.keys())}"
            return {"type": "dict", "obj": obj, "note": note}

        # otherwise try to import and construct the class
        try:
            module_path, class_name = model_class_import.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            Klass = getattr(mod, class_name)
        except Exception as e:
            return {"type": "dict", "obj": obj, "note": f"failed to import {model_class_import}: {e}"}

        # Try to instantiate Klass intelligently
        instance = None
        try:
            # try zero-arg constructor
            instance = Klass()
        except Exception as ctor_err:
            # inspect signature and attempt to build kwargs from config
            try:
                sig = inspect.signature(Klass.__init__)
                params = list(sig.parameters.keys())[1:]  # skip self
                # load config example for sensible defaults if available
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

                # heuristics for common param names
                ctor_kwargs = {}
                if "state_size" in params:
                    state_size = rl_cfg.get("state_size") or rl_cfg.get("state_dim") or 50
                    ctor_kwargs["state_size"] = int(state_size)
                if "action_size" in params:
                    action_size = rl_cfg.get("action_size") or 3
                    ctor_kwargs["action_size"] = int(action_size)
                if "config" in params:
                    ctor_kwargs["config"] = rl_cfg if isinstance(rl_cfg, dict) else {}

                instance = Klass(**ctor_kwargs)
            except Exception as e:
                return {"type": "dict", "obj": obj, "note": f"failed to construct {model_class_import}: {e}"}

        # At this point we have an instance. Prefer instance.load_model(path) if exists.
        try:
            if hasattr(instance, "load_model"):
                try:
                    # try path-based loader first
                    instance.load_model(str(model_path))
                    instance.eval() if hasattr(instance, "eval") else None
                    return {"type": "nn_module_loaded_via_class", "model": instance, "note": "loaded via load_model(path)"}
                except Exception:
                    # try passing checkpoint dict
                    try:
                        instance.load_model(obj)
                        instance.eval() if hasattr(instance, "eval") else None
                        return {"type": "nn_module_loaded_via_class", "model": instance, "note": "loaded via load_model(dict)"}
                    except Exception:
                        pass
            # If load_model not available or failed, try mapping discovered state_dicts to instance parts
            loaded_any = False
            # 1) if instance has load_state_dict and we have root state dict
            if hasattr(instance, "load_state_dict") and "root" in state_dicts:
                try:
                    instance.load_state_dict(state_dicts["root"])
                    loaded_any = True
                except Exception:
                    # maybe the state dict keys have 'module.' prefix or differ; let caller handle later
                    pass
            # 2) try to load q_network state if present
            if hasattr(instance, "q_network") and "q_network" in state_dicts:
                try:
                    instance.q_network.load_state_dict(state_dicts["q_network"])
                    loaded_any = True
                except Exception:
                    pass
            # 3) try other keys by name: for each discovered key try to set attribute and load_state_dict
            for key, sd in state_dicts.items():
                if key in ("root", "q_network"):
                    continue
                if hasattr(instance, key):
                    attr = getattr(instance, key)
                    if hasattr(attr, "load_state_dict"):
                        try:
                            attr.load_state_dict(sd)
                            loaded_any = True
                        except Exception:
                            pass
            if loaded_any:
                try:
                    if hasattr(instance, "eval"):
                        instance.eval()
                except Exception:
                    pass
                return {"type": "nn_module_loaded_via_class", "model": instance, "note": "loaded via mapping discovered state_dicts"}
            # fallback: try to set attributes from checkpoint that match exactly
            try:
                # attempt common attribute names
                for k in ("state_dict", "model_state_dict", "agent_state_dict"):
                    if k in obj and isinstance(obj[k], dict) and hasattr(instance, "load_state_dict"):
                        instance.load_state_dict(obj[k])
                        if hasattr(instance, "eval"):
                            instance.eval()
                        return {"type": "nn_module_loaded_via_class", "model": instance, "note": f"loaded via obj['{k}']"}
            except Exception:
                pass

            # Last resort: if instance has attribute 'q_network' but we didn't find q_network state,
            # maybe whole state dict maps directly — try to load the root
            if hasattr(instance, "load_state_dict") and "root" in state_dicts:
                try:
                    instance.load_state_dict(state_dicts["root"])
                    instance.eval() if hasattr(instance, "eval") else None
                    return {"type": "nn_module_loaded_via_class", "model": instance, "note": "loaded via fallback root"}
                except Exception:
                    pass

        except Exception:
            pass

        # If we reach here, return the checkpoint dict (with discovery note)
        return {"type": "dict", "obj": obj, "note": f"Could not load into {model_class_import}; discovered_state_dicts={list(state_dicts.keys())}"}

    # If obj is something else (e.g., list/tuple), just return it
    return {"type": "unknown", "obj": obj, "note": "torch.load returned non-dict object"}

def compute_confidences_and_stats(model_obj, X: np.ndarray, batch_size: int = 512):
    """
    Return dict {"probs": probs, "confs": confs, "preds": preds}.
    Behavior:
      - If model_obj has attribute 'q_network' (PyTorch nn.Module), use it to compute logits -> softmax -> probs.
      - Else if model_obj is a torch.nn.Module or callable, call it with tensors -> softmax -> probs.
      - Else if model_obj has predict_proba, use it.
      - Otherwise raise RuntimeError.
    """
    import numpy as _np
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        torch = None
        F = None

    if X is None or len(X) == 0:
        raise RuntimeError("Empty X passed to compute_confidences_and_stats")

    probs_list = []

    def logits_to_probs(x):
        # x may be torch.Tensor or numpy array
        if 'torch' in globals() and isinstance(x, torch.Tensor):
            if F is not None:
                return F.softmax(x, dim=1).cpu().numpy()
            else:
                t = x.detach().cpu()
                e = torch.exp(t - t.max(dim=1, keepdim=True)[0])
                p = (e / e.sum(dim=1, keepdim=True)).cpu().numpy()
                return p
        else:
            arr = _np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # stable softmax
            arr = arr - arr.max(axis=1, keepdims=True)
            ex = _np.exp(arr)
            return ex / ex.sum(axis=1, keepdims=True)

    # Case A: agent has q_network
    if hasattr(model_obj, "q_network") and getattr(model_obj, "q_network") is not None:
        net = getattr(model_obj, "q_network")
        if 'torch' not in globals():
            raise RuntimeError("torch is required to use q_network")
        device = torch.device("cpu")
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32)).to(device)
                out = net(xb)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs_chunk = logits_to_probs(out)
                probs_list.append(probs_chunk)
        probs = _np.vstack(probs_list)

    # Case B: callable model (nn.Module/scripted)
    elif callable(model_obj):
        # prefer torch if module
        try:
            is_torch = 'torch' in globals() and isinstance(model_obj, torch.nn.Module)
        except Exception:
            is_torch = False
        if is_torch:
            model_obj.eval()
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32))
                    out = model_obj(xb)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = logits_to_probs(out)
                    probs_list.append(probs_chunk)
            probs = _np.vstack(probs_list)
        else:
            # try sklearn-like or callable that returns numpy
            if hasattr(model_obj, "predict_proba"):
                probs = model_obj.predict_proba(X)
                probs = _np.asarray(probs)
            else:
                # try calling with numpy batches
                for i in range(0, len(X), batch_size):
                    try:
                        out = model_obj(X[i:i+batch_size])
                    except Exception:
                        out = model_obj(X[i:i+batch_size].astype(_np.float32))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    probs_chunk = logits_to_probs(out)
                    probs_list.append(probs_chunk)
                probs = _np.vstack(probs_list)

    # Case C: sklearn-like predict_proba attribute (fallback)
    elif hasattr(model_obj, "predict_proba"):
        probs = _np.asarray(model_obj.predict_proba(X))

    else:
        raise RuntimeError("Model object is not callable, has no q_network, and has no predict_proba")

    # Normalize shapes (binary -> 2-col, 1D -> softmax)
    if probs.ndim == 1:
        probs = _np.vstack([1 - probs, probs]).T
    elif probs.shape[1] == 1:
        probs = _np.hstack([1 - probs, probs])

def _compute_reliability(probs, labels=None, n_bins=10):
    import numpy as _np
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = _np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = ((bins[:-1] + bins[1:]) / 2).tolist()
    accuracies = [None] * n_bins
    avg_conf = [None] * n_bins
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            accuracies[i] = None
            avg_conf[i] = None
            continue
        if labels is not None:
            acc = float((preds[mask] == labels[mask]).mean())
            accuracies[i] = acc
        else:
            accuracies[i] = None
        avg_conf[i] = float(confidences[mask].mean())
    return {"bin_centers": bin_centers, "accuracies": accuracies, "avg_conf": avg_conf}

def _plot_confidence_hist(confs, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3))
    plt.hist(confs, bins=10, range=(0.0,1.0), color="#1f77b4")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence histogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pth or scripted model")
    ap.add_argument("--samples", required=False, help="CSV file of sample inputs for inference (rows of raw features)")
    ap.add_argument("--scaler", required=False, help="Path to scaler (joblib pickle)")
    ap.add_argument("--model-class-import", required=False, help="Optional: import path for model class to load state_dict, e.g. src.ml.reinforcement_learning.TradingRLAgent")
    ap.add_argument("--state-import", required=False, help="Optional: import path for state vector builder module, e.g. scripts.state_vector_wrapper")
    ap.add_argument("--label-col", default="label", help="If samples CSV has a label column")
    args = ap.parse_args()

    args.model = args.model.strip() if args.model else None
    args.samples = args.samples.strip() if args.samples else None
    args.scaler = args.scaler.strip() if args.scaler else None
    args.state_import = args.state_import.strip() if args.state_import else None
    args.model_class_import = args.model_class_import.strip() if args.model_class_import else None

    report = {}
    X = None
    df = None

    # Load samples CSV (if provided)
    if args.samples:
        df = pd.read_csv(args.samples)
        report["X_shape_raw"] = list(df.shape)
    else:
        df = None

    # If a state-import module is provided and has compute_state_vector, use it.
    if args.state_import and df is not None:
        try:
            mod = importlib.import_module(args.state_import)
            if hasattr(mod, "compute_state_vector"):
                records = df.to_dict(orient="records")
                X_list = []
                for r in records:
                    sv = mod.compute_state_vector(r)
                    X_list.append(sv)
                X = np.vstack(X_list).astype(np.float32)
                print("Using compute_state_vector from", args.state_import)
                report["notes"] = report.get("notes", []) + [f"Used state_import: {args.state_import}"]
        except Exception as e:
            print(f"Import of {args.state_import} failed: {e}")
            report["notes"] = report.get("notes", []) + [f"state_import failed: {str(e)}"]

    # If compute_state_vector not used, but df has f0..fN style columns, use them
    if X is None and df is not None:
        f_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
        if f_cols:
            # sort by numeric index
            f_cols = sorted(f_cols, key=lambda k: int(k[1:]))
            X = df[f_cols].to_numpy(dtype=np.float32)
            print("Using f* columns:", len(f_cols))
            report["notes"] = report.get("notes", []) + [f"Used {len(f_cols)} f* columns"]
        else:
            # fallback: use numeric columns in natural order (excluding label)
            numeric_df = df.select_dtypes(include=[np.number]).copy() if df is not None else None
            if numeric_df is not None and args.label_col in numeric_df.columns:
                numeric_df = numeric_df.drop(columns=[args.label_col])
            if numeric_df is not None and numeric_df.shape[1] > 0:
                X = numeric_df.to_numpy(dtype=np.float32)
                report["notes"] = report.get("notes", []) + [f"Used numeric columns fallback: {numeric_df.shape[1]} cols"]

    # Apply scaler if provided
    scaler = None
    if args.scaler and os.path.exists(args.scaler):
        try:
            scaler = joblib.load(args.scaler)
            if X is not None:
                try:
                    X = scaler.transform(X)
                    print("Applied scaler.transform to X")
                    report["notes"] = report.get("notes", []) + [f"Applied scaler: {args.scaler}"]
                except Exception as e:
                    print("Scaler.transform failed:", e)
                    report["notes"] = report.get("notes", []) + [f"scaler.transform failed: {e}"]
        except Exception as e:
            print("Scaler not found or failed to load:", e)
            report["notes"] = report.get("notes", []) + [f"scaler load failed: {e}"]

    # If still no X, error
    if X is None:
        report["error"] = "No input features X could be built from samples."
        os.makedirs("diagnostics", exist_ok=True)
        with open("diagnostics/report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("No X constructed; exiting")
        return

    # Basic X stats
    report["X_shape"] = [int(X.shape[0]), int(X.shape[1])]
    report["X_nan_count"] = int(np.isnan(X).sum())
    report["X_inf_count"] = int(np.isinf(X).sum())
    report["X_mean"] = np.nanmean(X, axis=0)[:20].tolist()  # small preview
    report["X_std"] = np.nanstd(X, axis=0)[:20].tolist()

    # Load model using robust loader
    loaded = try_load_model(args.model, args.model_class_import)
    model_obj = loaded.get("model") if isinstance(loaded, dict) and "model" in loaded else loaded
    report["model_load"] = loaded.get("type") if isinstance(loaded, dict) else str(type(loaded))

    # Compute probabilities / confidences
    probs = None
    confs = None
    preds = None
    try:
        res = compute_confidences_and_stats(model_obj, X)
        probs = res["probs"]
        confs = res["confs"]
        preds = res["preds"]
    except Exception as e:
        print("Error computing confidences:", e)
        report["inference_error"] = str(e)

    # Save probs for inspection
    os.makedirs("diagnostics", exist_ok=True)
    if probs is not None:
        try:
            np.save("diagnostics/probs_sample.npy", probs.astype(np.float32))
        except Exception as e:
            print("Failed to save probs_sample.npy:", e)

    # Update report with confidence statistics
    if confs is not None:
        report["conf_mean"] = float(np.mean(confs))
        report["conf_median"] = float(np.median(confs))
        report["conf_std"] = float(np.std(confs))
        # mean prob per class
        try:
            class_means = {str(i): float(probs[:, i].mean()) for i in range(probs.shape[1])}
            report["class_confidence_mean"] = class_means
        except Exception:
            pass

    # If labels present, compute Brier and ECE
    if df is not None and args.label_col in df.columns and probs is not None:
        try:
            y_true = df[args.label_col].to_numpy().astype(int)
            # multiclass Brier
            y_onehot = np.zeros_like(probs)
            for i, yy in enumerate(y_true):
                if 0 <= int(yy) < probs.shape[1]:
                    y_onehot[i, int(yy)] = 1.0
            report["brier_score"] = float(((probs - y_onehot) ** 2).sum(axis=1).mean())
            # ECE
            def compute_ece(probs_arr, labels_arr, n_bins=10):
                confidences = probs_arr.max(axis=1)
                predictions = probs_arr.argmax(axis=1)
                ece = 0.0
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                for i in range(n_bins):
                    mask = (confidences > bins[i]) & (confidences <= bins[i+1])
                    if mask.sum() == 0:
                        continue
                    acc = (predictions[mask] == labels_arr[mask]).mean()
                    avg_conf = confidences[mask].mean()
                    ece += (mask.sum() / len(confidences)) * abs(avg_conf - acc)
                return float(ece)
            report["ece"] = compute_ece(probs, y_true, n_bins=10)
        except Exception as e:
            report["brier_ece_error"] = str(e)

    # Reliability (binning) and save
    try:
        rel = _compute_reliability(probs, labels=(df[args.label_col].to_numpy() if (df is not None and args.label_col in df.columns) else None), n_bins=10)
        with open("diagnostics/reliability.json", "w") as f:
            json.dump(rel, f, indent=2)
    except Exception as e:
        report["reliability_error"] = str(e)

    # Save confidence histogram
    if confs is not None:
        try:
            _plot_confidence_hist(confs, "diagnostics/confidence_hist.png")
        except Exception as e:
            report["hist_plot_error"] = str(e)

    # Final report write
    with open("diagnostics/report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()

    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    return {"probs": probs, "confs": confs, "preds": preds}
