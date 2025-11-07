#!/usr/bin/env python3
"""
diagnose_model_confidence.py
---------------------------------
Loads a model (optional) and a CSV of samples, runs inference to compute per-sample
probabilities, confidences, and predictions, and writes a JSON report.

BU SÜRÜM: "Analiz J"ye göre otomatik ölçeklendirme (auto-scaling) 
mantığını (`try_scalings_and_choose`) içerir.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, List, Tuple

import importlib
import numpy as np
import math # _entropy_rows için

# 'try_scalings_and_choose' için numpy'ı _np olarak da import et
import numpy as _np
import math as _math
from typing import Tuple, Dict

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

# joblib is optional for sklearn pickle loads
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# Optional YAML (when configs are used to describe models)
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# "safe_torch_load"u import ediyoruz
from scripts.safe_torch_load import safe_torch_load

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def load_samples(csv_path: str, limit: Optional[int] = None, expected_shape: Optional[int] = None) -> np.ndarray:
    """
    CSV'yi yükler. 'label' sütununu atar.
    Eğer expected_shape verilirse, veriyi [:, :expected_shape]
    şeklinde keser (slice).
    """
    if pd is None:
        raise RuntimeError("pandas is required to read CSV files")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV is empty")

    num_df = df.select_dtypes(include=["number"]).copy()
    
    # 'label' sütununu (ve benzerlerini) çıkar
    label_cols_found = []
    for drop_col in ("label", "target", "y", "timestamp", "time", "datetime"):
        if drop_col in num_df.columns: # Orijinal df'ten çıkar
            label_cols_found.append(drop_col)
            
    if label_cols_found:
        print(f"Dropping non-feature columns: {label_cols_found}")
        num_df = num_df.drop(columns=label_cols_found, errors='ignore')

    if num_df.empty:
        # Eğer 'label'ı attıktan sonra sayısal sütun kalmazsa, tüm df'i dene
        # (ancak label'ları tekrar çıkar)
        if label_cols_found:
             X = df.drop(columns=label_cols_found, errors='ignore').values
        else:
             X = df.values
    else:
        X = num_df.values

    if limit is not None and limit > 0:
        X = X[:limit]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # --- VERİ KESME (SLICING) BLOĞU ---
    if expected_shape is not None:
        print(f"Data shape is {X.shape}, model expects {expected_shape}.")
        if X.shape[1] > expected_shape:
            # Slicing from {X.shape[1]} to {expected_shape}
            X = X[:, :expected_shape]
        elif X.shape[1] < expected_shape:
            print(f"[WARN] Data shape ({X.shape[1]}) is smaller than model expected shape ({expected_shape})!")
    # --- BLOK SONU ---
            
    return X.astype(np.float32, copy=False)


def try_load_model(model_path: Optional[str], model_class_import: Optional[str] = None) -> Dict[str, Any]:
    """
    'safe_torch_load' kullanarak modeli yükler.
    """
    if not model_path:
        return {"model": None, "note": "No model path provided"}
    if not os.path.exists(model_path):
        return {"model": None, "note": f"Model path does not exist: {model_path}"}

    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        if torch is None:
            return {"model": None, "note": "torch not available"}
        try:
            # 'safe_torch_load'u çağırıyoruz
            obj = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
            return {"model": obj, "note": "Loaded via safe_torch_load"}
        except Exception as e:
            # Hata mesajını yakala
            return {"model": None, "note": f"safe_torch_load failed: {e}"}
            
    # Diğer dosya türleri (pickle, yaml) değişmedi...
    if ext in (".pkl", ".pickle"):
        try:
            if joblib is not None: obj = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, "rb") as f: obj = pickle.load(f)
            return {"model": obj, "note": "Loaded via joblib/pickle"}
        except Exception as e:
            return {"model": None, "note": f"Failed pickle/joblib load: {e}"}
    if ext in (".yml", ".yaml"):
        if yaml is None: return {"model": None, "note": "PyYAML not available"}
        try:
            with open(model_path, "r", encoding="utf-8") as f: y = yaml.safe_load(f)
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

# --------------------------------------------------------------------------------------
# "ANALİZ J"DEN GELEN OTOMATİK ÖLÇEKLENDİRME (AUTO-SCALING) FONKSİYONLARI
# --------------------------------------------------------------------------------------

def _entropy_rows(probs: _np.ndarray) -> _np.ndarray:
    # probs: (N, C)
    eps = 1e-12
    p = _np.clip(probs, eps, 1.0)
    return -_np.sum(p * _np.log(p), axis=1)

def _model_probs_for_X(model_obj, X: _np.ndarray) -> _np.ndarray:
    """
    Try to run model.q_network forward and softmax; fallbacks if model provides act/predict_proba.
    Returns numpy array shape (N, C) or raises.
    """
    import torch
    from torch.nn import functional as F

    # try predict_proba
    try:
        if hasattr(model_obj, "predict_proba"):
            out = model_obj.predict_proba(X)
            return _np.asarray(out)
    except Exception:
        pass

    # try act OR q_network
    try:
        # if model expects torch tensors, convert
        xt = torch.tensor(X, dtype=torch.float32)
        if hasattr(model_obj, "q_network"):
            with torch.no_grad():
                logits = model_obj.q_network(xt)
                probs = F.softmax(logits, dim=-1)
                return probs.detach().cpu().numpy()
        # try calling model directly
        with torch.no_grad():
            logits = model_obj(xt)
            probs = F.softmax(logits, dim=-1)
            return probs.detach().cpu().numpy()
    except Exception:
        # last-resort: try act per-sample returning int/class -> convert to one-hot
        if hasattr(model_obj, "act"):
            outs = []
            for row in X:
                try:
                    a = model_obj.act(row)
                    # if act returns int action index
                    if isinstance(a, int):
                        v = _np.zeros(getattr(model_obj, "action_size", 3))
                        v[a] = 1.0
                        outs.append(v)
                    else:
                        outs.append(_np.asarray(a))
                except Exception:
                    outs.append(None)
            # Fill Nones with uniform prob
            num_actions = getattr(model_obj, "action_size", 3)
            default_prob = _np.full((num_actions,), 1.0 / num_actions)
            return _np.asarray([o if o is not None else default_prob for o in outs])
    
    raise RuntimeError("Model forward failed for all known interfaces (q_network, callable, predict_proba, act)")

def try_scalings_and_choose(X: _np.ndarray, model_obj, checkpoint: Dict = None) -> Tuple[_np.ndarray, Dict]:
    """
    Try several scaling strategies on X, run model, compute mean entropy and choose best (lowest mean entropy).
    Returns (X_chosen, metadata) where metadata contains method and per-method stats.
    """
    print("Starting auto-scaling probe... Trying different scaling strategies.")
    results = {}
    methods = []

    # prepare input
    Xf = _np.asarray(X, dtype=float)
    N, D = Xf.shape

    # candidate: checkpoint scaler if present
    if checkpoint and isinstance(checkpoint, dict):
        if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
            try:
                mean = _np.asarray(checkpoint["scaler_mean"], dtype=float)
                std = _np.asarray(checkpoint["scaler_std"], dtype=float)
                # Boyutları (shape) kontrol et
                if mean.shape == (D,) and std.shape == (D,):
                    Xc = (Xf - mean.reshape(1, -1)) / (std.reshape(1, -1) + 1e-12)
                    methods.append(("checkpoint", Xc))
                else:
                    print(f"[WARN] Checkpoint scaler dims ({mean.shape}) mismatch data dims ({D}). Skipping.")
            except Exception:
                pass # Checkpoint scaler başarısız oldu

    # 'none' (raw data)
    methods.append(("none", Xf.copy()))

    # z-score (use sample stats)
    means = Xf.mean(axis=0)
    stds = Xf.std(axis=0)
    stds_safe = stds + 1e-12 # 0'a bölmeyi engelle
    z = (Xf - means.reshape(1, -1)) / stds_safe.reshape(1, -1)
    methods.append(("zscore_sample", z))

    # min-max
    mn = Xf.min(axis=0)
    mx = Xf.max(axis=0)
    denom_mm = (mx - mn) + 1e-12
    denom_mm[denom_mm < 1e-12] = 1.0 # 0'a bölmeyi engelle
    mm = (Xf - mn.reshape(1, -1)) / denom_mm.reshape(1, -1)
    methods.append(("minmax_sample", mm))

    # div by maxabs
    maxabs = _np.max(_np.abs(Xf), axis=0)
    maxabs_adj = maxabs.copy()
    maxabs_adj[maxabs_adj == 0] = 1.0
    dm = Xf / maxabs_adj.reshape(1, -1)
    methods.append(("maxabs", dm))

    # evaluate each method
    best = None
    for name, Xm in methods:
        try:
            probs = _model_probs_for_X(model_obj, Xm)  # (N, C)
            ent = _entropy_rows(probs)
            mean_ent = float(_np.mean(ent))
            results[name] = {"mean_entropy": mean_ent, "entropy_per_sample": ent.tolist(), "probs_example": probs[:5].tolist()}
            if best is None or mean_ent < best[0]:
                best = (mean_ent, name, Xm, probs)
        except Exception as e:
            results[name] = {"error": str(e)}

    if best is None:
        print("[ERROR] All scaling attempts failed.")
        raise RuntimeError("All scaling attempts failed")

    chosen_entropy, chosen_name, chosen_X, chosen_probs = best
    meta = {"chosen_method": chosen_name, "chosen_entropy": float(chosen_entropy), "per_method": results}
    print(f"Auto-scaling complete. Best method: '{chosen_name}' (Entropy: {chosen_entropy:.4f})")
    
    # Not: X_chosen'i (ölçeklenmiş veri) değil, asıl probs'u döndürüyoruz.
    # compute_confidences_and_stats zaten probs'u alıyor.
    return chosen_probs, meta

# --------------------------------------------------------------------------------------
# compute_confidences_and_stats'in GÜNCELLENMİŞ versiyonu
# --------------------------------------------------------------------------------------
def compute_confidences_and_stats(model_obj: Any, X: np.ndarray, batch_size: int = 512, checkpoint: Dict = None) -> Dict[str, Any]:
    """
    Runs auto-scaling probe, selects best probabilities, and returns stats.
    """
    
    # 'try_scalings_and_choose'u çağırıyoruz
    # Bu fonksiyon artık X'i değil, en iyi 'probs' dizisini döndürür
    probs, scale_meta = try_scalings_and_choose(X, model_obj, checkpoint=checkpoint)
    
    # Probs üzerinden istatistikleri hesapla
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T
    elif probs.shape[1] == 1:
        probs = np.hstack([1 - probs, probs])

    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    
    return {
        "probs": probs, 
        "confs": confs, 
        "preds": preds,
        "scale_meta": scale_meta # Raporlama için ölçeklendirme meta verisini ekle
    }

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def build_report(X: np.ndarray,
                 result: Optional[Dict[str, np.ndarray]],
                 load_note: str,
                 error: Optional[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"samples_count": int(len(X)), "model_load_note": load_note}
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
    
    # Ölçeklendirme meta verisini rapora ekle
    if "scale_meta" in result:
        report["scaling_results"] = result["scale_meta"]
        
    return report


def save_report(report: Dict[str, Any], out_path: str) -> None:
    # ... (bu fonksiyon değişmedi) ...
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str) # numpy int/float için default=str eklendi


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
    
    # 'safe_torch_load' ve 'checkpoint' okuması için bu argümanlara ihtiyacımız var
    p.add_argument("--model-class-import", default=None, help="Python import path for model class (e.g. src.agent.MyAgent)")
    p.add_argument("--original-model-path", default=None, help="Path to the original checkpoint file (for reading config/scaler)")
    
    return p.parse_args(argv)


def _write_diagnostics(out_dir: str, result: Dict[str, np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # X_scaled.npy kaydetmek yerine artık probs_sample.npy'yi kaydediyoruz
    np.save(os.path.join(out_dir, "probs_sample.npy"), result["probs"])
    confs = result["confs"]
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    hist = np.histogram(confs, bins=bins)[0].tolist()
    rel = {"bins": bins, "counts": hist, "avg_confidence": float(confs.mean()) if len(confs) else 0.0, "n": int(len(confs))}
    with open(os.path.join(out_dir, "reliability.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    args = parse_args(argv)
    
    report_extras = {}
    inference_err = None
    
    # Modelin beklediği state_size'ı (42) dosyadan oku
    expected_state_size = None
    try:
        with open("diagnostics/inferred_state_size.txt", "r") as f:
            expected_state_size = int(f.read().strip())
        print(f"Read expected state_size from file: {expected_state_size}")
    except Exception as e:
        print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")

    # Load samples
    try:
        X = load_samples(args.csv, limit=args.limit, expected_shape=expected_state_size)
    except Exception as e:
        err_report = {
            "samples_count": 0, "model_load_note": "skip (no model used)",
            "inference_ok": False, "inference_error": f"Failed to load samples: {e}",
        }
        save_report(err_report, args.out)
        print(f"[ERROR] {e}")
        return 2

    # Load "canlı" model (inst_model.pth)
    loaded = try_load_model(args.model, args.model_class_import)
    model_obj = loaded.get("model", None)
    load_note = loaded.get("note", "")
    
    # Orijinal checkpoint'i (config/scaler için) yüklemeyi dene
    checkpoint_dict = None
    if args.original_model_path:
        try:
            # 'safe_torch_load'u orijinal (dict) checkpoint için de kullanabiliriz
            checkpoint_dict = safe_torch_load(args.original_model_path, args.model_class_import)
            if not isinstance(checkpoint_dict, dict):
                checkpoint_dict = None # Eğer canlı modelse (yanlışlıkla) None yap
            else:
                load_note += " | Successfully loaded original checkpoint for config."
        except Exception as e:
            load_note += f" | Failed to load original checkpoint: {e}"
    
    # Run inference best-effort
    result = None
    if inference_err is None:
        if model_obj is None:
            inference_err = "No model object available for inference"
        else:
            try:
                # 'checkpoint_dict'i otomatik ölçeklendirme fonksiyonuna iletiyoruz
                result = compute_confidences_and_stats(
                    model_obj, 
                    X, 
                    batch_size=args.batch_size, 
                    checkpoint=checkpoint_dict
                )
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
                np.save(os.path.join(args.out_dir, "X_raw.npy"), X) # Ham X'i kaydet
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
