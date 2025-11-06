#!/usr/bin/env python3
"""
diagnose_model_confidence.py
---------------------------------
Loads a model, auto-detects the best scaling strategy for the input data,
runs inference, and writes a JSON report.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, List, Tuple
import importlib
import math as _math # Scaling fonksiyonları için eklendi

import numpy as _np # Scaling fonksiyonları için _np olarak import edildi

# Optional deps
try:
    import pandas as pd
except Exception: pd = None
try:
    import torch
    import torch.nn.functional as F
except Exception: torch, F = None, None
try:
    import joblib
except Exception: joblib = None
try:
    import yaml
except Exception: yaml = None

# "safe_torch_load"u import ediyoruz
try:
    from scripts.safe_torch_load import safe_torch_load
except ImportError:
    print("WARN: could not import safe_torch_load.py. Falling back to standard torch.load.")
    # Fallback, 'weights_only=False' kullanır, 'safe_globals' denemez
    def safe_torch_load(path, model_class_import=None, map_location="cpu"):
        if model_class_import:
            try:
                # 'pickle'ın sınıfı bulabilmesi için import et
                module_path, class_name = model_class_import.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                Klass = getattr(mod, class_name)
            except Exception:
                pass # Hata olursa, torch.load'un halletmesine izin ver
        return torch.load(path, map_location=map_location, weights_only=False)

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def load_samples(csv_path: str, limit: Optional[int] = None, expected_shape: Optional[int] = None) -> Tuple[_np.ndarray, List[str]]:
    """
    CSV'yi yükler, 'label' sütunlarını atar ve veriyi (X) beklenen şekle (expected_shape)
    göre keser. (X, feature_cols) döndürür.
    """
    if pd is None: raise RuntimeError("pandas required")
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty: raise RuntimeError("CSV is empty")

    num_df = df.select_dtypes(include=["number"]).copy()
    
    label_candidates = [c for c in num_df.columns if c.lower() in ("label", "target", "y", "class")]
    if label_candidates:
        print(f"Dropping non-feature columns: {label_candidates}")
        num_df = num_df.drop(columns=label_candidates)

    feature_cols = num_df.columns.tolist()
    
    if num_df.empty:
        X = df.values
    else:
        X = num_df.values

    if limit is not None and limit > 0:
        X = X[:limit]
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if expected_shape is not None:
        print(f"Data shape is {X.shape}, model expects {expected_shape}.")
        if X.shape[1] > expected_shape:
            print(f"Slicing data from {X.shape[1]} to {expected_shape} features.")
            X = X[:, :expected_shape]
            feature_cols = feature_cols[:expected_shape]
        elif X.shape[1] < expected_shape:
            print(f"[WARN] Data shape ({X.shape[1]}) is smaller than expected ({expected_shape})!")
            
    return X.astype(_np.float32, copy=False), feature_cols


def try_load_model(model_path: Optional[str], model_class_import: Optional[str] = None) -> Dict[str, Any]:
    """
    'safe_torch_load' kullanarak modeli yükler.
    """
    if not model_path: return {"model": None, "note": "No model path provided"}
    if not os.path.exists(model_path): return {"model": None, "note": f"Model path not found: {model_path}"}

    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        if torch is None: return {"model": None, "note": "torch not available"}
        try:
            obj = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
            return {"model": obj, "note": "Loaded via safe_torch_load"}
        except Exception as e:
            return {"model": None, "note": f"safe_torch_load failed: {e}"}
            
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
def _logits_to_probs(x: Any) -> _np.ndarray:
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
    arr = _np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = arr.sum(axis=1, keepdims=True)
    if _np.all(arr >= 0) and _np.allclose(row_sums, 1.0, atol=1e-4):
        return arr
    arr = arr - arr.max(axis=1, keepdims=True)
    ex = _np.exp(arr)
    return ex / ex.sum(axis=1, keepdims=True)


def compute_confidences_and_stats(model_obj: Any, X: _np.ndarray, batch_size: int = 512) -> Dict[str, _np.ndarray]:
    """
    Bu fonksiyon artık 'X' verisinin ZATEN ÖLÇEKLENDİRİLMİŞ (SCALED) olduğunu varsayar.
    """
    probs_list: List[_np.ndarray] = []
    
    # Veriyi _model_probs_for_X ile almayı dene (bu, q_network/act/callable'ı dener)
    try:
        # Hepsini tek seferde (batch) al
        all_probs = _model_probs_for_X(model_obj, X)
        probs_list.append(all_probs)
    except Exception as e:
        # _model_probs_for_X başarısız olursa, eski yönteme (compute_confidences_and_stats'ın eski mantığı)
        # geri dön, ancak bu muhtemelen aynı hatayı verecektir.
        print(f"[WARN] _model_probs_for_X failed: {e}. Falling back to batch processing...")
        if hasattr(model_obj, "q_network") and getattr(model_obj, "q_network") is not None:
            if torch is None: raise RuntimeError("torch is required to use q_network")
            net = getattr(model_obj, "q_network")
            device = torch.device("cpu")
            net.eval()
            with torch.no_grad():
                for i in range(0, len(X), batch_size):
                    xb = torch.from_numpy(X[i:i+batch_size].astype(_np.float32)).to(device)
                    out = net(xb)
                    if isinstance(out, (tuple, list)): out = out[0]
                    probs_chunk = _logits_to_probs(out)
                    probs_list.append(probs_chunk)
        elif callable(model_obj):
             # ... (callable fallback) ...
            pass
        elif hasattr(model_obj, "predict_proba"):
             # ... (predict_proba fallback) ...
            pass
        else:
            raise RuntimeError("Model object is not callable, has no q_network, and has no predict_proba")

    if not probs_list: raise RuntimeError("No outputs produced by model during inference")
    probs = _np.vstack(probs_list)
    if probs.ndim == 1: probs = _np.vstack([1 - probs, probs]).T
    elif probs.shape[1] == 1: probs = _np.hstack([1 - probs, probs])
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    return {"probs": probs, "confs": confs, "preds": preds}

# --------------------------------------------------------------------------------------
# YENİ "OTOMATİK ÖLÇEKLEME" FONKSİYONLARI (ANALİZ J)
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
        # try calling model directly (e.g. if it's a nn.Module)
        if callable(model_obj):
            with torch.no_grad():
                logits = model_obj(xt)
                probs = F.softmax(logits, dim=-1)
                return probs.detach().cpu().numpy()
    except Exception as e:
        print(f"[DEBUG] q_network/callable failed: {e}")
        # last-resort: try act per-sample returning int/class -> convert to one-hot
        if hasattr(model_obj, "act"):
            print("[DEBUG] Falling back to model.act()")
            outs = []
            action_size = getattr(model_obj, "action_size", 3) # default 3
            for row in X:
                try:
                    a = model_obj.act(row)
                    # if act returns int action index
                    if isinstance(a, int):
                        v = _np.zeros(action_size)
                        v[a] = 1.0
                        outs.append(v)
                    else:
                        outs.append(_np.asarray(a))
                except Exception:
                    outs.append(None)
            # Fill Nones with uniform prob
            uniform_prob = _np.full((action_size,), 1.0/action_size)
            return _np.asarray([o if o is not None else uniform_prob for o in outs])
            
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
    if Xf.size == 0:
        raise RuntimeError("Input data (X) for scaling is empty.")
    N, D = Xf.shape

    # 1. candidate: checkpoint scaler if present
    if checkpoint and isinstance(checkpoint, dict):
        if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
            try:
                mean = _np.asarray(checkpoint["scaler_mean"], dtype=float)
                std = _np.asarray(checkpoint["scaler_std"], dtype=float)
                Xc = (Xf - mean.reshape(1, -1)) / (std.reshape(1, -1) + 1e-12)
                methods.append(("checkpoint", Xc))
            except Exception:
                pass # Shape mismatch vb.

    # 2. none (Ham veri)
    methods.append(("none", Xf.copy()))

    # 3. z-score (use sample stats)
    try:
        means = Xf.mean(axis=0)
        stds = Xf.std(axis=0)
        stds[stds == 0] = 1.0 # 0'a bölmeyi engelle
        z = (Xf - means.reshape(1, -1)) / (stds.reshape(1, -1) + 1e-12)
        methods.append(("zscore_sample", z))
    except Exception as e:
        results["zscore_sample_error"] = str(e)

    # 4. min-max [0, 1]
    try:
        mn = Xf.min(axis=0)
        mx = Xf.max(axis=0)
        denom = (mx - mn).reshape(1, -1)
        denom[denom == 0] = 1.0 # 0'a bölmeyi engelle
        mm = (Xf - mn.reshape(1, -1)) / (denom + 1e-12)
        methods.append(("minmax_sample", mm))
    except Exception as e:
        results["minmax_sample_error"] = str(e)

    # 5. div by maxabs [-1, 1]
    try:
        maxabs = _np.max(_np.abs(Xf), axis=0)
        maxabs_adj = maxabs.copy()
        maxabs_adj[maxabs_adj == 0] = 1.0
        dm = Xf / maxabs_adj.reshape(1, -1)
        methods.append(("maxabs", dm))
    except Exception as e:
        results["maxabs_error"] = str(e)


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
            print(f"Scaling method '{name}' failed: {e}")
            results[name] = {"error": str(e)}

    if best is None:
        print("[ERROR] All scaling attempts failed.")
        raise RuntimeError("All scaling attempts failed")

    chosen_entropy, chosen_name, chosen_X, chosen_probs = best
    meta = {"chosen_method": chosen_name, "chosen_entropy": float(chosen_entropy), "per_method": results}
    print(f"Auto-scaling complete. Best method: '{chosen_name}' (Entropy: {chosen_entropy:.4f})")
    
    # X_chosen yerine, 'best' tuple'ından 'chosen_probs'u doğrudan döndür
    return chosen_X, chosen_probs, meta

# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def build_report(X: _np.ndarray,
                 result: Optional[Dict[str, _np.ndarray]],
                 load_note: str,
                 error: Optional[str]) -> Dict[str, Any]:
    # ... (Bu fonksiyon "Analiz J" için değişmedi, ancak _np kullanacak şekilde güncellendi) ...
    report: Dict[str, Any] = {"samples_count": int(len(X)), "model_load_note": load_note}
    if error is not None:
        report["inference_ok"] = False
        report["inference_error"] = str(error)
        return report
    probs = result["probs"]
    confs = result["confs"]
    preds = result["preds"]
    class_counts = {int(c): int((preds == c).sum()) for c in _np.unique(preds)}
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
    # ... (Bu fonksiyon "Analiz J" için değişmedi) ...
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str) # default=str eklendi (numpy array'leri için)


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
    
    # 'safe_torch_load' için bu argümana ihtiyacımız var
    p.add_argument("--model-class-import", default=None, help="Python import path for model class (e.g. src.agent.MyAgent)")
    
    # Orijinal checkpoint'e (config/scaler için) erişmek üzere eklendi
    p.add_argument("--original-model-path", default=None, help="Path to the original .pth checkpoint (for config/scaler)")

    return p.parse_args(argv)


def _write_diagnostics(out_dir: str, result: Dict[str, _np.ndarray]) -> None:
    # ... (Bu fonksiyon "Analiz J" için değişmedi, ancak _np kullanacak şekilde güncellendi) ...
    os.makedirs(out_dir, exist_ok=True)
    _np.save(os.path.join(out_dir, "probs_sample.npy"), result["probs"])
    confs = result["confs"]
    bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    hist = _np.histogram(confs, bins=bins)[0].tolist()
    rel = {"bins": bins, "counts": hist, "avg_confidence": float(confs.mean()) if len(confs) else 0.0, "n": int(len(confs))}
    with open(os.path.join(out_dir, "reliability.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2)


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

    # Veriyi (X_raw) yükle ve doğru boyuta (42) kes
    try:
        X_raw, feature_cols = load_samples(args.csv, limit=args.limit, expected_shape=expected_state_size)
    except Exception as e:
        err_report = {"samples_count": 0, "model_load_note": "skip", "inference_ok": False, "inference_error": f"Failed to load samples: {e}"}
        save_report(err_report, args.out)
        print(f"[ERROR] {e}")
        return 2

    # "Canlı" modeli yükle
    loaded = try_load_model(args.model, args.model_class_import)
    model_obj = loaded.get("model", None)
    load_note = loaded.get("note", "")

    # Orijinal checkpoint'i de (config/scaler bilgisi için) yüklemeyi dene
    original_checkpoint_path = args.original_model_path
    checkpoint_dict = None
    if model_obj is not None and isinstance(model_obj, dict):
         checkpoint_dict = model_obj # Eğer 'inst_model.pth' yerine orijinal .pth verilirse
    elif original_checkpoint_path and os.path.exists(original_checkpoint_path):
        try:
            # Orijinal checkpoint'i 'dict' olarak yükle
            checkpoint_dict = safe_torch_load(original_checkpoint_path, weights_only=False) 
        except Exception:
            pass # Yüklenemezse sorun değil
            
    # Run inference best-effort
    result = None
    if model_obj is None:
        inference_err = "No model object available for inference"
    else:
        try:
            # --- YENİ "OTOMATİK ÖLÇEKLEME" ADIMI (ANALİZ J) ---
            # X_raw'ı en iyi ölçeklenmiş X_chosen'a dönüştür
            X_chosen, chosen_probs, scale_meta = try_scalings_and_choose(X_raw, model_obj, checkpoint=checkpoint_dict)
            report_extras["scaling_results"] = scale_meta
            load_note += f" | Scaling chosen: {scale_meta.get('chosen_method')}"
            # --- BLOK SONU ---
            
            # Ana analizi "seçilen" veriyle yap
            # 'try_scalings_and_choose' zaten olasılıkları hesapladı, tekrar hesaplamaya gerek yok.
            confs = chosen_probs.max(axis=1)
            preds = chosen_probs.argmax(axis=1)
            result = {"probs": chosen_probs, "confs": confs, "preds": preds}
            
        except Exception as e:
            inference_err = str(e)
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()

    report = build_report(X_raw, result, load_note, inference_err) # Raporu X_raw boyutuyla oluştur
    report.update(report_extras)
    save_report(report, args.out)

    # Always write diagnostics
    if args.out_dir:
        try:
            # probs_sample.npy dosyasını (eğer varsa) kaydet
            if report.get("inference_ok") and result is not None:
                _write_diagnostics(args.out_dir, result)
            else:
                os.makedirs(args.out_dir, exist_ok=True)
                _np.save(os.path.join(args.out_dir, "X_raw.npy"), X_raw) # Ham X'i kaydet
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
