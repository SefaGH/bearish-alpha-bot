#!/usr/bin/env python3
"""
Run model on multiple input variants and summarize logits/probs/entropy.
Writes diagnostics/input_response_summary.json
"""
import os
import json
import traceback
import sys # sys.exit için eklendi

OUT="diagnostics/input_response_summary.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
    import pandas as pd
    from torch.nn import functional as F
    from scripts.safe_torch_load import safe_torch_load # 'safe_torch_load'u kullan
except Exception as e:
    with open(OUT,"w") as f:
        json.dump({"error":f"import failed: {e}","trace":traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

def entropy_rows(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)

model_path = "diagnostics/inst_model.pth"
model_class_import = os.environ.get("MODEL_CLASS_IMPORT", None) # safe_torch_load için gerekli

if not os.path.exists(model_path):
    with open(OUT,"w") as f:
        json.dump({"error":"inst_model.pth not found"}, indent=2)
    print(f"Wrote {OUT} (file not found)")
    sys.exit(0)

try:
    model = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
    model.eval()
except Exception as e:
    with open(OUT,"w") as f:
        json.dump({"error":f"safe_torch_load failed: {e}","trace":traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)


# state_size'ı oku (veri kesmek için)
expected_state_size = None
try:
    with open("diagnostics/inferred_state_size.txt", "r") as f:
        expected_state_size = int(f.read().strip())
    print(f"Read expected state_size: {expected_state_size}")
except Exception as e:
    print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")


# load samples
csv = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
res = {"model_type": str(type(model)), "variants": {}}
X_raw = None

if os.path.exists(csv):
    df = pd.read_csv(csv)
    # drop label-like columns
    label_cols = [c for c in df.columns if c.lower() in ("label","target","y","class")]
    if label_cols:
        print(f"Dropping label/target columns: {label_cols}")
        df = df.drop(columns=label_cols)
    X_raw = df.select_dtypes(include=[float,int]).to_numpy(dtype=float)[:10] # İlk 10 satırı al
else:
    # state_size'ı 'model' objesinden veya tahmin edilenden al
    state_size = getattr(model, "state_size", expected_state_size if expected_state_size else 42)
    X_raw = np.random.randn(5, state_size)
    print(f"Sample CSV not found. Using random data shape (5, {state_size})")

# --- VERİYİ DOĞRU BOYUTA KES ---
if expected_state_size is not None:
    print(f"Data shape is {X_raw.shape}, model expects {expected_state_size}.")
    if X_raw.shape[1] > expected_state_size:
        print(f"Slicing raw data from {X_raw.shape[1]} to {expected_state_size} features.")
        X_raw = X_raw[:, :expected_state_size]
    res["final_data_shape"] = list(X_raw.shape)
# --- BLOK SONU ---


variants = {}
variants["raw"] = X_raw
# zscore sample
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0) + 1e-12
variants["zscore_sample"] = (X_raw - means) / stds
# minmax
mn = X_raw.min(axis=0)
mx = X_raw.max(axis=0)
denom_mm = (mx - mn) + 1e-12
denom_mm[denom_mm < 1e-12] = 1.0 # 0'a bölmeyi engelle
variants["minmax_sample"] = (X_raw - mn) / denom_mm
# maxabs
maxabs = np.max(np.abs(X_raw), axis=0)
maxabs[maxabs==0]=1.0
variants["maxabs_sample"] = X_raw / maxabs
# random normal with same shape and scale as sample std/mean
variants["rand_normal_scaled"] = np.random.randn(*X_raw.shape) * (stds.reshape(1,-1)) + means.reshape(1,-1)
variants["rand_uniform"] = np.random.uniform(low=-1.0, high=1.0, size=X_raw.shape)

def run_variant(name, X):
    out = {"shape": list(X.shape)}
    try:
        xt = torch.tensor(X, dtype=torch.float32)
        logits = None
        # prefer q_network
        if hasattr(model, "q_network"):
            with torch.no_grad():
                logits = model.q_network(xt).detach().cpu().numpy()
        elif callable(model): # Fallback (eğer modelin kendisi nn.Module ise)
            with torch.no_grad():
                logits = model(xt).detach().cpu().numpy()
        else:
            raise RuntimeError("Model is not callable and has no q_network attribute.")

        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        out["logits_mean"] = logits.mean(axis=0).tolist()
        out["logits_std"] = logits.std(axis=0).tolist()
        out["probs_mean"] = probs.mean(axis=0).tolist()
        out["probs_std"] = probs.std(axis=0).tolist()
        out["entropy_mean"] = float(entropy_rows(probs).mean())
    except Exception as e:
        out["error"] = str(e)
        out["trace"] = traceback.format_exc()[:2000]
    return out

for k, X_variant in variants.items():
    if X_variant is not None and X_variant.size > 0:
        res["variants"][k] = run_variant(k, X_variant)
    else:
        res["variants"][k] = {"error": "Input data X was empty or None."}


with open(OUT,"w") as f:
    json.dump(res, indent=2, default=str) # default=str (numpy türleri için)
print(f"Wrote {OUT}")
