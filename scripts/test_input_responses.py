#!/usr/bin/env python3
"""
Run model on multiple input variants and summarize logits/probs/entropy.
Writes diagnostics/input_response_summary.json

This version is robust to agent wrappers that don't implement .eval()
and will call .eval() on the underlying q_network module when present.
Ayrıca "inferred_state_size.txt" okur ve veriyi doğru boyuta keser.
"""
import os
import json
import traceback
import sys # sys.exit için eklendi

OUT = "diagnostics/input_response_summary.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
    import pandas as pd
    from torch.nn import functional as F
    from scripts.safe_torch_load import safe_torch_load # 'safe_torch_load'u kullanıyoruz
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

def entropy_rows(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)

model_path = os.environ.get("MODEL_TO_USE", "diagnostics/inst_model.pth")
model_class_import = os.environ.get("MODEL_CLASS_IMPORT", None) # safe_torch_load için gerekli

if not os.path.exists(model_path):
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": "inst_model.pth not found"}, f, indent=2)
    print(f"Wrote {OUT} (model not found)")
    sys.exit(0)

try:
    model = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": "safe_torch_load failed: " + str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)

# Put underlying module in eval mode if available
def set_eval_on_module(obj):
    try:
        if hasattr(obj, "eval") and callable(getattr(obj, "eval")):
            obj.eval()
            return True
        if hasattr(obj, "q_network"):
            q = getattr(obj, "q_network")
            if hasattr(q, "eval") and callable(q.eval):
                q.eval()
                return True
    except Exception:
        return False
    return False

set_eval_on_module(model)

# Resolve forward function
def model_forward_logits(x_tensor):
    # x_tensor: torch.Tensor (N, D)
    if hasattr(model, "q_network"):
        with torch.no_grad():
            return model.q_network(x_tensor).detach().cpu().numpy()
    try:
        with torch.no_grad():
            out = model(x_tensor)
            return out.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(model, "act"):
        outs = []
        action_size = getattr(model, "action_size", 3)
        for row in x_tensor.detach().cpu().numpy():
            try:
                a = model.act(row)
                if isinstance(a, int):
                    v = np.zeros(action_size)
                    v[a] = 1.0
                    outs.append(v)
                else:
                    outs.append(np.asarray(a))
            except Exception:
                outs.append(np.zeros(action_size))
        return np.vstack(outs)
    raise RuntimeError("No usable forward interface found on model (q_network, callable, act)")

# state_size'ı oku (veri kesmek için)
expected_state_size = None
try:
    with open("diagnostics/inferred_state_size.txt", 'r', encoding='utf-8') as f:
        expected_state_size = int(f.read().strip())
    print(f"Read expected state_size: {expected_state_size}")
except Exception as e:
    print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")

# load samples
csv = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
X_raw = None
if os.path.exists(csv):
    df = pd.read_csv(csv)
    label_cols = [c for c in df.columns if c.lower() in ("label", "target", "y", "class")]
    if label_cols:
        df = df.drop(columns=label_cols)
    X_raw = df.select_dtypes(include=[float, int]).to_numpy(dtype=float)[:10]
    
    # Veriyi kes (slice)
    if expected_state_size is not None and X_raw.shape[1] > expected_state_size:
        print(f"Slicing data from {X_raw.shape[1]} to {expected_state_size} features.")
        X_raw = X_raw[:, :expected_state_size]
else:
    # Fallback: Rastgele veri oluştur
    state_size_to_use = expected_state_size if expected_state_size else 42
    print(f"CSV not found. Generating random data with shape (5, {state_size_to_use}).")
    X_raw = np.random.randn(5, state_size_to_use)

if X_raw is None or X_raw.size == 0:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": "No sample data (X_raw) could be loaded or generated."}, f, indent=2)
    print(f"Wrote {OUT} (no sample data)")
    sys.exit(0)

variants = {}
variants["raw"] = X_raw
means = X_raw.mean(axis=0)
stds = X_raw.std(axis=0) + 1e-12
variants["zscore_sample"] = (X_raw - means) / stds
mn = X_raw.min(axis=0); mx = X_raw.max(axis=0)
variants["minmax_sample"] = (X_raw - mn) / (mx - mn + 1e-12)
maxabs = np.max(np.abs(X_raw), axis=0); maxabs[maxabs == 0] = 1.0
variants["maxabs_sample"] = X_raw / maxabs
variants["rand_normal_scaled"] = np.random.randn(*X_raw.shape) * (stds.reshape(1, -1)) + means.reshape(1, -1)
variants["rand_uniform"] = np.random.uniform(low=-1.0, high=1.0, size=X_raw.shape)

res = {"model_type": str(type(model)), "variants": {}}

for k, X in variants.items():
    entry = {"shape": list(X.shape)}
    try:
        xt = torch.tensor(X, dtype=torch.float32)
        logits = model_forward_logits(xt)  # numpy
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        entry["logits_mean"] = logits.mean(axis=0).tolist()
        entry["logits_std"] = logits.std(axis=0).tolist()
        entry["probs_mean"] = probs.mean(axis=0).tolist()
        entry["probs_std"] = probs.std(axis=0).tolist()
        entry["entropy_mean"] = float(entropy_rows(probs).mean())
    except Exception as e:
        entry["error"] = str(e)
        entry["trace"] = traceback.format_exc()[:2000]
    res["variants"][k] = entry

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, default=str)
print("Wrote", OUT)
