#!/usr/bin/env python3
"""
Probe model sensitivity to input scale multipliers.
Writes diagnostics/scale_probe.json with stats for each multiplier.
"""
import os
import json
import traceback
import sys  # sys.exit için eklendi

OUT = "diagnostics/scale_probe.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
    import pandas as pd
    from torch.nn.functional import softmax
    from scripts.safe_torch_load import safe_torch_load # 'safe_torch_load'u kullan
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

MODEL_PATH = os.environ.get("MODEL_TO_USE", "diagnostics/inst_model.pth")
CSV = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
MODEL_CLASS_IMPORT = os.environ.get("MODEL_CLASS_IMPORT", None)
multipliers = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

res = {"model_path": MODEL_PATH, "csv": CSV, "results": {}}

if not os.path.exists(MODEL_PATH):
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": f"model not found: {MODEL_PATH}"}, f, indent=2)
    print(f"Wrote {OUT} (model not found)")
    sys.exit(0)

try:
    m = safe_torch_load(MODEL_PATH, model_class_import=MODEL_CLASS_IMPORT, map_location="cpu")
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": f"safe_torch_load failed: {e}", "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)

# resolve network
net = getattr(m, "q_network", m)

# state_size'ı oku (veri kesmek için)
expected_state_size = None
try:
    with open("diagnostics/inferred_state_size.txt", 'r', encoding='utf-8') as f:
        expected_state_size = int(f.read().strip())
    print(f"Read expected state_size: {expected_state_size}")
except Exception as e:
    print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")


# load X
if os.path.exists(CSV):
    df = pd.read_csv(CSV)
    for c in ["label", "target", "y", "class"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    X0 = df.select_dtypes(include=[float, int]).to_numpy(dtype=float)[:10]
    
    # Veriyi kes (slice)
    if expected_state_size is not None and X0.shape[1] > expected_state_size:
        print(f"Slicing data from {X0.shape[1]} to {expected_state_size} features.")
        X0 = X0[:, :expected_state_size]
else:
    state_size_to_use = expected_state_size if expected_state_size else 42
    print(f"CSV not found. Generating random data with shape (5, {state_size_to_use}).")
    X0 = np.random.randn(5, state_size_to_use)

if X0 is None or X0.size == 0:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": "No sample data (X0) could be loaded or generated."}, f, indent=2)
    print(f"Wrote {OUT} (no sample data)")
    sys.exit(0)


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


for k in multipliers:
    try:
        X = X0 * float(k)
        xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = net(xt).detach().cpu().numpy()
        probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
        ent = entropy(probs)
        res["results"][str(k)] = {
            "shape": list(X.shape),
            "logits_mean": logits.mean(axis=0).tolist(),
            "logits_std": logits.std(axis=0).tolist(),
            "probs_mean": probs.mean(axis=0).tolist(),
            "probs_std": probs.std(axis=0).tolist(),
            "entropy_mean": float(ent.mean()),
            "entropy_std": float(ent.std())
        }
    except Exception as e:
        res["results"][str(k)] = {"error": str(e), "trace": traceback.format_exc()[:2000]}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, default=str) # default=str eklendi
print("Wrote", OUT)
