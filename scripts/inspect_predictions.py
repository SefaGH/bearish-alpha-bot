#!/usr/bin/env python3
"""
scripts/inspect_predictions.py

Load diagnostics/inst_model.pth (if exists) and sample CSV (SAMPLES_PATH env var).
Try several fallback ways to get model outputs for the first few samples:
 - model.predict_proba(X)
 - model.act(x)  (if present)
 - model.q_network(torch.tensor(x)) -> apply softmax
Write diagnostics/predictions_debug.json with attempts and shapes.
"""
import os
import json
import traceback

OUT = "diagnostics/predictions_debug.json"
os.makedirs("diagnostics", exist_ok=True)

try:
    import torch
    import numpy as np
    import pandas as pd
    import importlib
    from torch.nn import functional as F
    # 'safe_torch_load'u kullanarak 'weights_only' hatasını güvenle aşıyoruz
    from scripts.safe_torch_load import safe_torch_load 
except Exception as e:
    with open(OUT, "w") as f:
        json.dump({"error": f"import failed: {e}", "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

model_path = os.environ.get("INST_MODEL_PATH", "diagnostics/inst_model.pth")
samples_path = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
# safe_torch_load'un 'TradingRLAgent' sınıfını bulabilmesi için bu değişkene ihtiyacı var
model_class_import = os.environ.get("MODEL_CLASS_IMPORT", None) 
res = {"model_path": model_path, "samples_path": samples_path, "attempts": []}

def safe_tolist(t):
    try:
        return t.detach().cpu().numpy().tolist()
    except Exception:
        try:
            return list(t)
        except Exception:
            return str(type(t))

# load samples (first 10)
X = None
if os.path.exists(samples_path):
    try:
        df = pd.read_csv(samples_path)
        num = df.select_dtypes(include=[int,float]).to_numpy()
        if num.size == 0:
            num = df.to_numpy()
        X = num[:10]
        res["sample_shape"] = list(X.shape)
        res["sample_head"] = df.head(3).to_dict(orient="records")
    except Exception as e:
        res["sample_load_error"] = str(e)
else:
    res["sample_load_error"] = "samples file not found"

# load model
try:
    # torch.load yerine bizim 'akıllı' yükleyicimizi kullanıyoruz
    obj = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
    res["loaded_type"] = str(type(obj))
except Exception as e:
    res["load_error"] = str(e)
    res["load_trace"] = traceback.format_exc()
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)

if X is None:
    print(f"Cannot run predictions, sample data (X) is None.")
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    sys.exit(0)

# try predict_proba
try:
    if hasattr(obj, "predict_proba"):
        outp = obj.predict_proba(X)
        res["predict_proba"] = safe_tolist(outp)
    else:
        res["predict_proba"] = "not supported"
except Exception as e:
    res["predict_proba_error"] = str(e)
    res["predict_proba_trace"] = traceback.format_exc()

# try act (agent method)
try:
    if hasattr(obj, "act"):
        acts = []
        for row in X:
            try:
                # 'act' genellikle tek bir state (1D array) alır
                a = obj.act(row)
                acts.append(safe_tolist(a))
            except Exception as ex:
                acts.append({"error": str(ex)})
        res["act"] = acts
    else:
        res["act"] = "not supported"
except Exception as e:
    res["act_error"] = str(e)
    res["act_trace"] = traceback.format_exc()

# try q_network forward (apply softmax)
try:
    if hasattr(obj, "q_network"):
        qouts = []
        for row in X:
            try:
                # q_network genellikle (batch_size, state_size) bekler
                tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = obj.q_network(tensor)
                    probs = F.softmax(out, dim=-1)
                    qouts.append(safe_tolist(probs))
            except Exception as ex:
                qouts.append({"error": str(ex)})
        res["q_network"] = qouts
    else:
        res["q_network"] = "not supported"
except Exception as e:
    res["q_network_error"] = str(e)
    res["q_network_trace"] = traceback.format_exc()

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"Wrote {OUT}")
