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
import sys # sys.exit için eklendi

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
    with open(OUT, 'w', encoding='utf-8') as f:
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

# --- YENİ EKLENEN ADIM ---
# Modelin beklediği state_size'ı (42) dosyadan oku
expected_state_size = None
try:
    with open("diagnostics/inferred_state_size.txt", 'r', encoding='utf-8') as f:
        expected_state_size = int(f.read().strip())
    print(f"Read expected state_size from file: {expected_state_size}")
except Exception as e:
    print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")
# --- BLOK SONU ---

# load samples (first 10)
X = None
if os.path.exists(samples_path):
    try:
        df = pd.read_csv(samples_path)
        
        # 'label' sütununu (ve benzerlerini) çıkar
        label_cols_found = []
        for drop_col in ("label", "target", "y", "timestamp", "time", "datetime"):
            if drop_col in df.columns: # Orijinal df'ten çıkar
                label_cols_found.append(drop_col)
                
        if label_cols_found:
            print(f"Dropping non-feature columns: {label_cols_found}")
            df = df.drop(columns=label_cols_found, errors='ignore')

        num = df.select_dtypes(include=[int,float]).to_numpy()
        if num.size == 0:
            print("[WARN] No numeric columns found after dropping labels. Using all columns.")
            num = df.to_numpy()
            
        X = num[:10]
        res["sample_shape_raw"] = list(X.shape)
        res["sample_head"] = df.head(3).to_dict(orient="records")
        
        # --- YENİ EKLENEN VERİ KESME (SLICING) BLOĞU ---
        if expected_state_size is not None:
            print(f"Data shape is {X.shape}, model expects {expected_state_size}.")
            if X.shape[1] > expected_state_size:
                print(f"Slicing data from {X.shape[1]} features to {expected_state_size} features.")
                X = X[:, :expected_state_size]
            elif X.shape[1] < expected_state_size:
                print(f"[WARN] Data shape ({X.shape[1]}) is smaller than model expected shape ({expected_state_size})!")
        res["sample_shape_final"] = list(X.shape)
        # --- BLOK SONU ---

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
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)

if X is None:
    print(f"Cannot run predictions, sample data (X) is None.")
    with open(OUT, 'w', encoding='utf-8') as f:
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

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)
print(f"Wrote {OUT}")
