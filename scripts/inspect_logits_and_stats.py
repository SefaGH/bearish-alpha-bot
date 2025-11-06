#!/usr/bin/env python3
"""
scripts/inspect_logits_and_stats.py

Compute feature stats and inspect model logits + softmax for first rows of CSV.
Writes diagnostics/logits_and_stats.json with:
 - sample_shape
 - feature_means, feature_stds
 - logits (raw q_network outputs) and softmaxed probs
 - optionally any config/training_history keys if present in checkpoint
"""
import os
import json
import traceback
import sys # sys.exit için eklendi

OUT = "diagnostics/logits_and_stats.json"
os.makedirs("diagnostics", exist_ok=True) # Raporlama için dizin oluştur

try:
    import torch
    import numpy as np
    import pandas as pd
    from torch.nn import functional as F
    # 'safe_torch_load'u import etmiyoruz, çünkü bu script'in 'weights_only=False'
    # kullanması ve kendi checkpoint'ini de (config için) okuması gerekiyor.
except Exception as e:
    with open(OUT, "w") as f:
        json.dump({"error": f"import failed: {e}", "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

# Çevre değişkenlerini al
MODEL_PATH = os.environ.get("MODEL_TO_USE", "diagnostics/inst_model.pth")
ORIGINAL_CKPT_PATH = os.environ.get("ORIGINAL_MODEL_PATH", "data/models/rl_agent_final.pth")
SAMPLES_PATH = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
N = int(os.environ.get("INSPECT_N", "10"))
res = {"model_path": MODEL_PATH, "samples_path": SAMPLES_PATH, "N": N}

# state_size'ı oku (veri kesmek için)
expected_state_size = None
try:
    with open("diagnostics/inferred_state_size.txt", "r") as f:
        expected_state_size = int(f.read().strip())
    print(f"Read expected state_size: {expected_state_size}")
except Exception as e:
    print(f"[WARN] Could not read 'inferred_state_size.txt': {e}")

# Örnekleri (sample) yükle
X = None
if not os.path.exists(SAMPLES_PATH):
    res["error"] = "samples file not found"
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (sample file not found)")
    sys.exit(0)

try:
    df = pd.read_csv(SAMPLES_PATH)
    # 'label' sütunlarını (ve benzerlerini) çıkar
    label_cols = [c for c in df.columns if c.lower() in ("label","target","y","class")]
    if label_cols:
        print(f"Dropping label/target columns: {label_cols}")
        df = df.drop(columns=label_cols)
    
    num = df.select_dtypes(include=[int,float]).to_numpy(dtype=float)
    X_full = num
    X = num[:N] # Sadece ilk N satırı al
    
    res["sample_shape_raw"] = list(X_full.shape)
    res["sample_head"] = df.head(3).to_dict(orient="records")
    
    # Özellik (feature) istatistikleri (tüm veri üzerinden)
    res["feature_means"] = np.mean(X_full, axis=0).tolist() if X_full.size else []
    res["feature_stds"] = np.std(X_full, axis=0).tolist() if X_full.size else []

    # Veriyi modele uyması için kes (slice)
    if expected_state_size is not None and X.shape[1] > expected_state_size:
        print(f"Slicing sample data from {X.shape[1]} to {expected_state_size} features.")
        X = X[:, :expected_state_size]
    res["sample_shape_final"] = list(X.shape)
    
except Exception as e:
    res["sample_load_error"] = str(e)
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (sample load error)")
    sys.exit(0)


# "Canlı" modeli yükle (weights_only=False ile)
try:
    obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    res["loaded_type"] = str(type(obj))
except Exception as e:
    res["load_error"] = str(e)
    res["load_trace"] = traceback.format_exc()
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (model load error)")
    sys.exit(0)

# Orijinal checkpoint'ten 'config' veya 'training_history' bulmaya çalış
try:
    ck = torch.load(ORIGINAL_CKPT_PATH, map_location="cpu")
    if isinstance(ck, dict):
        for k in ("config", "training_history", "epsilon"):
            if k in ck:
                v = ck[k]
                # Sadece basit tipleri kaydet
                if isinstance(v, (int, float, str, bool, dict, list)):
                    res.setdefault("checkpoint_info", {})[k] = v
                else:
                    res.setdefault("checkpoint_info", {})[k] = f"<{type(v).__name__}>"
except Exception as e:
    res.setdefault("checkpoint_info", {})["error"] = f"Could not load original checkpoint: {e}"


# Logit'leri ve olasılıkları (probs) 'q_network' üzerinden al
res["logits"] = []
res["probs"] = []
if hasattr(obj, "q_network"):
    try:
        for row in X:
            tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = obj.q_network(tensor) # Bu, ham logit'ler olmalı
            
            # Logit'leri (ham çıktı) kaydet
            try:
                logits = out.detach().cpu().numpy().tolist()
            except Exception:
                logits = str(type(out))
            
            # Softmax uygulanmış olasılıkları (probs) kaydet
            try:
                probs = F.softmax(out, dim=-1).detach().cpu().numpy().tolist()
            except Exception:
                probs = None
                
            res["logits"].append(logits)
            res["probs"].append(probs)
            
    except Exception as e:
        res["q_network_error"] = str(e)
        res["q_network_trace"] = traceback.format_exc()
else:
    res["q_network_error"] = "model has no attribute q_network"

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print(f"Wrote {OUT}")
