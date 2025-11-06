#!/usr/bin/env python3
"""
scripts/inspect_logits_and_stats.py

Compute feature stats and inspect model logits + softmax for first rows of CSV.
Writes diagnostics/logits_and_stats.json with:
 - sample_shape
 - feature_means, feature_stds
 - logits (raw q_network outputs) and softmaxed probs
 - optionally any config/training_history keys if present in checkpoint
 
YENİ: Artık veriyi (X - mean) / std kullanarak Z-score normalizasyonu yapar.
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
X_scaled = None
X_raw = None
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
    
    num_df = df.select_dtypes(include=[int,float])
    X_full = num_df.to_numpy(dtype=float)
    X_raw = X_full[:N] # Sadece ilk N satırı al
    
    res["sample_shape_raw"] = list(X_full.shape)
    res["sample_head"] = df.head(3).to_dict(orient="records")
    
    # Özellik (feature) istatistikleri (tüm veri üzerinden)
    means = np.mean(X_full, axis=0)
    stds = np.std(X_full, axis=0)
    # 0'a bölme hatasını önle (std 0 ise 1 yap)
    stds[stds == 0] = 1.0 
    
    res["feature_means"] = means.tolist() if X_full.size else []
    res["feature_stds"] = stds.tolist() if X_full.size else []

    # Veriyi modele uyması için kes (slice)
    if expected_state_size is not None and X_raw.shape[1] > expected_state_size:
        print(f"Slicing sample data from {X_raw.shape[1]} to {expected_state_size} features.")
        X_raw = X_raw[:, :expected_state_size]
        means = means[:expected_state_size]
        stds = stds[:expected_state_size]
    res["sample_shape_final"] = list(X_raw.shape)
    
    # --- YENİ ADIM: Veriyi Z-Score ile Normalize Et ---
    print("Applying Z-score normalization (X - mean) / std to data...")
    X_scaled = (X_raw - means) / stds
    # --- BLOK SONU ---
    
except Exception as e:
    res["sample_load_error"] = str(e)
    traceback.print_exc()
    with open(OUT, "w") as f: json.dump(res, f, indent=2)
    print(f"Wrote {OUT} (sample load error)")
    sys.exit(0)


# "Canlı" modeli yükle (weights_only=False ile)
try:
    # Not: 'safe_torch_load'u burada kullanmıyoruz çünkü bu script'in 'weights_only=False'
    # kullanması ve kendi checkpoint'ini de (config için) okuması gerekiyor.
    # Bu script zaten 'inst_model.pth'yi yüklüyor, bu yüzden 'model_class_import'a gerek yok.
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
        # Analiz H'den gelen daha iyi anahtar listesi:
        for k in ("config", "training_history", "epsilon", "scaler_mean", "scaler_std"):
            if k in ck:
                v = ck[k]
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
        # --- GÜNCELLEME: Ham (X_raw) yerine ölçeklenmiş (X_scaled) veriyi kullan ---
        for row in X_scaled: 
            tensor = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = obj.q_network(tensor) # Bu, ham logit'ler olmalı
            
            try:
                logits = out.detach().cpu().numpy().tolist()
            except Exception:
                logits = str(type(out))
            
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
