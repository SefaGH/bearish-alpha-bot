#!/usr/bin/env python3
"""
Summarize checkpoint training_history if present. Writes diagnostics/training_history_summary.json
"""
import os
import json
import traceback
import sys # sys.exit için eklendi

OUT = "diagnostics/training_history_summary.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
    from scripts.safe_torch_load import safe_torch_load # 'safe_torch_load'u kullan
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

# Orijinal checkpoint'i (ağırlıklar/dict) oku
cp = os.environ.get("ORIGINAL_MODEL_PATH", "data/models/rl_agent_final.pth")
model_class_import = os.environ.get("MODEL_CLASS_IMPORT", None) # safe_torch_load için gerekli

if not os.path.exists(cp):
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": "checkpoint not found: " + cp}, f, indent=2)
    print(f"Wrote {OUT} (checkpoint not found)")
    sys.exit(0)

try:
    ck = safe_torch_load(cp, model_class_import=model_class_import)
except Exception as e:
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({"error": f"safe_torch_load failed: {e}", "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)
    
res = {"checkpoint_keys": list(ck.keys()) if isinstance(ck, dict) else "not-dict"}
if isinstance(ck, dict) and "training_history" in ck:
    th = ck["training_history"]
    out = {}
    for k in ("losses", "train_loss", "val_loss", "loss_history"):
        if k in th:
            arr = th[k]
            try:
                arr = np.asarray(arr)
                out[k] = {"len": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "last": float(arr[-1])}
            except Exception:
                out[k] = "present but unreadable"
                
    if not out and isinstance(th, (list, tuple)) and len(th) > 0 and isinstance(th[0], dict):
        losses = []
        for e in th:
            if "loss" in e:
                losses.append(e["loss"])
        if losses:
            arr = np.asarray(losses)
            out["loss_from_entries"] = {"len": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "last": float(arr[-1])}
            
    res["training_history_summary"] = out
else:
    res["training_history_summary"] = "no training_history key found"

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2, default=str) # default=str eklendi
print("Wrote", OUT)
