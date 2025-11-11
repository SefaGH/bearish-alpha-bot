#!/usr/bin/env python3
"""
scripts/inspect_model.py

Loads a torch checkpoint from the path in MODEL_PATH env var and writes
diagnostics/model_load.json with a summary (type, top keys, shapes).
Intended to be safe to run inside CI.
"""
import os
import json
import traceback
import sys  # sys.exit(0) için eklendi

# Çıktı dosyasını 'diagnostics' klasörüne yaz
OUT_PATH = "diagnostics/model_load.json"
MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/rl_agent_final.pth")

# Raporlama için 'diagnostics' dizininin var olduğundan emin olun
os.makedirs("diagnostics", exist_ok=True)

out = {"path": MODEL_PATH}
try:
    import torch
except Exception as e:
    out["error"] = f"import torch failed: {e}"
    out["traceback"] = traceback.format_exc()
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH} (torch import error)")
    sys.exit(0) # Hata olsa bile CI adımını kırmamak için 0 ile çık

try:
    obj = torch.load(MODEL_PATH, map_location="cpu")
except Exception as e:
    out["error"] = f"torch.load failed: {e}"
    out["traceback"] = traceback.format_exc()
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH} (torch.load error)")
    sys.exit(0) # Hata olsa bile 0 ile çık

out["type"] = str(type(obj))
if isinstance(obj, dict):
    keys = list(obj.keys())
    out["n_keys"] = len(keys)
    out["keys_sample"] = keys[:200]
    sample_info = {}
    for k in keys[:200]:
        try:
            v = obj[k]
            t = type(v).__name__
            # shape/size almayı dene
            shape = None
            if hasattr(v, "size"):
                try:
                    shape = list(v.size())
                except Exception:
                    shape = None
            elif hasattr(v, "shape"):
                try:
                    shape = list(v.shape)
                except Exception:
                    shape = None
            sample_info[k] = {"type": t, "shape": shape}
        except Exception as ex:
            sample_info[k] = {"error": str(ex)}
    out["sample_values_info"] = sample_info
else:
    try:
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        out["attrs_sample"] = attrs[:400]
    except Exception as e:
        out["attrs_error"] = str(e)
    try:
        out["repr"] = repr(obj)[:2000]
    except Exception:
        out["repr_error"] = "repr failed"

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT_PATH}")
