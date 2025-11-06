#!/usr/bin/env python3
"""
Dump per-parameter statistics for the instantiated model at diagnostics/inst_model.pth
Writes diagnostics/weight_stats.json
"""
import json
import traceback
import os
import sys # sys.exit için eklendi

OUT="diagnostics/weight_stats.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
except Exception as e:
    with open(OUT,"w") as f:
        json.dump({"error":f"import torch/numpy failed: {e}","trace":traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

p="diagnostics/inst_model.pth"
if not os.path.exists(p):
    with open(OUT,"w") as f:
        json.dump({"error":"file not found: "+p}, indent=2)
    print(f"Wrote {OUT} (file not found)")
    sys.exit(0)

try:
    # 'safe_torch_load'u burada kullanmıyoruz, çünkü bu script'in 'weights_only=False'
    # kullanması gerekiyor ve 'verify_model.py' zaten çalıştığını doğruladı.
    m = torch.load(p, map_location="cpu", weights_only=False)
except Exception as e:
    with open(OUT,"w") as f:
        json.dump({"error":f"torch.load failed: {e}","trace":traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)


# if the saved object is a wrapper, try to get q_network attribute
if hasattr(m, "q_network"):
    net = m.q_network
else:
    # try common names
    net = getattr(m, "model", m) # Eğer q_network yoksa modelin kendisini 'net' olarak kabul et

summary = {"model_type": str(type(m)), "q_network_type": str(type(net))}
params = []

if not hasattr(net, "named_parameters"):
    with open(OUT,"w") as f:
        json.dump({"error":f"'net' object (type {type(net)}) has no named_parameters attribute"}, indent=2)
    print(f"Wrote {OUT} (net has no named_parameters)")
    sys.exit(0)

for name, param in net.named_parameters():
    arr = param.detach().cpu().numpy()
    params.append({
        "name": name,
        "shape": list(arr.shape),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "abs_mean": float(abs(arr).mean())
    })

# identify final linear-like layers by name
final_candidates = [p for p in params if any(s in p["name"].lower() for s in ("out","head","fc","value","adv","final","action"))]
summary["param_count"] = len(params)
summary["params"] = params
summary["final_layer_candidates"] = final_candidates[:10] # Sadece ilk 10 adayı göster

with open(OUT,"w") as f:
    json.dump(summary, indent=2)
print(f"Wrote {OUT}")
