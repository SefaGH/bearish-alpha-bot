#!/usr/bin/env python3
"""
Dump per-parameter statistics for the instantiated model at diagnostics/inst_model.pth
Writes diagnostics/weight_stats.json
Handles agent wrappers that expose a .q_network attribute.
"""
import json
import traceback
import os
import sys  # sys.exit için eklendi

OUT = "diagnostics/weight_stats.json"
os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

try:
    import torch
    import numpy as np
except Exception as e:
    with open(OUT, "w") as f:
        json.dump({"error": str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (import error)")
    sys.exit(0)

p = "diagnostics/inst_model.pth"
if not os.path.exists(p):
    with open(OUT, "w") as f:
        json.dump({"error": "file not found: " + p}, f, indent=2)
    print(f"Wrote {OUT} (file not found)")
    sys.exit(0)

# Load full object (we saved the instance)
try:
    # 'safe_torch_load'u burada KULLANMIYORUZ, çünkü 'weights_only=False'
    # bu script'in (verify_model'in aksine) varsayılan beklentisidir.
    # verify_model'da bu hatayı zaten aştık.
    obj = torch.load(p, map_location="cpu", weights_only=False)
except Exception as e:
    with open(OUT, "w") as f:
        json.dump({"error": "torch.load failed: " + str(e), "trace": traceback.format_exc()}, f, indent=2)
    print(f"Wrote {OUT} (load error)")
    sys.exit(0)

# Resolve the network module to inspect
net = None
candidates = []
if hasattr(obj, "q_network"):
    net = getattr(obj, "q_network")
    candidates.append("q_network")
# common fallbacks
for attr in ("model", "net", "network", "policy", "actor", "critic"):
    if net is None and hasattr(obj, attr):
        cand = getattr(obj, attr)
        # take it only if it's a torch.nn.Module-like object (has named_parameters)
        if hasattr(cand, "named_parameters"):
            net = cand
            candidates.append(attr)
            break

# if the loaded object itself is a Module
try:
    import torch.nn as _nn
    if net is None and isinstance(obj, _nn.Module):
        net = obj
        candidates.append("self_module")
except Exception:
    pass

if net is None:
    with open(OUT, "w") as f:
        json.dump({"error": "Could not find a torch module (q_network/model/net) on the loaded object", "available_attrs": [a for a in dir(obj) if not a.startswith('_')][:200]}, indent=2)
    print(f"Wrote {OUT} (module not found)")
    sys.exit(0)

summary = {"model_type": str(type(obj)), "net_attr_candidates_used": candidates}
params = []
try:
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
except Exception as e:
    summary["error_listing_params"] = str(e)
    params = [] # Parametreleri listeleme başarısız oldu

# Heuristic: find final linear layers (named like fc, out, head, final, value, adv)
final_candidates = [p for p in params if any(s in p["name"].lower() for s in ("out", "head", "fc", "final", "value", "adv", "action", "bias"))]
# Also consider last N params as candidates
if not final_candidates and params:
    final_candidates = params[-6:]

summary["param_count"] = len(params)
summary["params_sample_count"] = min(40, len(params))
summary["params_sample"] = params[:summary["params_sample_count"]]
summary["final_layer_candidates"] = final_candidates[:20]
# quick checks for collapsed weights
collapsed = [p["name"] for p in params if (p["abs_mean"] < 1e-6 and p["std"] < 1e-6)]
summary["collapsed_param_count"] = len(collapsed)
summary["collapsed_params"] = collapsed[:50]

with open(OUT, "w") as f:
    json.dump(summary, f, indent=2, default=str) # default=str eklendi
print("Wrote", OUT)
