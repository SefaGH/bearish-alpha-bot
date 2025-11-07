#!/usr/bin/env python3
"""
Summarize a large diagnostics/model_inst_attempt.json into a small text summary.

Usage:
  python scripts/summarize_inst_attempt.py
"""
import json
import textwrap
import os
import sys

F = "diagnostics/model_inst_attempt.json"
OUT_F = "diagnostics/model_inst_summary.txt"

os.makedirs("diagnostics", exist_ok=True) # Dizin yoksa oluştur

if not os.path.exists(F):
    print(f"File not found, skipping summary: {F}")
    with open(OUT_F, "w") as f:
        f.write(f"File not found: {F}")
    sys.exit(0)

try:
    with open(F, 'r') as f:
        o = json.load(f)
except Exception as e:
    print(f"Failed to load JSON from {F}: {e}")
    with open(OUT_F, "w") as f:
        f.write(f"Failed to load JSON from {F}: {e}")
    sys.exit(0)

def short(x, n=1000):
    if x is None: return None
    s = str(x)
    return s if len(s) <= n else s[:n] + "...[truncated]"

# Analizde istenen anahtar anahtarlar
keys_to_show = [
    "import_error", "import_traceback",
    "instantiation_kwargs", "inferred", "inferred_state_size", "inferred_action_size",
    "instantiate_error", "instantiate_traceback",
    "load_error", "load_traceback",
    "model_inst_load_errors", "inst_saved", "loaded"
]

out = []
out.append("ALL_KEYS: " + ", ".join(sorted(o.keys())))

for k in keys_to_show:
    if k in o:
        out.append(f"--- {k} ---")
        v = o[k]
        if isinstance(v, dict):
            # Küçük sözlükleri (config/kwargs gibi) güzel bas
            try:
                pretty_v = json.dumps(v, indent=2)
                out.append(short(pretty_v, 1000))
            except Exception:
                 # Serileştirilemeyen bir şey varsa (örn. tensor)
                 for kk, vv in list(v.items())[:40]:
                    out.append(f"  {kk}: {short(vv, 400)}")
                 if len(v) > 40:
                    out.append(f"  ... {len(v) - 40} more keys ...")
        else:
            out.append(short(v, 1000))

# 'loaded', 'inst_saved' ve 'inferred' zaten keys_to_show'da var veya (inferred)
# instantiation_kwargs içinde yer alıyor.

txt = "\n".join(out)
with open(OUT_F, "w") as f:
    f.write(txt)

print(f"Wrote {OUT_F}")
# Loglara da bas
print(f"\n--- Content of {OUT_F} ---")
print(txt)
print("------------------------------------------")
