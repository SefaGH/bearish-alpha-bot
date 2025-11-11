#!/usr/bin/env python3
"""
Summarize a probs_sample.npy file (numpy array shape (N, C)) into small JSON.

Usage:
  python scripts/summarize_probs_npy.py diagnostics/probs_sample.npy diagnostics/probs_summary.json
"""
import sys
import json
import numpy as np
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: summarize_probs_npy.py <input.npy> <output.json>")
    raise SystemExit(1)

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])

if not inp.exists():
    print(f"Input file not found: {inp}")
    sys.exit(1)

arr = np.load(inp)

summary = {}
summary['shape'] = list(arr.shape)
# per-class mean/std
summary['per_class_mean'] = arr.mean(axis=0).tolist()
summary['per_class_std'] = arr.std(axis=0).tolist()

# per-sample entropy
import math
def entropy_row(r):
    # klipslenmiş logaritma ile sayısal stabilite sağla
    r_clipped = np.clip(r, 1e-12, 1.0)
    return -float((r_clipped * np.log(r_clipped)).sum())

ents = [entropy_row(r) for r in arr]
summary['entropy_mean'] = float(np.mean(ents))
summary['entropy_std'] = float(np.std(ents))
# top-k probs stats
summary['sample_top1_mean'] = float(np.max(arr,axis=1).mean())
summary['sample_top1_std'] = float(np.max(arr,axis=1).std())

outp.write_text(json.dumps(summary, indent=2))
print(f"Wrote {outp}")
