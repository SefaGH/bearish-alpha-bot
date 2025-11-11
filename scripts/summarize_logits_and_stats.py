#!/usr/bin/env python3
"""
Summarize a large diagnostics/logits_and_stats.json into a small JSON report.

Usage:
  python scripts/summarize_logits_and_stats.py diagnostics/logits_and_stats.json diagnostics/logits_summary.json
"""
import sys
import json
import math
import statistics
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: summarize_logits_and_stats.py <input.json> <output.json>")
    raise SystemExit(1)

inp = Path(sys.argv[1])
outp = Path(sys.argv[2])

if not inp.exists():
    print(f"Input file not found: {inp}")
    sys.exit(1)

data = json.loads(inp.read_text())

summary = {}
# basic infos
summary['model_path'] = data.get('model_path')
summary['samples_path'] = data.get('samples_path')
summary['sample_shape_raw'] = data.get('sample_shape_raw')
# checkpoint info keys
summary['checkpoint_info_keys'] = list(data.get('checkpoint_info', {}).keys()) if data.get('checkpoint_info') else []

# feature means/stds: report min/max/median and indices with near-zero std
fmeans = data.get('feature_means') or []
fstds = data.get('feature_stds') or []
if fmeans and fstds:
    summary['feature_count'] = len(fmeans)
    summary['feature_mean_min'] = min(fmeans)
    summary['feature_mean_max'] = max(fmeans)
    summary['feature_std_min'] = min(fstds)
    summary['feature_std_max'] = max(fstds)
    # find features with near-zero std
    near_zero = [i for i,s in enumerate(fstds) if (s is not None and float(s) < 1e-6)]
    summary['features_near_zero_std_count'] = len(near_zero)
    summary['features_near_zero_std_indices'] = near_zero[:20]
else:
    summary['feature_count'] = None

# logits/probs: flatten and compute per-class stats if available
logits = data.get('logits') or []
probs = data.get('probs') or []

def per_class_stats(arrs):
    # arrs: list of per-sample arrays (possibly nested)
    import math
    flat = []
    for a in arrs:
        # try to normalize shape: find numeric entries inside
        if isinstance(a, list) and len(a):
            # handle nested lists
            if isinstance(a[0], list):
                # e.g. [[v1,v2,v3]]
                vals = a[0]
            else:
                vals = a
            try:
                flat.append([float(x) for x in vals])
            except (TypeError, ValueError):
                continue # Skip non-numeric entries
    if not flat:
        return None
    
    try:
        # transpose to per-class lists
        per_class = list(zip(*flat))
        stats = []
        for pc in per_class:
            stats.append({
                'mean': float(statistics.mean(pc)),
                'stdev': float(statistics.pstdev(pc)) if len(pc)>1 else 0.0,
                'min': float(min(pc)),
                'max': float(max(pc))
            })
        return stats
    except Exception:
        return None # Transpose failed (e.g., jagged lists)

summary['logits_per_class_stats'] = per_class_stats(logits)
summary['probs_per_class_stats'] = per_class_stats(probs)

# entropy / kl vs uniform on probs
def entropy(p):
    return -sum((x and x>0) and x*math.log(x) or 0.0 for x in p)
def kl_vs_uniform(p):
    k = len(p)
    if k == 0: return 0.0
    u = 1.0/k
    return sum((x and x>0) and x*math.log((x/u)) or 0.0 for x in p)

entropies = []
kls = []
for p in probs:
    # handle nested list
    if isinstance(p, list):
        arr = p[0] if p and isinstance(p[0], list) else p
        try:
            arr2 = [float(x) for x in arr]
            if not arr2: continue
            entropies.append(entropy(arr2))
            kls.append(kl_vs_uniform(arr2))
        except Exception:
            pass
summary['probs_sample_count'] = len(entropies)
if entropies:
    summary['probs_entropy_mean'] = float(sum(entropies)/len(entropies))
    summary['probs_entropy_min'] = float(min(entropies))
    summary['probs_entropy_max'] = float(max(entropies))
    summary['probs_kl_vs_uniform_mean'] = float(sum(kls)/len(kls))
else:
    summary['probs_entropy_mean'] = None

# small example slices for manual inspection (first 5)
summary['example_logits'] = logits[:5]
summary['example_probs'] = probs[:5]

outp.write_text(json.dumps(summary, indent=2))
print(f"Wrote {outp}")
