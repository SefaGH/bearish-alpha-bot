#!/usr/bin/env python3
"""
scripts/feature_engineer_and_run.py

1) Load OHLCV CSV (open, high, low, close, volume, optional timestamp)
2) Run FeatureEngineeringPipeline.extract_features -> align_and_finalize_features
3) Save diagnostics/feature_engineered_samples.csv
4) Load saved scaler (data/models/regime/scaler.pkl) and apply transform
5) Run diagnostics/inst_model.pth (q_network) on first rows and save diagnostics/scaler_apply_result.json
6) Save diagnostics/feature_stats.json with NaN/inf checks and basic stats
"""
import os, sys, json, traceback
OUT_DIR = "diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python scripts/feature_engineer_and_run.py path/to/ohlcv.csv")
    raise SystemExit(1)

csv_path = sys.argv[1]

try:
    import pandas as pd, numpy as np, joblib
    from src.ml.feature_engineering import FeatureEngineeringPipeline
    import torch
    from torch.nn.functional import softmax
except Exception as e:
    print("Imports failed:", e)
    print(traceback.format_exc())
    raise SystemExit(1)

# Read OHLCV
if not os.path.exists(csv_path):
    print("CSV not found:", csv_path)
    raise SystemExit(1)

df = pd.read_csv(csv_path)
# Basic normalization of column names
cols_lower = {c.lower(): c for c in df.columns}
# Map common variations to expected names
rename_map = {}
for want in ("open","high","low","close","volume","timestamp"):
    if want in cols_lower:
        rename_map[cols_lower[want]] = want
# apply rename
if rename_map:
    df = df.rename(columns=rename_map)

# Verify OHLCV presence
required = ["open","high","low","close"]
if not all(c in df.columns for c in required):
    print("CSV missing required OHLCV columns. Found:", list(df.columns))
    print("Expected at least:", required)
    # continue but warn
else:
    print("Found OHLCV columns, proceeding")

# Run feature engineering
pipe = FeatureEngineeringPipeline()
features = pipe.extract_features(df)
aligned = pipe.align_and_finalize_features(features)

# Save engineered features
fe_path = os.path.join(OUT_DIR, "feature_engineered_samples.csv")
aligned.to_csv(fe_path, index=False)

# Feature stats (NaN, inf, mean/std)
feat_stats = {}
feat_stats["shape"] = list(aligned.shape)
feat_stats["nan_counts"] = aligned.isna().sum().to_dict()
feat_stats["inf_counts"] = np.isinf(aligned.values).sum(axis=0).tolist()
feat_stats["per_col_mean_sample"] = aligned.mean().fillna(0).tolist()[:20]
feat_stats["per_col_std_sample"] = aligned.std().fillna(0).tolist()[:20]
open(os.path.join(OUT_DIR, "feature_stats.json"), "w").write(json.dumps(feat_stats, indent=2))

# Load scaler and apply
scaler_path = "data/models/regime/scaler.pkl"
res = {"scaler_path": scaler_path, "aligned_path": fe_path}
if not os.path.exists(scaler_path):
    res["error"] = "scaler not found"
    open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
    print("Scaler not found:", scaler_path)
    raise SystemExit(0)

scaler = joblib.load(scaler_path)
X = aligned.values.astype(float)
# Try transform
try:
    Xs = scaler.transform(X)
    res["applied"] = "scaler.transform"
except Exception:
    # fallback to manual mean_/scale_
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = scaler.mean_.reshape(1, -1)
        scale = scaler.scale_.reshape(1, -1)
        Xs = (X - mean) / (scale + 1e-12)
        res["applied"] = "manual_mean_scale"
    else:
        res["error"] = "scaler has no transform or mean_/scale_"
        open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
        raise SystemExit(0)

res["scaled_mean_sample"] = list(np.nanmean(Xs, axis=0)[:20])
res["scaled_std_sample"] = list(np.nanstd(Xs, axis=0)[:20])

# Save scaled features CSV for debugging
np.save(os.path.join(OUT_DIR, "X_scaled.npy"), Xs)

# Run model if present
model_path = "diagnostics/inst_model.pth"
if not os.path.exists(model_path):
    res["model_error"] = f"inst_model.pth not found at {model_path}"
    open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
    print("Model not found:", model_path)
    raise SystemExit(0)

try:
    model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    net = getattr(model_obj, "q_network", model_obj)
    if hasattr(net, "eval"):
        net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(Xs[:5], dtype=torch.float32)).detach().cpu().numpy()
        probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
    res["model_results"] = {
        "logits_mean": logits.mean(axis=0).tolist(),
        "logits_std": logits.std(axis=0).tolist(),
        "probs_mean": probs.mean(axis=0).tolist(),
        "entropy_mean": float(- (probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean())
    }
except Exception as e:
    res["model_error"] = str(e)
    res["model_trace"] = traceback.format_exc()[:4000]

open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
print("Wrote diagnostics:", fe_path, "and scaler_apply_result.json")
