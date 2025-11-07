#!/usr/bin/env python3
"""
scripts/feature_engineer_and_run.py

(Updated) After feature extraction:
 - coerce to numeric
 - ffill/bfill as in training prepare_for_training
 - fill remaining NaNs with scaler.mean_ (fallback)
 - warn and abort if too many NaNs or zero rows remain
 - use last row(s) for inference
"""
import os
import sys

# Add scripts directory to path to import setup_path module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Setup Python path to enable imports from src module
from setup_path import setup_project_path
setup_project_path()

# Now imports will work
import json
import traceback

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
cols_lower = {c.lower(): c for c in df.columns}
rename_map = {}
for want in ("open","high","low","close","volume","timestamp"):
    if want in cols_lower:
        rename_map[cols_lower[want]] = want
if rename_map:
    df = df.rename(columns=rename_map)

pipe = FeatureEngineeringPipeline()
features = pipe.extract_features(df)
aligned = pipe.align_and_finalize_features(features)

# Coerce to numeric (non-numeric -> NaN)
aligned = aligned.apply(pd.to_numeric, errors='coerce')

# Mirror training preprocessing: forward-fill then back-fill
aligned_ff = aligned.ffill().bfill()

# If still NaNs remain, fill with scaler.mean_ if available (preferable) otherwise 0
scaler_path = "data/models/regime/scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    if hasattr(scaler, "mean_"):
        mean_vec = np.asarray(scaler.mean_, dtype=float).reshape(1, -1)
    else:
        mean_vec = None
else:
    scaler = None
    mean_vec = None

# Count NaNs after ffill/bfill
nan_counts_after = aligned_ff.isna().sum().sum()
total_cells = aligned_ff.size
nan_fraction = nan_counts_after / total_cells if total_cells>0 else 1.0

# If proportion of NaNs is high, warn the user and continue but report
THRESH_WARN = 0.05
THRESH_ABORT = 0.5
report = {}
report["nan_after_ffill_bfill"] = int(nan_counts_after)
report["nan_fraction"] = nan_fraction

if nan_fraction >= THRESH_ABORT:
    report["error"] = "Too many NaNs in engineered features after ffill/bfill. Provide longer OHLCV."
    with open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w") as f:
        f.write(json.dumps(report, indent=2))
    print("Aborting: too many NaNs after fills. See diagnostics/scaler_apply_result.json")
    raise SystemExit(0)
elif nan_fraction >= THRESH_WARN:
    report["warning"] = f"High NaN fraction ({nan_fraction:.2%}) after ffill/bfill; using scaler.mean_ to fill remaining."

# Fill remaining NaNs with scaler mean or 0
if mean_vec is not None:
    # broadcast mean to shape
    col_means = pd.Series(mean_vec.ravel(), index=aligned_ff.columns)
    aligned_filled = aligned_ff.fillna(col_means)
else:
    aligned_filled = aligned_ff.fillna(0.0)

# Save aligned & filled features for inspection
fe_path = os.path.join(OUT_DIR, "feature_engineered_samples.csv")
aligned_filled.to_csv(fe_path, index=False)

# Stats (safe): use numeric numpy array
arr = aligned_filled.values.astype(float)
feat_stats = {
    "shape": list(arr.shape),
    "nan_counts": aligned_filled.isna().sum().to_dict(),
    "inf_counts": np.isinf(arr).sum(axis=0).tolist(),
    "per_col_mean_sample": np.nan_to_num(np.nanmean(arr, axis=0))[:20].tolist(),
    "per_col_std_sample": np.nan_to_num(np.nanstd(arr, axis=0))[:20].tolist()
}
with open(os.path.join(OUT_DIR, "feature_stats.json"), "w") as f:
    f.write(json.dumps(feat_stats, indent=2))

# If no rows, abort
if arr.shape[0] == 0:
    out = {"error": "No rows after feature engineering."}
    with open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w") as f:
        f.write(json.dumps(out, indent=2))
    print("No rows to process; aborting.")
    raise SystemExit(0)

# Apply scaler
res = {"scaler_path": scaler_path, "aligned_path": fe_path}
if scaler is None:
    res["error"] = "scaler not found"
    with open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w") as f:
        f.write(json.dumps(res, indent=2))
    raise SystemExit(0)

try:
    Xs = scaler.transform(arr)
    res["applied"] = "scaler.transform"
except Exception:
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = scaler.mean_.reshape(1, -1)
        scale = scaler.scale_.reshape(1, -1)
        Xs = (arr - mean) / (scale + 1e-12)
        res["applied"] = "manual_mean_scale"
    else:
        res["error"] = "scaler has no transform or mean_/scale_"
        with open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w") as f:
            f.write(json.dumps(res, indent=2))
        raise SystemExit(0)

res["scaled_mean_sample"] = list(np.nan_to_num(np.nanmean(Xs, axis=0))[:20])
res["scaled_std_sample"] = list(np.nan_to_num(np.nanstd(Xs, axis=0))[:20])

# Save scaler results first
with open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w") as f:
    f.write(json.dumps(res, indent=2))

# Run model on last row(s) (use last row)
# Try multiple possible model paths
model_paths = [
    "data/models/regime/rl_agent_final.pth",
    "diagnostics/inst_model.pth",
    "data/models/rl_agent_final.pth",
    "data/models/regime/transformer_regime.pth",
    "data/models/regime/lstm_regime.pth"
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        print(f"Using model: {model_path}")
        break

model_res = {}
if model_path is None:
    model_res["error"] = "No model found in any of the expected paths"
    model_res["searched_paths"] = model_paths
else:
    try:
        model_obj = torch.load(model_path, map_location="cpu", weights_only=False)
        net = getattr(model_obj, "q_network", model_obj)
        if hasattr(net, "eval"):
            net.eval()
        with torch.no_grad():
            # use last row for inference
            logits = net(torch.tensor(Xs[-1:].astype(np.float32))).detach().cpu().numpy()
            probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
        model_res = {
            "model_path": model_path,
            "logits": logits.tolist(),
            "probabilities": probs.tolist(),
            "entropy_mean": float(- (probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean())
        }
    except Exception as e:
        model_res["error"] = str(e)
        model_res["trace"] = traceback.format_exc()[:4000]
        model_res["model_path"] = model_path

# Save model results to separate file
with open(os.path.join(OUT_DIR, "model_results.json"), "w") as f:
    f.write(json.dumps(model_res, indent=2))
print("Wrote diagnostics:", fe_path, ", scaler_apply_result.json, and model_results.json")
