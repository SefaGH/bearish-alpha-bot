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

# --- GÜVENLİK DÜZELTMESİ (ANALİZ-O) ---
# np.isinf'in 'object' tipiyle çalışabilmesi için önce sayısala zorla
# Ve 'object' tipindeki sütunları (örn. boş stringler) sayısala dönüştür
aligned_numeric = aligned.apply(pd.to_numeric, errors='coerce')
# --- DÜZELTME SONU ---

# Feature stats (NaN, inf, mean/std)
feat_stats = {}
feat_stats["shape"] = list(aligned_numeric.shape)
feat_stats["nan_counts"] = aligned_numeric.isna().sum().to_dict()
# Güvenli .values kullanımı (artık 'object' tipi içermiyor)
# --- GÜNCELLENMİŞ BLOK (ANALİZ-P) ---
# Coerce to numeric (safe): convert all columns to numeric, non-convertible -> NaN
aligned_numeric = aligned.apply(pd.to_numeric, errors='coerce')

# Feature stats (NaN, inf, mean/std)
feat_stats = {}
feat_stats["shape"] = list(aligned_numeric.shape)
feat_stats["nan_counts"] = aligned_numeric.isna().sum().to_dict()

# Now compute inf counts safely on numeric array
arr = aligned_numeric.values.astype(float)
feat_stats["inf_counts"] = np.isinf(arr).sum(axis=0).tolist()
feat_stats["per_col_mean_sample"] = np.nan_to_num(np.nanmean(arr, axis=0))[:20].tolist()
feat_stats["per_col_std_sample"] = np.nan_to_num(np.nanstd(arr, axis=0))[:20].tolist()
open(os.path.join(OUT_DIR, "feature_stats.json"), "w").write(json.dumps(feat_stats, indent=2))

# If no rows after extraction, bail out gracefully
if arr.shape[0] == 0:
    res = {"error": "No feature rows after extraction (0 samples). Aborting scaler/model step."}
    open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
    print("No features to process; wrote scaler_apply_result.json with error.")
    raise SystemExit(0)
# --- GÜNCELLEME SONU ---

# Load scaler and apply
scaler_path = "data/models/regime/scaler.pkl"
res = {"scaler_path": scaler_path, "aligned_path": fe_path}
if not os.path.exists(scaler_path):
    res["error"] = "scaler not found"
    open(os.path.join(OUT_DIR, "scaler_apply_result.json"), "w").write(json.dumps(res, indent=2))
    print("Scaler not found:", scaler_path)
    raise SystemExit(0)

scaler = joblib.load(scaler_path)

# --- GÜVENLİK DÜZELTMESİ (ANALİZ-O) ---
# Scaler'a göndermeden önce NaN değerleri doldur (prepare_for_training gibi)
# Önce ffill (ileri doldurma), sonra bfill (geri doldurma)
aligned_filled = aligned_numeric.ffill().bfill()
# Hala NaN kaldıysa (örn. tüm sütun NaN ise) 0 ile doldur
aligned_filled = aligned_filled.fillna(0)
X = aligned_filled.values.astype(float)
# --- DÜZELTME SONU ---

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
        # Xs'te hala NaN varsa (çok olası değil ama)
        Xs_safe = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        logits = net(torch.tensor(Xs_safe[:5], dtype=torch.float32)).detach().cpu().numpy()
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
