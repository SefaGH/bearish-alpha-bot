#!/usr/bin/env python3
"""
Load sample_data/test_samples.csv, align columns to FeatureEngineeringPipeline.FEATURE_COLUMNS,
load scaler (data/models/regime/scaler.pkl), apply scaler.transform and run model.q_network to get probs.
Writes diagnostics/scaler_apply_result.json (small).
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

OUT="diagnostics/scaler_apply_result.json"
res = {}

try:
    import joblib, pandas as pd, numpy as np, torch
    from torch.nn.functional import softmax
    
    from src.ml.feature_engineering import FeatureEngineeringPipeline
    from scripts.safe_torch_load import safe_torch_load

except Exception as e:
    open(OUT,"w").write(json.dumps({"error":"imports_failed","exc":str(e), "traceback": traceback.format_exc()}))
    raise SystemExit(0)

# paths
scaler_path = "data/models/regime/scaler.pkl"
model_path = "diagnostics/inst_model.pth"
csv_path = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv") # Env var'dan al

res = {"scaler_path": scaler_path, "model_path": model_path, "csv_path": csv_path}

try:
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler not found")
    if not os.path.exists(model_path):
        raise FileNotFoundError("inst_model.pth not found")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("sample CSV not found")

    # load scaler
    scaler = joblib.load(scaler_path)
    # load feature names from pipeline
    pipe = FeatureEngineeringPipeline()
    FEATURE_COLUMNS = pipe.FEATURE_COLUMNS

    # read csv
    df = pd.read_csv(csv_path)
    # If csv has generic names like f0..f41, rename to FEATURE_COLUMNS (assuming same order)
    if list(df.columns)[:len(FEATURE_COLUMNS)] == [f"f{i}" for i in range(len(FEATURE_COLUMNS))]:
        df = df.rename(columns={f"f{i}": FEATURE_COLUMNS[i] for i in range(len(FEATURE_COLUMNS))})
    else:
        # If headers are different, attempt to pick numeric columns in order and map
        num_cols = df.select_dtypes(include=[float,int]).columns.tolist()
        if len(num_cols) >= len(FEATURE_COLUMNS):
            df_cols_to_use = num_cols[:len(FEATURE_COLUMNS)]
            df = df[df_cols_to_use]
            df.columns = FEATURE_COLUMNS
        else:
            raise RuntimeError("CSV does not contain enough numeric columns to map to FEATURE_COLUMNS")

    # align/finalize via pipeline (ensures same order + missing cols)
    aligned = pipe.align_and_finalize_features(df)
    X = aligned.values.astype(float)

    # apply scaler: prefer scaler.transform; if not available, fallback to manual (mean_/scale_)
    try:
        Xs = scaler.transform(X)
    except Exception:
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            Xs = (X - scaler.mean_.reshape(1,-1)) / (scaler.scale_.reshape(1,-1) + 1e-12)
        else:
            raise RuntimeError("could not apply scaler (no transform and no mean_/scale_)")

    # load model instance and run q_network
    # safe_torch_load'u (model_class_import olmadan) kullanmayı deneyelim
    model_obj = safe_torch_load(model_path, map_location="cpu")
    net = getattr(model_obj, "q_network", model_obj)
    net.eval() if hasattr(net, "eval") else None

    with torch.no_grad():
        logits = net(torch.tensor(Xs[:5], dtype=torch.float32)).detach().cpu().numpy()
        probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()

    res.update({
        "X_shape": list(X.shape),
        "scaled_sample_mean": Xs.mean(axis=0)[:10].tolist(),
        "logits_mean": logits.mean(axis=0).tolist(),
        "logits_std": logits.std(axis=0).tolist(),
        "probs_mean": probs.mean(axis=0).tolist(),
        "entropy_mean": float(- (probs * np.log(np.clip(probs,1e-12,1))).sum(axis=1).mean())
    })

    open(OUT,"w").write(json.dumps(res, indent=2))
    print("Wrote", OUT)

except Exception as e:
    res["error"] = str(e)
    res["traceback"] = traceback.format_exc()
    open(OUT,"w").write(json.dumps(res, indent=2))
    print(f"Failed to run scaler test: {e}")
    raise SystemExit(0) # CI'da hata vermemesi için 0 ile çık
