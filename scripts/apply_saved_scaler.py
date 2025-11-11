#!/usr/bin/env python3
"""
Apply saved scaler (data/models/regime/scaler.pkl) to an input DataFrame or numpy array.
Returns (X_scaled, meta) where X_scaled is np.ndarray suitable for model forward.
Safe: tries scaler.transform, falls back to manual mean_/scale_; attempts column mapping.
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

def apply_saved_scaler(X_df, model_obj=None, scaler_path="data/models/regime/scaler.pkl"):
    """
    X_df: pandas.DataFrame containing either FEATURE_COLUMNS-named columns or numeric columns in order.
    model_obj: optional, if you need underlying model for checks (unused here)
    Returns: (Xs, meta) where Xs is numpy array, meta is dict with info
    """
    meta = {"scaler_path": scaler_path}
    try:
        import joblib, numpy as np
        from src.ml.feature_engineering import FeatureEngineeringPipeline
    except Exception as e:
        meta["error"] = f"import_error: {e}"
        return None, meta

    if not os.path.exists(scaler_path):
        meta["error"] = "scaler_not_found"
        return None, meta

    scaler = joblib.load(scaler_path)
    pipe = FeatureEngineeringPipeline()
    FEATURE_COLUMNS = pipe.FEATURE_COLUMNS

    # Attempt to map/rename columns:
    df = X_df.copy()
    # If already has FEATURE_COLUMNS, use directly
    if set(FEATURE_COLUMNS).issubset(set(df.columns)):
        df_aligned = pipe.align_and_finalize_features(df)
        meta["mapping"] = "by_name"
    else:
        # Try numeric columns mapping
        num_cols = df.select_dtypes(include=[float,int]).columns.tolist()
        if len(num_cols) >= len(FEATURE_COLUMNS):
            df = df[num_cols[:len(FEATURE_COLUMNS)]]
            df.columns = FEATURE_COLUMNS
            df_aligned = pipe.align_and_finalize_features(df)
            meta["mapping"] = "by_numeric_order"
        else:
            meta["error"] = "not_enough_numeric_columns_to_map"
            return None, meta

    X = df_aligned.values.astype(float)
    meta["X_shape"] = list(X.shape)

    # Apply scaler.transform safely
    try:
        Xs = scaler.transform(X)
        meta["applied"] = "scaler.transform"
    except Exception:
        try:
            # fallback to manual mean_/scale_
            if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
                mean = scaler.mean_.reshape(1, -1)
                scale = scaler.scale_.reshape(1, -1)
                Xs = (X - mean) / (scale + 1e-12)
                meta["applied"] = "manual_mean_scale"
            else:
                meta["error"] = "scaler has no transform and no mean_/scale_"
                return None, meta
        except Exception as e:
            meta["error"] = f"scaler_apply_exception: {e}"
            return None, meta

    meta["scaled_mean_sample"] = Xs.mean(axis=0)[:20].tolist()
    return Xs, meta

if __name__ == "__main__":
    # quick CLI test: python scripts/apply_saved_scaler.py path/to/csv
    import sys, pandas as pd, json
    p = sys.argv[1] if len(sys.argv)>1 else "sample_data/test_samples.csv"
    if not os.path.exists(p):
        print(json.dumps({"error":"csv_not_found","path":p}))
        raise SystemExit(0)
    df = pd.read_csv(p)
    Xs, meta = apply_saved_scaler(df)
    print(json.dumps(meta, indent=2))
