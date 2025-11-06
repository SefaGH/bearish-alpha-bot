#!/usr/bin/env python3
"""
compute_state_vector(record) wrapper used by scripts/diagnose_model_confidence.py

- record: a dict (one CSV row as returned by df.to_dict(orient='records'))
- returns: 1D numpy array (float32) representing the state vector for the model
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import re

# Try to import the project's FeatureEngineeringPipeline. If import fails,
# fallback to a simple numeric-vector conversion.
try:
    from src.ml.feature_engineering import FeatureEngineeringPipeline
    FE_AVAILABLE = True
except Exception:
    FE_AVAILABLE = False

_pipeline: Optional["FeatureEngineeringPipeline"] = None
def _get_pipeline():
    global _pipeline
    if not FE_AVAILABLE:
        return None
    if _pipeline is None:
        _pipeline = FeatureEngineeringPipeline()
    return _pipeline

def compute_state_vector(record: Dict[str, Any]) -> np.ndarray:
    """
    Convert a single record (dict of raw features) into a state vector.
    Expected usage: record comes from pandas.DataFrame.to_dict(orient='records')
    """
    # QUICK GUARD: if the record looks like precomputed features (f0..fN),
    # return those directly and skip the FeatureEngineeringPipeline.
    # This is the guard you asked about and should be placed at the very
    # beginning of compute_state_vector (before calling _get_pipeline()).
    try:
        # Detect keys matching f0, f1, f2, ... (case-sensitive)
        f_keys = [k for k in record.keys() if re.fullmatch(r"f\d+", k)]
        if f_keys:
            # Sort keys by numeric index (f0, f1, f2, ...)
            def key_index(k: str) -> int:
                return int(k[1:])
            f_keys_sorted = sorted(f_keys, key=key_index)
            vals = []
            for k in f_keys_sorted:
                v = record.get(k, 0.0)
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(0.0)
            return np.array(vals, dtype=np.float32)
    except Exception:
        # If anything goes wrong in guard, fall back to full pipeline below
        pass

    # Defensive: if record has nested objects that are not numeric, handle gracefully
    try:
        # If FeatureEngineeringPipeline is available, prefer it
        pipeline = _get_pipeline()
        if pipeline is not None:
            # Convert single record to DataFrame (1 row)
            df = pd.DataFrame([record])
            # Ensure columns that pipeline expects exist; pipeline should handle missing columns
            feats = pipeline.extract_features(df)
            # If pipeline returns a DataFrame with multiple rows, pick the first
            row = feats.iloc[0].to_numpy(dtype=np.float32)
            return row.astype(np.float32)
        else:
            # Fallback: try to extract numeric columns in sorted order
            numeric_vals = []
            for k, v in sorted(record.items()):
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    # Non-numeric -> 0.0 placeholder
                    numeric_vals.append(0.0)
            return np.array(numeric_vals, dtype=np.float32)
    except Exception:
        # Last-resort fallback: turn values to numbers or zeros
        vals = []
        for k in sorted(record.keys()):
            try:
                vals.append(float(record[k]))
            except Exception:
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)
