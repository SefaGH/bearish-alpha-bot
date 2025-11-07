#!/usr/bin/env python3
"""
scripts/feature_matcher.py

Automatic matcher: try to map numeric input CSV columns (f0..fN or arbitrary numeric columns)
to trained FEATURE_COLUMNS using scaler mean_/scale_ patterns.

Outputs:
 - diagnostics/mapping.json : mapping {input_col -> feature_name, feature_index, cost}
 - diagnostics/mapped_samples.csv : input CSV renamed to FEATURE_COLUMNS order (first N features)
 - diagnostics/mapping_apply_result.json : post-scaling stats (logits/probs if model present)

Usage:
  python scripts/feature_matcher.py sample_data/test_samples.csv
"""
import os
import sys

# Add project root to Python path (insert at beginning to ensure local src module is used)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Now imports will work
import json
import traceback
from pathlib import Path

IN_CSV = sys.argv[1] if len(sys.argv) > 1 else "sample_data/test_samples.csv"
OUT_DIR = "diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

def safe_imports():
    try:
        import joblib, pandas as pd, numpy as np
        from src.ml.feature_engineering import FeatureEngineeringPipeline
        return joblib, pd, np, FeatureEngineeringPipeline
    except Exception as e:
        raise RuntimeError(f"Imports failed: {e}")

def build_cost_matrix(X_cols, means, scales):
    # X_cols: list of numpy arrays shape (n_samples,)
    # means/scales: arrays length M (n_features)
    import numpy as _np
    n = len(X_cols); m = len(means)
    cost = _np.zeros((n, m), dtype=float)
    for i, col in enumerate(X_cols):
        for j in range(m):
            z = (col - means[j]) / (scales[j] + 1e-12)
            # use mean absolute z as cost (robust)
            cost[i, j] = float(_np.mean(_np.abs(z)))
    return cost

def solve_assignment(cost):
    # try Hungarian via scipy, else greedy
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # greedy fallback: iteratively pick smallest cost
        import numpy as _np
        cost_copy = cost.copy()
        n, m = cost_copy.shape
        assigned = []
        used_cols = set()
        for _ in range(min(n, m)):
            idx = _np.unravel_index(_np.argmin(cost_copy, axis=None), cost_copy.shape)
            i, j = int(idx[0]), int(idx[1])
            if j in used_cols:
                cost_copy[i, j] = 1e12
                continue
            assigned.append((i, j))
            used_cols.add(j)
            cost_copy[i, :] = 1e12
            cost_copy[:, j] = 1e12
        return assigned

def main():
    try:
        joblib, pd, np, FE = safe_imports()
    except Exception as e:
        print("Import error:", e)
        traceback.print_exc()
        raise SystemExit(1)

    if not Path(IN_CSV).exists():
        print("Input CSV not found:", IN_CSV)
        raise SystemExit(1)

    # load scaler
    scaler_path = "data/models/regime/scaler.pkl"
    if not Path(scaler_path).exists():
        print("Scaler not found:", scaler_path)
        raise SystemExit(1)
    scaler = joblib.load(scaler_path)
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        print("Scaler missing mean_/scale_ attributes")
        raise SystemExit(1)
    means = np.array(scaler.mean_, dtype=float)
    scales = np.array(scaler.scale_, dtype=float)
    M = len(means)

    # feature names
    pipe = FE()
    FEATURE_COLUMNS = pipe.FEATURE_COLUMNS
    if len(FEATURE_COLUMNS) != M:
        print("WARNING: FEATURE_COLUMNS length != scaler n_features ({} != {})".format(len(FEATURE_COLUMNS), M))

    # read csv
    df = pd.read_csv(IN_CSV)
    # drop known label-like cols
    for c in ["label","target","y","class"]:
        if c in df.columns:
            try:
                df = df.drop(columns=[c])
            except Exception:
                pass

    # get numeric columns in order found
    numeric_cols = df.select_dtypes(include=[float,int]).columns.tolist()
    if len(numeric_cols) == 0:
        print("No numeric columns found in CSV")
        raise SystemExit(1)
    X = df[numeric_cols].astype(float).to_numpy() # shape (N, n_input_cols)
    n_input = X.shape[1]

    # prepare per-column arrays
    X_cols = [X[:, i] for i in range(n_input)]

    # build cost matrix (n_input x M)
    cost = build_cost_matrix(X_cols, means, scales)

    # solve assignment
    assigned = solve_assignment(cost)

    # build mapping: input_col -> feature_name
    mapping = {}
    for i, j in assigned:
        inp_col = numeric_cols[i]
        feat_idx = int(j)
        feat_name = FEATURE_COLUMNS[feat_idx] if feat_idx < len(FEATURE_COLUMNS) else f"feat_{feat_idx}"
        mapping[inp_col] = {"feature_index": feat_idx, "feature_name": feat_name, "cost": float(cost[i, j])}

    # for unmatched input cols (if any), leave unmapped
    mapped_input_cols = set(mapping.keys())
    for c in numeric_cols:
        if c not in mapped_input_cols:
            mapping[c] = {"feature_index": None, "feature_name": None, "cost": None}

    # produce mapped DataFrame: create df_map with FEATURE_COLUMNS order
    mapped_df = pd.DataFrame(columns=FEATURE_COLUMNS, index=df.index)
    # fill mapped columns
    for inp_col, info in mapping.items():
        if info["feature_index"] is not None:
            mapped_df[info["feature_name"]] = df[inp_col].astype(float)
    # missing columns will remain NaN; align_and_finalize_features will fill them if needed
    mapped_df = pipe.align_and_finalize_features(mapped_df)

    # save mapped csv
    mapped_csv = Path(OUT_DIR) / "mapped_samples.csv"
    mapped_df.to_csv(mapped_csv, index=False)

    # apply scaler (safe)
    try:
        Xs = scaler.transform(mapped_df.values.astype(float))
        applied = "scaler.transform"
    except Exception:
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            mean = scaler.mean_.reshape(1, -1)
            scale = scaler.scale_.reshape(1, -1)
            Xs = (mapped_df.values.astype(float) - mean) / (scale + 1e-12)
            applied = "manual_mean_scale"
        else:
            Xs = None
            applied = "failed"

    # if model present, run q_network on first 5 rows and collect logits/probs
    model_path = "diagnostics/inst_model.pth"
    model_present = Path(model_path).exists()
    model_results = None
    if model_present and Xs is not None:
        try:
            import torch
            from torch.nn.functional import softmax
            obj = torch.load(model_path, map_location="cpu", weights_only=False)
            net = getattr(obj, "q_network", obj)
            if hasattr(net, "eval"):
                net.eval()
            with torch.no_grad():
                logits = net(torch.tensor(Xs[:5], dtype=torch.float32)).detach().cpu().numpy()
                probs = softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
            model_results = {
                "logits_mean": logits.mean(axis=0).tolist(),
                "logits_std": logits.std(axis=0).tolist(),
                "probs_mean": probs.mean(axis=0).tolist(),
                "entropy_mean": float(- (probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean())
            }
        except Exception as e:
            model_results = {"error": str(e)}

    # write outputs
    out_map = {"input_csv": IN_CSV, "scaler_path": scaler_path, "mapping": mapping, "applied": applied, "model_present": model_present}
    Path(OUT_DIR, "mapping.json").write_text(json.dumps(out_map, indent=2))
    Path(OUT_DIR, "mapping_apply_result.json").write_text(json.dumps({"applied": applied, "model_results": model_results}, indent=2))
    print("Wrote diagnostics/mapping.json, mapped_samples.csv, mapping_apply_result.json")
    print("Mapping summary (first 10):")
    for k, v in list(mapping.items())[:10]:
        print(k, "->", v)

if __name__ == "__main__":
    main()
