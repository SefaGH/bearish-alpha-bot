#!/usr/bin/env python3
"""
Load data/models/regime/scaler.pkl and print small summary: type, len(mean_), sample of mean_/scale_.
"""
import os, json
try:
    import joblib
except Exception as e:
    print("joblib import failed:", e)
    raise SystemExit(1)

p = "data/models/regime/scaler.pkl"
if not os.path.exists(p):
    print("scaler not found at", p)
    # Hata vermek yerine JSON çıktısı üret
    print(json.dumps({"error": "scaler not found", "path": p}))
    raise SystemExit(1)

try:
    scaler = joblib.load(p)
    out = {"type": str(type(scaler)), "path": p}
    
    # sklearn StandardScaler
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        out["kind"] = "StandardScaler"
        out["n_features_in"] = int(getattr(scaler, "n_features_in_", getattr(scaler, "mean_", [0]).shape[0]))
        out["mean_sample"] = list(map(float, scaler.mean_[:20]))
        out["scale_sample"] = list(map(float, scaler.scale_[:20]))
        
        # if feature names saved
        if hasattr(scaler, "feature_names_in_"):
            out["feature_names_in_sample"] = list(scaler.feature_names_in_[:20])
    else:
        out["info"] = "Scaler does not expose mean_/scale_ attributes; type: " + str(type(scaler))
    
    print(json.dumps(out, indent=2))

except Exception as e:
    print(json.dumps({"error": "Failed to load or inspect scaler", "exc": str(e), "path": p}))
    raise SystemExit(1)
