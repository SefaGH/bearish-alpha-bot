#!/usr/bin/env python3
"""
scripts/multiplier_probe.py

Probe input multipliers, choose one that minimizes mean entropy of model outputs.
Provides choose_multiplier_by_entropy(model_net, X, multipliers) -> (best_multiplier, meta)
"""
import json
import traceback
import sys
import os

# CLI testi için __main__ bloğuna import'lar eklendi
try:
    import numpy as _np
    import torch
    from torch.nn import functional as F
except Exception as e:
    print(f"Import Error: {e}. Lütfen torch, numpy kurun.", file=sys.stderr)
    sys.exit(1)


def _entropy_rows(probs: _np.ndarray) -> _np.ndarray:
    # probs: (N, C)
    eps = 1e-12
    p = _np.clip(probs, eps, 1.0)
    return -_np.sum(p * _np.log(p), axis=1)

def choose_multiplier_by_entropy(model_net, X, multipliers=(1.0,2,5,10,20,50,100,200,500,1000)):
    """
    model_net: a torch.nn.Module object that accepts a torch.Tensor input and returns logits
    X: numpy array (N, D)
    multipliers: iterable of numeric multipliers to try
    returns: (best_multiplier, meta_dict)
    meta_dict includes per-multiplier stats: mean_entropy, logits_mean/std etc. (converted to python types)
    """
    
    meta = {"tried": []}
    best = None
    
    # model_net'in eval modunda olduğundan emin ol
    try:
        if hasattr(model_net, "eval") and callable(model_net.eval):
            model_net.eval()
    except Exception:
        pass # Hata verirse devam et

    for m in multipliers:
        try:
            Xm = _np.asarray(X, dtype=float) * float(m)
            xt = torch.tensor(Xm, dtype=torch.float32)
            with torch.no_grad():
                logits = model_net(xt).detach().cpu().numpy()
            probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
            ent = _entropy_rows(probs)
            record = {
                "multiplier": float(m),
                "logits_mean": logits.mean(axis=0).tolist(),
                "logits_std": logits.std(axis=0).tolist(),
                "probs_mean": probs.mean(axis=0).tolist(),
                "probs_std": probs.std(axis=0).tolist(),
                "entropy_mean": float(ent.mean()),
                "entropy_std": float(ent.std())
            }
            meta["tried"].append(record)
            if best is None or record["entropy_mean"] < best[0]:
                best = (record["entropy_mean"], float(m), record)
        except Exception as e:
            meta["tried"].append({"multiplier": float(m), "error": str(e)})
            
    if best is None:
        raise RuntimeError("No multiplier produced a valid output")
        
    meta["chosen"] = {"multiplier": best[1], "entropy_mean": float(best[0]), "chosen_record": best[2]}
    return best[1], meta

# convenience CLI for local testing
if __name__ == "__main__":
    import sys, os
    try:
        import numpy as np, torch, pandas as pd
        from pathlib import Path
        from scripts.safe_torch_load import safe_torch_load # Güvenli yükleyiciyi kullan
        
        print("Running multiplier_probe.py as CLI for testing...")
        
        model_path = os.environ.get("MODEL_TO_USE", "diagnostics/inst_model.pth")
        csv = os.environ.get("SAMPLES_PATH", "sample_data/test_samples.csv")
        model_class_import = os.environ.get("MODEL_CLASS_IMPORT", "src.ml.reinforcement_learning.TradingRLAgent")
        multipliers = [float(x) for x in os.environ.get("MULTIPLIERS","1,2,5,10,20,50,100,200,500,1000").split(",")]
        
        if not Path(model_path).exists():
            raise SystemExit(f"Model not found: {model_path}")
            
        model = safe_torch_load(model_path, model_class_import=model_class_import, map_location="cpu")
        net = getattr(model, "q_network", model)
        
        # state_size'ı oku (veri kesmek için)
        expected_state_size = None
        try:
            with open("diagnostics/inferred_state_size.txt", "r") as f:
                expected_state_size = int(f.read().strip())
        except Exception:
            pass # Bulamazsa None kalır
        
        if Path(csv).exists():
            df = pd.read_csv(csv)
            for c in ["label","target","y","class"]:
                if c in df.columns: df = df.drop(columns=[c])
            X = df.select_dtypes(include=[float,int]).to_numpy(dtype=float)[:10]
            
            # Veriyi kes
            if expected_state_size is not None and X.shape[1] > expected_state_size:
                X = X[:, :expected_state_size]
                
        else:
            state_size_to_use = expected_state_size if expected_state_size else 42
            X = np.random.randn(5, state_size_to_use)
            
        best, meta = choose_multiplier_by_entropy(net, X, multipliers=multipliers)
        out = {"best_multiplier": best, "meta_report": meta}
        print(json.dumps(out, indent=2))
        
    except Exception as e:
        print("FAILED:", e)
        print(traceback.format_exc())
        sys.exit(2)
