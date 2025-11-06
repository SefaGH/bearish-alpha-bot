#!/usr/bin/env python3
"""
scripts/verify_model.py
"""
import torch
import inspect
import sys
import os
import importlib
import traceback
from scripts.safe_torch_load import safe_torch_load # <-- YENİ İMPORT

MODEL_CLASS_IMPORT = os.environ.get("MODEL_CLASS_IMPORT", "src.ml.reinforcement_learning.TradingRLAgent")
print(f"Verification will use model_class_import: {MODEL_CLASS_IMPORT}")

p = "diagnostics/inst_model.pth"
print(f"Loading instantiated model: {p}")

if not os.path.exists(p):
    print(f"[ERROR] File not found: {p}")
    sys.exit(1)

try:
    # --- ÇÖZÜM: safe_torch_load kullanılıyor ---
    obj = safe_torch_load(p, model_class_import=MODEL_CLASS_IMPORT, map_location="cpu")
except Exception as e:
    print(f"[ERROR] safe_torch_load failed: {e}")
    print("---------------------------------")
    print("Stack Trace:")
    print(traceback.format_exc())
    print("---------------------------------")
    sys.exit(1)

print("\n--- VERIFICATION RESULTS ---")
print(f"TYPE: {type(obj)}")

try:
    import torch.nn as nn
    print(f"is nn.Module: {isinstance(obj, nn.Module)}")
except Exception:
    pass

print(f"callable(obj): {callable(obj)}")
print(f"has q_network: {hasattr(obj, 'q_network')}")
print(f"has predict_proba: {hasattr(obj, 'predict_proba')}")

try:
    print(f"dir sample: {[n for n in dir(obj) if not n.startswith('_')][:200]}")
except Exception as e:
    print(f"dir sample failed: {e}")

if hasattr(obj, "q_network"):
    try:
        print(f"q_network type: {type(obj.q_network)}")
        if hasattr(obj.q_network, "children"):
             q_children = list(obj.q_network.children())
             print(f"q_network children count: {len(q_children)}")
             if q_children:
                 print(f"q_network first child type: {type(q_children[0])}")
    except Exception as e:
        print(f"q_network inspect failed: {e}")
        
print("--- END VERIFICATION ---")
