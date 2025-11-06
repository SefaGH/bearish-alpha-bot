#!/usr/bin/env python3
"""
scripts/verify_model.py

Bu script, 'instantiate_model.py' tarafından oluşturulmuş olması beklenen
'diagnostics/inst_model.pth' dosyasını yüklemeyi dener.

Yüklenen nesnenin (object) tipini, 'callable' olup olmadığını, 
'nn.Module' olup olmadığını ve 'q_network' özelliğine sahip olup 
olmadığını kontrol eder ve konsola yazdırır.
"""
import torch
import inspect
import sys
import os
import importlib

# --- ÖNEMLİ: Ana sınıfı import et ---
# torch.load'un pickle'ı çözebilmesi için, TradingRLAgent sınıfının
# bu script tarafından da bilinmesi gerekir.
MODEL_CLASS_IMPORT = os.environ.get("MODEL_CLASS_IMPORT", "src.ml.reinforcement_learning.TradingRLAgent")
if MODEL_CLASS_IMPORT:
    try:
        print(f"Pre-importing class {MODEL_CLASS_IMPORT} for torch.load verification...")
        module_path, class_name = MODEL_CLASS_IMPORT.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        Klass = getattr(mod, class_name)
        print("Class imported successfully for verification.")
    except Exception as e:
        print(f"[WARN] Failed to pre-import model class: {e}. torch.load might fail.")
# --- BİTTİ ---

p = "diagnostics/inst_model.pth"
print(f"Loading instantiated model: {p}")

if not os.path.exists(p):
    print(f"[ERROR] File not found: {p}")
    sys.exit(1)

try:
    obj = torch.load(p, map_location="cpu")
except Exception as e:
    print(f"[ERROR] torch.load failed: {e}")
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
