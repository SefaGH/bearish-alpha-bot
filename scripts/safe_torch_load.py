#!/usr/bin/env python3
"""
scripts/safe_torch_load.py

safe_torch_load(path, model_class_import=None, map_location='cpu')

Attempts to load a torch object robustly across PyTorch versions:
- Try normal torch.load(...) first (weights-only default)
- If that fails with an unpickling/weights-only error and model_class_import is provided,
  import the class and use torch.serialization.add_safe_globals([Class]) (if available)
  to unpickle safely (allowed global).
- If the above fails, try torch.load(..., weights_only=False) as a last resort.

Security note: loading pickled Python objects can execute code. Only allow this for
trusted model files.
"""
from __future__ import annotations
import importlib
import typing as t
import traceback # Hata ayıklama için eklendi

def safe_torch_load(path: str, model_class_import: t.Optional[str] = None, map_location: str = "cpu"):
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"torch is required to load model: {e}")

    # 1) Try the normal load first (this will usually return state_dicts or weights-only)
    try:
        return torch.load(path, map_location=map_location)
    except Exception as first_err:
        # keep first_err for re-raising if all attempts fail
        first_exc = first_err

    # 2) If a model class import is provided, try to import it and allowlist it for unpickling
    if model_class_import:
        try:
            module_path, cls_name = model_class_import.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            Klass = getattr(mod, cls_name)
            
            # Try add_safe_globals context manager if available (PyTorch 2.6+)
            try:
                from torch.serialization import add_safe_globals
                try:
                    # add_safe_globals ile denerken, weights_only=False olmalı
                    with add_safe_globals([Klass]):
                        return torch.load(path, map_location=map_location, weights_only=False)
                except Exception:
                    # fallthrough to next attempt
                    pass
            except Exception:
                # add_safe_globals not available or failed; fallthrough
                pass
        except Exception as import_exc:
            # Can't import class - continue to next fallback
            import_exc_str = traceback.format_exc()
            # proceed to next attempt

    # 3) Last resort: try loading with weights_only=False (note: may be unsafe)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as final_err:
        # raise the original first error for better debugging context
        raise RuntimeError(
            "All torch.load attempts failed. First error: "
            f"{first_exc}\nFinal attempt error: {final_err}"
        ) from final_err
