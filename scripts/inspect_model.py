# scripts/inspect_model.py dosyasına bu kodu yapıştırın

import os, json, sys

def normalize_path(p: str | None) -> str | None:
    if p is None:
        return None
    p = p.strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    return os.path.abspath(p)

def resolve_model_path() -> str:
    env_p = normalize_path(os.environ.get("MODEL_PATH"))
    cli_p = normalize_path(sys.argv[1] if len(sys.argv) > 1 else None)
    default_p = normalize_path("data/models/rl_agent_final.pth")
    chosen = env_p or cli_p or default_p
    if not chosen:
        raise RuntimeError("Model path is empty after resolution.")
    if not os.path.exists(chosen):
        raise FileNotFoundError(
            f"Model file not found: {chosen}\n"
            f"Resolved from -> ENV: {env_p!r}, CLI: {cli_p!r}, DEFAULT: {default_p!r}"
        )
    return chosen

def main():
    out = {}
    try:
        p = resolve_model_path()
        out["path"] = p
    except Exception as e:
        out = {"path_resolution_error": str(e),
               "MODEL_PATH_env": os.environ.get("MODEL_PATH"),
               "argv": sys.argv[1:]}
        os.makedirs("diagnostics", exist_ok=True)
        with open("diagnostics/model_load.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Wrote diagnostics/model_load.json (path resolution error)")
        sys.exit(0)

    try:
        import torch
    except Exception as e:
        out["error"] = f"import torch failed: {e}"
        os.makedirs("diagnostics", exist_ok=True)
        with open("diagnostics/model_load.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("Wrote diagnostics/model_load.json (torch import error)")
        sys.exit(0)

    try:
        obj = torch.load(p, map_location="cpu")
        out["type"] = str(type(obj))
        if isinstance(obj, dict):
            keys = list(obj.keys())
            out["n_keys"] = len(keys)
            out["keys_sample"] = keys[:50]
            sample_info = {}
            for k in keys[:40]:
                v = obj[k]
                t = type(v).__name__
                shape = getattr(v, "shape", None)
                sample_info[k] = {"type": t, "shape": str(shape)}
            out["sample_values_info"] = sample_info
        else:
            try:
                attrs = [a for a in dir(obj) if not a.startswith("_")]
                out["attrs_sample"] = attrs[:200]
            except Exception as e:
                out["attrs_error"] = str(e)
            try:
                out["repr"] = repr(obj)[:1000]
            except Exception:
                out["repr_error"] = "repr failed"
    except Exception as e:
        out["error"] = f"torch.load failed: {e}"

    os.makedirs("diagnostics", exist_ok=True)
    with open("diagnostics/model_load.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote diagnostics/model_load.json")

if __name__ == "__main__":
    main()
