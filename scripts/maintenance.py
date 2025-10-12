#!/usr/bin/env python3
from __future__ import annotations
import os, re, sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

def write_if_changed(p: Path, new: str) -> bool:
    old = read(p)
    if old == new:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(new, encoding="utf-8")
    return True

def replace_many(text: str, pairs: dict[str, str]) -> str:
    for a, b in pairs.items():
        text = text.replace(a, b)
    return text

def patch_standardize_secrets() -> list[Path]:
    changed = []
    # Hedef: workflow dosyalarında KEY/SECRET standardı, varsayılan kucoinfo değil burada
    mappings = {
        "BINGX_API_KEY": "BINGX_KEY",
        "BINGX_SECRET_KEY": "BINGX_SECRET",
        "BINANCE_API_KEY": "BINANCE_KEY",
        "BINANCE_SECRET_KEY": "BINANCE_SECRET",
        "BITGET_API_KEY": "BITGET_KEY",
        "BITGET_SECRET_KEY": "BITGET_SECRET",
        "KUCOIN_API_KEY": "KUCOIN_KEY",
        "KUCOIN_SECRET_KEY": "KUCOIN_SECRET",
        "KUCOINFUTURES_API_KEY": "KUCOIN_KEY",
        "KUCOINFUTURES_SECRET_KEY": "KUCOIN_SECRET",
    }
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        txt = read(wf)
        if not txt:
            continue
        new = replace_many(txt, mappings)
        if new != txt and write_if_changed(wf, new):
            changed.append(wf)
    return changed

def patch_docker_python_312() -> list[Path]:
    changed = []
    df = ROOT / "docker" / "Dockerfile"
    txt = read(df)
    if not txt:
        return changed
    new = re.sub(r"FROM\s+python:3\.\d+-slim", "FROM python:3.12-slim", txt)
    if write_if_changed(df, new):
        changed.append(df)
    return changed

def patch_backtest_imports() -> list[Path]:
    changed = []
    targets = [
        ROOT / "src" / "backtest" / "param_sweep.py",
        ROOT / "src" / "backtest" / "param_sweep_str.py",
    ]
    for p in targets:
        txt = read(p)
        if not txt:
            continue
        # Eski varyasyonları add_indicators alias'ına çevir
        new = re.sub(
            r"from\s+core\.indicators\s+import\s+enrich(?:\s+as\s+\w+)?",
            "from core.indicators import add_indicators as ind_enrich",
            txt,
        )
        new = re.sub(
            r"from\s+core\.indicators\s+import\s+add_indicators(?:(?!as ind_enrich).)*",
            "from core.indicators import add_indicators as ind_enrich",
            new,
            flags=re.MULTILINE,
        )
        if write_if_changed(p, new):
            changed.append(p)
    return changed

def patch_set_exchange_kucoinfutures() -> list[Path]:
    changed = []
    # Workflow'larda default 'bingx' → 'kucoinfutures'
    for wf in (ROOT / ".github" / "workflows").glob("*.yml"):
        txt = read(wf)
        if not txt:
            continue
        new = txt
        # '${{ secrets.EXECUTION_EXCHANGE || 'bingx' }}' → 'kucoinfutures'
        new = re.sub(r"\|\|\s*'bingx'\s*}}", "|| 'kucoinfutures' }}", new)
        # inputs default'ları (opsiyonel)
        new = new.replace("default: \"bingx\"", "default: \"kucoinfutures\"")
        if write_if_changed(wf, new) and new != txt:
            changed.append(wf)
    # env.example'da EXECUTION_EXCHANGE=bingx → kucoinfutures
    envf = ROOT / "env.example"
    txt = read(envf)
    if txt:
        new = txt.replace("EXECUTION_EXCHANGE=bingx", "EXECUTION_EXCHANGE=kucoinfutures")
        if write_if_changed(envf, new) and new != txt:
            changed.append(envf)
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=[
        "all", "standardize-secrets", "docker-python-312", "fix-backtest-imports", "set-exchange-kucoinfutures"
    ])
    args = ap.parse_args()

    changed: list[Path] = []
    if args.task in ("all", "standardize-secrets"):
        changed += patch_standardize_secrets()
    if args.task in ("all", "docker-python-312"):
        changed += patch_docker_python_312()
    if args.task in ("all", "fix-backtest-imports"):
        changed += patch_backtest_imports()
    if args.task in ("all", "set-exchange-kucoinfutures"):
        changed += patch_set_exchange_kucoinfutures()

    # Özet
    if changed:
        print("Patched files:")
        for p in sorted(set(changed)):
            print(" -", p.relative_to(ROOT))
        sys.exit(0)
    else:
        print("No changes were necessary (already up to date).")
        sys.exit(0)

if __name__ == "__main__":
    main()
