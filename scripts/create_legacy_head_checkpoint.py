#!/usr/bin/env python3
"""
Create a legacy-format RL checkpoint and optional meta JSON for the migration pytest.

Usage:
  python scripts/create_legacy_head_checkpoint.py \
    --out tests/data/legacy_rl_agent_head_scale.pth \
    --meta tests/data/legacy_rl_agent_head_scale_meta.json \
    --legacy-log-value 0.0

This emulates the older models that stored head_scale via exp(head_scale_log)
or a head_scale_alpha key; adjust --legacy-log-value to the old stored scalar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="tests/data/legacy_rl_agent_head_scale.pth",
        help="Output path for the legacy checkpoint",
    )
    parser.add_argument(
        "--meta",
        default="tests/data/legacy_rl_agent_head_scale_meta.json",
        help="Optional JSON metadata output path",
    )
    parser.add_argument(
        "--legacy-log-value",
        type=float,
        default=0.0,
        help="Legacy head_scale_log value (the old log parameter)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    meta_path = Path(args.meta)

    state: dict[str, object] = {}
    legacy_log_tensor = torch.tensor(args.legacy_log_value, dtype=torch.float32)
    state["head_scale_log"] = legacy_log_tensor
    state["head_scale_alpha"] = torch.tensor(0.0, dtype=torch.float32)
    state["model_state_dict"] = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(f"Wrote legacy checkpoint to {out_path}")

    try:
        legacy_effective = float(torch.exp(legacy_log_tensor).item())
    except Exception:  # pragma: no cover
        legacy_effective = None

    meta = {
        "effective_head_scale": legacy_effective,
        "legacy_log_value": args.legacy_log_value,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta JSON to {meta_path}")
    print("Legacy effective scale (approx):", legacy_effective)


if __name__ == "__main__":
    main()
