#!/usr/bin/env python3
"""Ensure MLStrategyIntegrationManager.get_ml_context is awaited."""
from __future__ import annotations

from pathlib import Path
import re
import sys

TARGET = Path(__file__).resolve().parents[1] / "src" / "core" / "strategy_coordinator.py"


def main() -> int:
    if not TARGET.exists():
        print(f"Target file not found: {TARGET}", file=sys.stderr)
        return 1

    text = TARGET.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(ml_context\s*=\s*)(self\.ml_integration\.get_ml_context\([^\n]+\))"
    )

    if "await self.ml_integration.get_ml_context" in text:
        print("Await already present in strategy_coordinator.py")
        return 0

    new_text, count = pattern.subn(r"\1await \2", text, count=1)
    if count == 0:
        print("Pattern not found; no changes applied", file=sys.stderr)
        return 1

    TARGET.write_text(new_text, encoding="utf-8")
    print("Inserted await before get_ml_context call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
