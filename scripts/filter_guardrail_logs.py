import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

INTEGRITY_PATTERNS = [
    r"IntegrityGuard",
    r"integrity_guard",
    r"integrity_data_unavailable",
    r"stale_signal_",
    r"stale_candle_",
    r"price_deviation_",
    r"convert_reverse_to_close",
]

REGIME_PATTERNS = [
    r"RegimeFilter",
    r"regime_filter",
    r"regime_weight",
    r"regime_veto_",
    r"low_regime_weight_",
]

TRANSITION_PATTERNS = [
    r"transition_policy",
    r"TransitionPolicy",
    r"AUTO-REVERSE\] Blocked reverse, converted to close",
    r"CONFLICT-RESOLUTION\] Reverse blocked, converted to close",
    r"convert_to_close",
    r"reverse blocked",
]

CONFLICT_PATTERNS = [
    r"CONFLICT-RESOLUTION",
    r"conflict_resolution",
]


def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _match_any(line: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(line) for p in patterns)


def _iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\n")


def filter_lines(lines: Iterable[str]) -> List[Tuple[str, str]]:
    integrity_re = _compile(INTEGRITY_PATTERNS)
    regime_re = _compile(REGIME_PATTERNS)
    transition_re = _compile(TRANSITION_PATTERNS)
    conflict_re = _compile(CONFLICT_PATTERNS)

    out: List[Tuple[str, str]] = []
    for line in lines:
        if _match_any(line, integrity_re):
            out.append(("INTEGRITY", line))
            continue
        if _match_any(line, regime_re):
            out.append(("REGIME", line))
            continue
        if _match_any(line, transition_re):
            out.append(("TRANSITION", line))
            continue
        if _match_any(line, conflict_re):
            out.append(("CONFLICT", line))
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter guardrail-related lines from a trading log")
    parser.add_argument("log_path", help="Path to live trading log file")
    parser.add_argument("--out", help="Optional output file to write filtered lines")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    results = filter_lines(_iter_lines(log_path))

    formatted = [f"[{tag}] {line}" for tag, line in results]

    if args.out:
        out_path = Path(args.out)
        out_path.write_text("\n".join(formatted), encoding="utf-8")
    else:
        for line in formatted:
            print(line)

    print(f"\nMatched lines: {len(formatted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
