#!/usr/bin/env bash
# diagnostics/parse_action_logs.sh
# Usage: ./diagnostics/parse_action_logs.sh [LOGDIR] [OUTDIR]
# Default LOGDIR=action_logs (place your raw GitHub Actions log files there)
# Default OUTDIR=diagnostics
set -euo pipefail

LOGDIR="${1:-action_logs}"
OUTDIR="${2:-diagnostics}"
mkdir -p "$OUTDIR"

echo "Action log parsing run at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" > "$OUTDIR/00_action_logs_header.txt"

# Find all text files in LOGDIR and scan for python setup messages and python -V lines
echo "== Summary of Python setup lines in action logs ==" > "$OUTDIR/action_logs_summary.txt"

if [ -d "$LOGDIR" ]; then
  find "$LOGDIR" -type f -name '*.txt' -o -name '*.log' | while read -r f; do
    echo "---- LOG: $f ----" >> "$OUTDIR/action_logs_summary.txt"
    # show lines that indicate CPython setup
    grep -n -E "Successfully set up CPython|set up CPython|Setup python|setup-python|python-version|python -V|python -V" "$f" 2>/dev/null || true >> "$OUTDIR/action_logs_summary.txt"
    # also capture any pip wheel / download big file lines that mention torch
    grep -n -E "Downloading .*torch|Downloading .*triton|Downloading .*nvidia|torch-.*\.whl" "$f" 2>/dev/null || true >> "$OUTDIR/action_logs_summary.txt"
    echo "" >> "$OUTDIR/action_logs_summary.txt"
  done
else
  echo "Log directory '$LOGDIR' not found. Create it and put GitHub Actions log files there (e.g. downloaded via 'Download logs' from Actions UI) and re-run this script." >> "$OUTDIR/action_logs_summary.txt"
fi

echo "Parsed action logs summary written to $OUTDIR/action_logs_summary.txt"
