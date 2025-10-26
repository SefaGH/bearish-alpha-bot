#!/usr/bin/env bash
# diagnostics/run_repo_checks.sh
# Usage: ./diagnostics/run_repo_checks.sh [OUTDIR]
# Default OUTDIR=diagnostics
set -euo pipefail

OUTDIR="${1:-diagnostics}"
mkdir -p "$OUTDIR"

echo "Repository diagnostics run at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" > "$OUTDIR/00_header.txt"

# 00 - list candidate ML directories
{
  echo "== ls -la src/ml (if exists) =="
  ls -la src/ml 2>/dev/null || echo "src/ml: not found"
  echo ""
  echo "== ls -la ml (if exists) =="
  ls -la ml 2>/dev/null || echo "ml: not found"
  echo ""
} > "$OUTDIR/00_ls_ml.txt"

# 01 - find unconditional ML/torch/sklearn imports across the repo
# Use multiple safe greps to capture different patterns
echo "== grep for ML-related imports (torch, torchvision, sklearn, scikit_learn, 'from ml.' , 'import ml') ==" > "$OUTDIR/01_ml_imports.txt"

# Patterns for torch/torchvision/sklearn
git grep -nE "^\s*(from|import)\s+(torch|torchvision|sklearn|scikit_learn)\b" -- . 2>/dev/null || true >> "$OUTDIR/01_ml_imports.txt"
# Patterns for "from ml." modules or "import ml"
git grep -n "from ml\." -- . 2>/dev/null || true >> "$OUTDIR/01_ml_imports.txt"
git grep -n "import ml" -- . 2>/dev/null || true >> "$OUTDIR/01_ml_imports.txt"

# 02 - find files referencing requirements.txt (any path)
echo "== grep for 'requirements.txt' references ==" > "$OUTDIR/02_requirements_refs.txt"
git grep -n "requirements.txt" -- . 2>/dev/null || true >> "$OUTDIR/02_requirements_refs.txt"

# 03 - show all requirements* files tracked in the repo
echo "== list requirements* files in the repo ==" > "$OUTDIR/06_requirements_files.txt"
git ls-files '*requirements*.txt' 2>/dev/null || true >> "$OUTDIR/06_requirements_files.txt"

# 04 - search .github/workflows for setup-python / python-version uses
echo "== search .github/workflows for setup-python / python-version ==" > "$OUTDIR/03_setup_python_usage.txt"
git grep -n "uses: actions/setup-python" .github/workflows 2>/dev/null || true >> "$OUTDIR/03_setup_python_usage.txt"
git grep -n "python-version" .github/workflows 2>/dev/null || true >> "$OUTDIR/03_setup_python_usage.txt"

# 05 - look for occurrences of --no-cache-dir in repo (workflow scripts / docker)
echo "== grep for --no-cache-dir occurrences ==" > "$OUTDIR/05_no_cache_dir_refs.txt"
git grep -n -- "--no-cache-dir" -- . 2>/dev/null || true >> "$OUTDIR/05_no_cache_dir_refs.txt"

# 06 - find Dockerfile python bases
echo "== grep for FROM python in repo (Dockerfiles) ==" > "$OUTDIR/04_docker_python_bases.txt"
git grep -n -E "^FROM[[:space:]]+python" -- . 2>/dev/null || true >> "$OUTDIR/04_docker_python_bases.txt"

# 07 - list any files that import 'ml.' modules (explicit)
echo "== files that import 'ml.' modules ==" > "$OUTDIR/07_import_ml_dot.txt"
git grep -n "from ml\." -- . 2>/dev/null || true >> "$OUTDIR/07_import_ml_dot.txt"

# 08 - quick list of top-level scripts that may be used as entrypoints
echo "== list scripts/ and top-level python files ==" > "$OUTDIR/08_entrypoint_files.txt"
(ls -la scripts 2>/dev/null || echo "scripts/ not found") >> "$OUTDIR/08_entrypoint_files.txt"
git ls-files '*.py' | sed -n '1,200p' >> "$OUTDIR/08_entrypoint_files.txt" || true

# 09 - create a short README with what we expect next
cat > "$OUTDIR/99_README.txt" <<'EOF'
Files created by diagnostics/run_repo_checks.sh:
- 00_ls_ml.txt                : directory listings for src/ml and ml
- 01_ml_imports.txt           : grep results for torch/sklearn/ml imports
- 02_requirements_refs.txt    : files referencing requirements.txt
- 03_setup_python_usage.txt   : workflow searches for setup-python / python-version
- 04_docker_python_bases.txt  : Dockerfiles that use a python base image
- 05_no_cache_dir_refs.txt    : uses of --no-cache-dir in repo
- 06_requirements_files.txt   : list of requirements* files in the repo
- 07_import_ml_dot.txt        : files importing from ml.*
- 08_entrypoint_files.txt     : scripts/ directory listing and python files
- 99_README.txt               : this help text

Next steps:
1) Upload the diagnostics/*.txt files here so I can inspect exact matches.
2) If you want me to prepare PR patches, also upload src/ml/* (or the grep results showing which ml modules exist).
3) If the repo is large, compress diagnostics/ into diagnostics.tar.gz and upload that.

Run: ./diagnostics/run_repo_checks.sh
EOF

echo "Diagnostics written to $OUTDIR (files:)"
ls -1 "$OUTDIR" | sed -e 's/^/  - /'
