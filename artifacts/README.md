# Analysis Artifacts

This directory contains automated analysis results for the Bearish Alpha Bot repository, generated from the `main` branch.

## Generated Files

### 1. Dependency Graphs

#### `deps.svg`
- **Description**: Static dependency graph for main.py entry point
- **Tool**: pydeps with graphviz
- **Parameters**: `--max-bacon=3` (3 levels of dependencies)
- **Size**: ~27KB
- **Usage**: Visualize the main application's dependency structure

#### `deps_src.svg`
- **Description**: Static dependency graph for src/ package
- **Tool**: pydeps with graphviz
- **Parameters**: `--max-bacon=2` (2 levels of dependencies)
- **Size**: ~606 bytes
- **Usage**: Visualize the source package structure

### 2. Static Import Analysis

#### `import_frequency.txt`
- **Description**: Frequency analysis of static imports across the codebase
- **Method**: Regex scan of all Python files for `import` and `from` statements
- **Size**: ~4.4KB
- **Top Imports**:
  1. `sys` (69 occurrences)
  2. `os` (67 occurrences)
  3. `logging` (67 occurrences)
  4. `typing` (57 occurrences)
  5. `datetime` (56 occurrences)

### 3. Test Coverage Reports

#### `coverage.xml`
- **Description**: Machine-readable coverage report in Cobertura XML format
- **Tool**: coverage.py
- **Test Command**: `coverage run -m pytest tests/smoke_test.py -q`
- **Size**: ~790KB
- **Usage**: For CI/CD integration and automated analysis

#### `coverage_html/` (excluded from git)
- **Description**: Human-readable HTML coverage report
- **Location**: Generated locally, excluded from repository
- **Tool**: coverage.py
- **View**: Open `coverage_html/index.html` in a browser after regenerating
- **Command to regenerate**: `coverage html -d artifacts/coverage_html`

### 4. Runtime Import Audit

#### `import_audit.json`
- **Description**: Tracking of all modules imported during smoke test execution
- **Tool**: Custom import auditor (`tools/import_audit.py`)
- **Size**: ~112KB
- **Statistics**:
  - **Total unique modules**: 3,265
  - **Project modules**: 40
  - **Standard library**: 192
  - **External dependencies**: 2,912

**Key Project Modules Imported**:
- Core: `ccxt_client`, `market_regime`, `risk_manager`, `portfolio_manager`, etc.
- Strategies: `adaptive_ob`, `adaptive_str`, `oversold_bounce`, `short_the_rip`
- ML: `regime_predictor`, `price_predictor`, `strategy_optimizer`
- Scripts: `live_trading_launcher`

### 5. Unused File Candidates

#### `unused_candidates.txt`
- **Description**: List of potentially unused files based on combined analysis
- **Analysis Method**: Combines coverage data + runtime import audit + static analysis
- **Total Candidates**: 28 files
- **Size**: ~1.6KB

**Analysis Criteria**:
- `not_imported_runtime`: File was not imported during smoke test execution
- `coverage=X%`: File has low test coverage (< 10%)

**Top Candidates**:
1. `src/scanners/dca_advisor.py` - not imported
2. `src/scanners/arbitrage_scan.py` - not imported
3. `src/monitoring/performance_analytics.py` - not imported
4. `src/monitoring/dashboard.py` - not imported
5. `src/monitoring/alert_manager.py` - not imported
6. `src/core/portfolio_manager.py` - 8.2% coverage
7. `src/core/performance_monitor.py` - 9.6% coverage
8. `src/core/execution_analytics.py` - 8.9% coverage

**⚠️ Important Notes**:
- This is automated analysis - **manual review required** before any file removal
- Some files may be used in production scenarios not covered by smoke tests
- Low coverage doesn't necessarily mean unused, just not well-tested
- Scripts and entry points are excluded from unused analysis
- Test files are excluded (they're not meant to be imported)

## Regenerating Artifacts

To regenerate all artifacts:

```bash
# 1. Install dependencies
pip install coverage pytest pydeps graphviz
sudo apt-get install graphviz  # System package

# 2. Static import frequency
grep -rh "^\s*\(from\|import\)\s" --include="*.py" --exclude-dir=venv . > /tmp/imports_raw.txt
cat /tmp/imports_raw.txt | sed -E 's/^\s*(from|import)\s+([^ ]+).*/\2/' | sort | uniq -c | sort -nr > artifacts/import_frequency.txt

# 3. Dependency graphs
pydeps --noshow --show-deps --max-bacon=3 -o artifacts/deps.svg src/main.py
pydeps --noshow --show-deps --max-bacon=2 -o artifacts/deps_src.svg src

# 4. Coverage
pip install -r requirements.txt
coverage run -m pytest tests/smoke_test.py -q
coverage xml -o artifacts/coverage.xml
coverage html -d artifacts/coverage_html

# 5. Runtime import audit
python3 -c "import tools.import_audit; import pytest; pytest.main(['tests/smoke_test.py', '-q'])"

# 6. Unused candidates analysis
python3 tools/analyze_unused.py
```

## Tools Used

All analysis tools are located in the `tools/` directory:

- `tools/import_audit.py` - Runtime import tracker
- `tools/analyze_unused.py` - Unused file detection by combining multiple data sources

## Analysis Date

Generated: 2025-10-16

## Acceptance Criteria Status

All required artifacts have been successfully generated:

- ✅ `artifacts/deps.svg` - Dependency graph available
- ✅ `artifacts/coverage.xml` - Coverage XML report available
- ✅ `artifacts/coverage_html/` - Coverage HTML report available (excluded from git)
- ✅ `artifacts/import_audit.json` - Runtime import audit available
- ✅ `artifacts/import_frequency.txt` - Import frequency analysis available
- ✅ `artifacts/unused_candidates.txt` - Unused candidates list available

## Next Steps

1. **Review unused candidates**: Each file in `unused_candidates.txt` should be manually reviewed to determine if it's truly unused
2. **Improve test coverage**: Files with low coverage should have tests added
3. **Document intentional exclusions**: If a file appears unused but is needed for production, document why
4. **Create cleanup PR**: After review, create a separate PR to safely remove truly unused files
