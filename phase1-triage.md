# Phase-1 Follow-up Triage Report

**Generated:** 2025-11-11  
**Branch:** copilot/fix-pytest-collection-errors  
**Status:** Pre-implementation analysis

---

## 🎯 Executive Summary

This triage report identifies issues introduced or exposed by Phase-1 merge that prevent pytest from collecting and running tests successfully. Issues are categorized by type and prioritized for remediation.

**Key Findings:**
- 156 file operations missing `encoding="utf-8"` specification
- 3 files with import-time `sys.path` modifications
- 29 files with module-level `logging.basicConfig()` calls
- Multiple files with unconditional imports of optional dependencies (torch, sklearn, xgboost, optuna)
- Diagnostics script lacks requirements.txt awareness

---

## 📊 Issue Categories

### 1. Encoding Issues
**Severity:** HIGH  
**Impact:** Pytest collection failures on systems with non-UTF-8 default encoding  
**Count:** 156 file operations

**Affected Files:**
- `src/api/prediction_service.py`
- `src/backtest/risk_aware_backtest.py`
- `src/monitoring/dashboard.py`
- `src/monitoring/performance_analytics.py`
- `src/ml/experience_replay.py`
- `src/core/system_info.py`
- `src/core/bingx_websocket.py`
- `src/config/live_trading_config.py`
- `tests/test_symbol_parameter_passing.py`
- `tests/test_health_report_logging.py`
- Many more...

**Root Cause:**
File operations using `open()` without explicit `encoding="utf-8"` parameter rely on system default encoding, which may not be UTF-8 in CI environments or Windows systems.

**Fix Strategy:**
Add `encoding="utf-8"` to all text file operations (read/write mode 'r', 'w', 'a'). Binary operations ('rb', 'wb') should remain unchanged.

---

### 2. Import-Time Side Effects
**Severity:** HIGH  
**Impact:** Module import causes unexpected behavior, breaks test isolation, conflicts with pytest collection  
**Count:** 3 files with sys.path modification, 29+ with logging.basicConfig

**Critical Files:**

#### A. sys.path Modifications (BLOCKER)
- `src/backtest/param_sweep.py` - Line 11: `sys.path.append(str(Path(__file__).parent.parent))`
- `src/backtest/param_sweep_str.py` - Similar issue
- `src/core/production_coordinator.py` - Line with sys.path.append

**Impact:** Modifying sys.path at import time affects the entire Python environment, can cause import conflicts, and breaks test isolation.

#### B. logging.basicConfig at Module Level (HIGH)
- `src/backtest/param_sweep.py` - Lines 24-29
- 28+ other files

**Impact:** First module imported configures logging globally, subsequent modules cannot reconfigure. Breaks test isolation and logging control.

**Fix Strategy:**
1. Remove sys.path modifications - use proper package structure instead
2. Move logging.basicConfig calls into `if __name__ == "__main__":` blocks or initialization functions
3. For modules meant to be imported, use `logger = logging.getLogger(__name__)` without basicConfig

---

### 3. Optional Dependencies (Missing Graceful Handling)
**Severity:** MEDIUM  
**Impact:** ImportError on missing optional packages breaks test collection  
**Count:** 15+ files

**Unconditional Imports Found:**
- **torch** (PyTorch):
  - `src/api/prediction_service.py`
  - `src/ml/model_trainer.py`
  - `src/ml/models.py`
  - `src/ml/rl_model_trainer.py`
  
- **sklearn** (scikit-learn):
  - `src/ml/model_trainer.py`
  - Multiple other files

- **xgboost, optuna**: Various ML files

**Conditional Imports (Good Examples):**
- `src/ml/regime_predictor.py` - Uses try/except for torch
- `src/ml/neural_networks.py` - Uses try/except for torch

**Fix Strategy:**
1. Wrap optional imports in try/except blocks
2. Provide graceful degradation or meaningful error messages
3. Mark tests requiring optional deps with `@pytest.mark.skipif` or `pytest.importorskip()`
4. Document which features require which optional dependencies

**Example Pattern:**
```python
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # For type checking

# In tests:
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
```

---

### 4. Import/Packaging Issues
**Severity:** MEDIUM  
**Impact:** Import errors, inconsistent module resolution  
**Count:** Multiple files with conditional import patterns

**Problem Files:**
- `src/core/position_manager.py` - Has complex conditional import logic
- `src/core/production_coordinator.py` - Multiple import strategies
- `src/core/risk_manager.py` - Dual import paths
- `src/core/realtime_risk.py` - Conditional imports

**Current Pattern (Problematic):**
```python
try:
    # Option 1: Direct import (scripts add src/ to sys.path)
    from utils.pnl_calculator import calculate_pnl
except ImportError:
    try:
        # Option 2: Absolute import (repo root on sys.path)
        from src.utils.pnl_calculator import calculate_pnl
    except ImportError:
        # Fallback...
```

**Root Cause:**
Inconsistent PYTHONPATH setup between different execution contexts (main.py, tests, scripts, standalone modules).

**Fix Strategy:**
1. Ensure `src/` is properly configured as a package (has __init__.py)
2. Set PYTHONPATH consistently in test configuration (pytest.ini or conftest.py)
3. Use consistent import style throughout (prefer absolute imports from package root)
4. Add minimal shim in conftest.py if needed (with warning)

---

### 5. Diagnostics Issues
**Severity:** LOW  
**Impact:** False warnings in health check reports  
**Count:** 1 file

**Affected File:**
- `scripts/pre_gemma_health_check.py`

**Current Behavior:**
- Only checks `pyproject.toml` for dependencies
- Does not recognize `requirements.txt` (which is the primary dependency specification)
- Generates false warnings about missing dependencies that are actually in requirements.txt

**Required Enhancement:**
Add `--dep-source {auto,requirements,pyproject}` option:
- `auto` (default): Check requirements.txt first, fallback to pyproject.toml
- `requirements`: Only parse requirements.txt
- `pyproject`: Only parse pyproject.toml

**Implementation:**
```python
def load_dependency_names(source: str = "auto") -> set[str]:
    if source == "auto":
        req_file = REPO_ROOT / "requirements.txt"
        if req_file.exists():
            return parse_requirements_txt(req_file)
        return parse_pyproject_toml()
    elif source == "requirements":
        return parse_requirements_txt(REPO_ROOT / "requirements.txt")
    else:  # pyproject
        return parse_pyproject_toml()
```

---

## 🗺️ Remediation Roadmap

### Phase 1: Critical Blockers (Week 1)
1. **Encoding Fixes** - Add `encoding="utf-8"` to all file operations
2. **Param Sweep Import-Time Side Effects** - Remove sys.path modifications, move logging.basicConfig

### Phase 2: Core Functionality (Week 1-2)
3. **Import/Packaging Fixes** - Establish consistent PYTHONPATH and import patterns
4. **Optional Dependencies** - Add graceful handling for torch, sklearn, etc.

### Phase 3: Improvements (Week 2)
5. **Diagnostics Enhancement** - Add requirements.txt awareness
6. **CI Split** - Separate core vs extended test suites

### Phase 4: Documentation (Week 2)
7. **Retrospective** - Document all changes and lessons learned

---

## 📋 Detailed File Lists

### Encoding Issues - High Priority Files
```
src/api/prediction_service.py (1 occurrence)
src/backtest/risk_aware_backtest.py (1 occurrence)
src/monitoring/dashboard.py (2 occurrences)
src/monitoring/performance_analytics.py (2 occurrences)
src/ml/experience_replay.py (2 occurrences - binary mode OK)
src/core/system_info.py (1 occurrence)
src/config/live_trading_config.py (2 occurrences)
tests/test_symbol_parameter_passing.py (2 occurrences)
tests/test_health_report_logging.py (5 occurrences)
... (146 more files to audit)
```

### Import-Time Side Effects - Critical Files
```
src/backtest/param_sweep.py - sys.path.append + logging.basicConfig
src/backtest/param_sweep_str.py - sys.path.append + logging.basicConfig
src/core/production_coordinator.py - sys.path.append
src/ml/price_predictor.py - sys.path.insert
```

### Optional Dependency Files
```
Unconditional torch imports:
- src/api/prediction_service.py
- src/ml/model_trainer.py
- src/ml/models.py
- src/ml/rl_model_trainer.py

Good examples (has try/except):
- src/ml/regime_predictor.py
- src/ml/neural_networks.py
- src/ml/reinforcement_learning.py
```

---

## ✅ Success Criteria

### Test Collection
- [ ] `pytest --collect-only` succeeds without errors
- [ ] All test files are discovered and collected
- [ ] No import-time side effects cause collection failures

### Test Execution
- [ ] Core test suite (unit + integration) passes on CI
- [ ] Tests with optional dependencies are properly skipped when deps missing
- [ ] No encoding-related failures on any platform

### Code Quality
- [ ] ruff passes on all Python files
- [ ] mypy passes with current configuration
- [ ] No test bypasses (xfail/skip) except for genuine optional features

### Diagnostics
- [ ] pre_gemma_health_check.py recognizes requirements.txt
- [ ] No false positive warnings about missing dependencies
- [ ] Report accurately reflects repository state

---

## 🔧 Next Steps

1. **Create branch:** `agent/fix-encoding` for encoding fixes
2. **Create branch:** `agent/param-sweep-guard` for import-time side effects
3. **Create branch:** `agent/optional-deps` for graceful dependency handling
4. **Create branch:** `agent/package-shims` for import/packaging fixes
5. **Create branch:** `agent/diagnostics-dep-source` for diagnostics enhancement

Each branch will have focused PRs with:
- Minimal, surgical changes
- Tests to prevent regression
- No behavior changes for end users
- Clear documentation of changes

---

## 📊 Estimated Effort

| Task | Effort | Risk | Priority |
|------|--------|------|----------|
| Encoding Fixes | 4h | Low | P0 |
| Param Sweep Import-Time | 2h | Medium | P0 |
| Optional Deps | 6h | Medium | P1 |
| Import/Packaging | 4h | High | P1 |
| Diagnostics Enhancement | 3h | Low | P2 |
| CI Split | 2h | Low | P2 |
| Documentation | 2h | Low | P3 |

**Total Estimated Effort:** 23 hours over 2 weeks

---

## 🎯 Success Metrics

- Core CI (pytest, ruff, mypy) passing: **Target 100%**
- Test coverage maintained or improved: **No decrease**
- Zero test bypasses in core suite: **Strict**
- False positive warnings eliminated: **100%**
- Documentation completeness: **All changes documented**

---

## 📝 Notes

- **Python Version:** Repository requires Python 3.11 only (not 3.12+) due to aiohttp 3.8.6 compatibility
- **CI Configuration:** All workflows must use `python-version: "3.11"`
- **Testing Environment:** This analysis was performed with Python 3.12 due to environment limitations; actual testing will occur with Python 3.11 in CI

---

**Report Status:** ✅ Complete  
**Next Action:** Begin Task 2 (Encoding Fixes) on branch `agent/fix-encoding`
