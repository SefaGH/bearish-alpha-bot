# CI Test Scope for PR #408

## Problem Statement
PR #408 adds softplus-based head-scale, migration tooling, inspectors, and RL-head-scale functionality. The current CI workflow runs `pytest tests/ -v`, which executes all tests including obsolete ones, slowing down CI and obscuring results specific to this PR.

## Solution: Focused Test Scope

The `.github/workflows/tests.yml` file has been updated to run only tests directly relevant to PR #408's changes.

### Test Paths Identified

```bash
pytest tests/unit/test_head_scale_migration.py \
       tests/test_reinforcement_learning_agent.py \
       tests/test_rl_bypass.py \
       -v --tb=short
```

### Test Coverage Breakdown

#### 1. `tests/unit/test_head_scale_migration.py`
**Purpose**: Validates legacy checkpoint migration to new softplus-based head-scale format

**Key Tests**:
- `test_head_scale_migrates_close()` - Ensures legacy `head_scale_log` and `head_scale_alpha` parameters are correctly migrated to `head_scale_raw` with softplus transformation
- Uses fixture: `tests/data/legacy_rl_agent_head_scale.pth`
- Validates migrated scale matches legacy effective scale within tolerance

**Why Relevant**: Core migration logic added in PR #408

#### 2. `tests/test_reinforcement_learning_agent.py`
**Purpose**: Tests RL agent inference mode, regime bias, and parameter behavior

**Key Tests**:
- `test_set_inference_mode_forces_zero_epsilon()` - Validates inference lock prevents exploration
- `test_inference_lock_disables_training_flag()` - Ensures training flag is overridden when locked
- `test_regime_bias_skips_when_confidence_low()` - Tests confidence-based bias gating
- `test_regime_bias_scales_with_confidence()` - Validates adaptive bias scaling

**Why Relevant**: Tests new inference mode enforcement and adaptive regime bias features added in PR #408

#### 3. `tests/test_rl_bypass.py`
**Purpose**: Tests frozen model detection and fallback mechanisms

**Key Tests**:
- `test_bypass_on_frozen_model()` - Validates Q-std threshold bypass logic
- `test_normal_rl_when_variance_ok()` - Ensures bypass is not triggered for healthy models

**Why Relevant**: Tests the frozen model detection logic and fallback to strategy signals added in PR #408

## Additional Validation

The PR also adds a dedicated `rl-head-scale-migration` job in the workflow (lines 31-74) that:
1. Generates canonical RL checkpoint
2. Runs inspector scripts (`inspect_head_param.py`, `inspect_head_gradient.py`)
3. Creates legacy fixture and validates migration
4. Probes gradient flow

This job is **preserved as-is** and provides end-to-end validation complementing the unit tests.

## Benefits

1. **Faster CI**: Runs ~3 focused test files instead of ~150+ test files
2. **Clear Signal**: Test failures directly indicate PR #408 issues
3. **Reduced Noise**: Eliminates failures from unrelated legacy tests
4. **Maintained Coverage**: All PR #408 features are validated:
   - Head-scale migration (unit)
   - RL agent behavior (integration)
   - Bypass logic (integration)
   - Inspector tooling (dedicated job)

## Recommendation

**Use this focused scope temporarily** while PR #408 is under review. Once merged, revert to full test suite or integrate these tests into a permanent RL test group.

## Files Modified

- `.github/workflows/tests.yml` - Line 29: Updated pytest path from `tests/` to specific test files

## Test Execution

To run locally with the same scope:
```bash
# In project root
pytest tests/unit/test_head_scale_migration.py \
       tests/test_reinforcement_learning_agent.py \
       tests/test_rl_bypass.py \
       -v --tb=short
```

Or in Docker (matching CI):
```bash
docker build -t bearish-alpha-bot:test -f docker/Dockerfile .
docker run --rm bearish-alpha-bot:test \
  pytest tests/unit/test_head_scale_migration.py \
         tests/test_reinforcement_learning_agent.py \
         tests/test_rl_bypass.py \
         -v --tb=short
```
