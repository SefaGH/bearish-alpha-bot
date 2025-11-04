# GitHub Copilot Instructions for Bearish Alpha Bot

## 🚨 CRITICAL PYTHON VERSION REQUIREMENT 🚨

**⚠️ MANDATORY: This repository REQUIRES Python 3.11**

- ✅ **ONLY** Python 3.11.x is SUPPORTED
- ❌ **NEVER USE** Python 3.12, 3.10, or any other version
- ❌ **DO NOT CHANGE** any Python version specifications to anything other than 3.11

## Why Python 3.11 Only?

This project requires `aiohttp==3.8.6` for ccxt.pro WebSocket functionality. 
The aiohttp 3.8.6 package does not compile on Python 3.12+ due to internal API changes 
(specifically, the `PyLongObject.ob_digit` structure was changed).

**ANY attempt to use Python 3.12+ will result in build failures and broken dependencies.**

## 📋 Required Setup in ALL GitHub Actions Workflows

### ✅ MANDATORY: Use This Exact Configuration

When creating or modifying ANY GitHub Actions workflow file, you **MUST** use:

```yaml
- name: Setup Python 3.11
  uses: actions/setup-python@v5
  with:
    python-version: "3.11"  # ⚠️ MANDATORY - DO NOT CHANGE TO 3.12 OR ANY OTHER VERSION
    python-version-file: ".python-version"  # Optional: reads from .python-version file
```

### ❌ FORBIDDEN: Do NOT Use

```yaml
# ❌ WRONG - DO NOT USE Python 3.12
- uses: actions/setup-python@v5
  with:
    python-version: "3.12"

# ❌ WRONG - DO NOT USE Python 3.10
- uses: actions/setup-python@v5
  with:
    python-version: "3.10"

# ❌ WRONG - DO NOT USE matrix with multiple Python versions
strategy:
  matrix:
    python-version: ["3.11", "3.12"]  # Only 3.11 is allowed
```

## 🔍 Before Creating/Modifying Workflows

1. **ALWAYS** check `.python-version` file (should contain `3.11`)
2. **ALWAYS** check `pyproject.toml` (should have `requires-python = ">=3.11,<3.12"`)
3. **ALWAYS** verify that `setup-python` action uses `python-version: "3.11"`
4. **NEVER** assume the runner's default Python version is correct

## 📝 Testing and CI/CD Guidelines

When running tests or creating GitHub Actions workflows:

1. **ALWAYS** explicitly set `python-version: "3.11"` in `setup-python` action
2. **NEVER** rely on runner's default Python version
3. **ALWAYS** verify Python version at the start of the job:
   ```yaml
   - name: Verify Python version
     run: |
       python --version  # Should output: Python 3.11.x
       python -c "import sys; assert sys.version_info[:2] == (3, 11), f'Wrong Python version: {sys.version}'"
   ```
4. **NEVER** use Python 3.12 or higher

## 🐳 Docker Configuration

The project Docker image uses `python:3.11-slim` as the base image.

**DO NOT change this to:**
- ❌ `python:3.12-slim`
- ❌ `python:3-slim` (uses latest Python 3.x)
- ❌ `python:latest` (uses latest Python version)

## 📦 Development Setup

To set up the development environment:

```bash
# 1. Verify Python 3.11 is installed
python --version  # MUST show Python 3.11.x

# 2. If Python 3.12 is active, switch to 3.11:
# Using pyenv:
pyenv install 3.11
pyenv local 3.11

# Using conda:
conda create -n bearish-bot python=3.11
conda activate bearish-bot

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔑 Key Dependencies Requiring Python 3.11

- `aiohttp==3.8.6` - WebSocket support for ccxt.pro (BREAKS on Python 3.12+)
- `yarl<2.0` - Required for aiohttp 3.8.x
- `multidict<7.0` - Required for aiohttp 3.8.x

## 📄 Configuration File References

This repository contains multiple files that enforce Python 3.11:

1. **`.python-version`** - Contains `3.11` (for pyenv/asdf)
2. **`runtime.txt`** - Contains `python-3.11` (for deployment platforms)
3. **`pyproject.toml`** - Has `requires-python = ">=3.11,<3.12"`
4. **`requirements.txt`** - Contains detailed Python 3.11 requirement explanation
5. **`Dockerfile`** - Uses `python:3.11-slim`
6. **`.github/copilot-config.yml`** - Copilot configuration for Python 3.11

## ✅ Validation Workflow

A validation workflow (`.github/workflows/python-version-check.yml`) automatically checks:
- All workflow files use Python 3.11
- Configuration files are correct
- No Python 3.12+ references exist

## 🎯 Summary: What You MUST Remember

1. **Python 3.11 ONLY** - No exceptions
2. **ALWAYS** use `python-version: "3.11"` in workflows
3. **NEVER** use Python 3.12+
4. **CHECK** `.python-version` before creating workflows
5. **VERIFY** Python version in every CI/CD job

## 🚫 Common Mistakes to Avoid

- ❌ Using runner's default Python (might be 3.12)
- ❌ Using `python-version: "3.x"` (gets latest Python 3.x)
- ❌ Using `python-version: "3"` (gets latest Python 3.x)
- ❌ Assuming Python 3.12 is compatible
- ❌ Not explicitly setting python-version in setup-python

## ✅ Correct Examples

```yaml
# Example 1: Explicit Python 3.11
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"

# Example 2: Using .python-version file
- uses: actions/setup-python@v5
  with:
    python-version-file: ".python-version"

# Example 3: With version verification
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
- run: python --version  # Verify it's 3.11.x
```
