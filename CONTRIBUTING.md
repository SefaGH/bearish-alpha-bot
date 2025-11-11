# Contributing to Bearish Alpha Bot

Thank you for your interest in contributing to the Bearish Alpha Bot project! This document provides guidelines and instructions for contributing.

## 🐍 Python Version Requirement

**⚠️ CRITICAL: This project REQUIRES Python 3.11**

- ✅ **ONLY** Python 3.11.x is supported
- ❌ **DO NOT USE** Python 3.12, 3.10, or any other version
- ❌ **DO NOT CHANGE** any Python version specifications to anything other than 3.11

### Why Python 3.11 Only?

This project requires `aiohttp==3.8.6` for ccxt.pro WebSocket functionality. The aiohttp 3.8.6 package does not compile on Python 3.12+ due to internal API changes.

**ANY attempt to use Python 3.12+ will result in build failures and broken dependencies.**

## 🚀 Getting Started

### Prerequisites

- Python 3.11 (required)
- Git
- pip (latest version)
- Virtual environment tool (venv, virtualenv, or conda)

### Setting Up Development Environment

#### 1. Install Python 3.11

**Using pyenv (recommended):**
```bash
pyenv install 3.11
pyenv local 3.11
```

**Using conda:**
```bash
conda create -n bearish-bot python=3.11
conda activate bearish-bot
```

**Verify Python version:**
```bash
python --version  # Should output: Python 3.11.x
```

#### 2. Clone the Repository

```bash
git clone https://github.com/SefaGH/bearish-alpha-bot.git
cd bearish-alpha-bot
```

#### 3. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-asyncio pytest-timeout pytest-mock pytest-cov
pip install ruff mypy bandit
```

#### 5. Verify Installation

```bash
# Verify aiohttp 3.8.6 installation (CRITICAL)
python -c "import aiohttp; print(f'aiohttp {aiohttp.__version__}')"

# Verify ccxt.pro availability
python -c "import ccxt.pro; print('ccxt.pro OK')"

# Run smoke tests
pytest tests/smoke_test.py -v
```

## 🧪 Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/test_specific.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Run integration tests (requires API keys)
pytest tests/integration/ -v -s

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Phase-Specific Tests

```bash
# Phase 1 tests
pytest tests/test_phase1_integration.py -v

# Phase 2 tests
pytest tests/test_phase2_*.py -v

# Phase 3 tests
pytest tests/test_phase3_*.py -v
```

## 🎨 Code Style

### Linting with Ruff

```bash
# Check code style
ruff check src/

# Auto-fix issues
ruff check src/ --fix

# Format code
ruff format src/
```

### Type Checking with mypy

```bash
# Run mypy
mypy src/

# Run mypy with strict mode
mypy src/ --strict
```

### Security Scanning with Bandit

```bash
# Run security checks
bandit -r src/

# Generate report
bandit -r src/ -f html -o security_report.html
```

## 📝 Code Guidelines

### Python Style

- Follow PEP 8 style guide
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions small and focused
- Use meaningful variable names

### File Encoding

- **ALWAYS** use UTF-8 encoding for text files
- Add `encoding='utf-8'` parameter to `open()` calls for text files
- Binary files (wb/rb mode) don't need encoding parameter

**Example:**
```python
# ✅ Good
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# ❌ Bad
with open('file.txt', 'r') as f:  # May cause encoding issues
    content = f.read()
```

### Type Hints

- Add type hints to function parameters and return types
- Use `Optional[Type]` for nullable values
- Use `Union[Type1, Type2]` for multiple possible types
- Avoid `# type: ignore` unless absolutely necessary
- Document why `# type: ignore` is needed if used

### Test Markers

Available pytest markers (defined in `pyproject.toml`):
- `@pytest.mark.unit` - Unit tests (fast, can use mocks)
- `@pytest.mark.integration` - Integration tests (slow, needs all dependencies)
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.phase3` - ML/AI features (Phase 3)
- `@pytest.mark.slow` - Tests that take more than 5 seconds

### Optional Dependencies

Some tests require optional dependencies:
- **ML/Training tests:** scikit-learn, tensorflow, torch
- **Monitoring tests:** prometheus-client, grafana-api
- **Websocket tests:** websocket-client

Tests requiring optional dependencies should:
1. Use appropriate pytest markers
2. Skip gracefully if dependency missing
3. Document the requirement in test docstring

**Example:**
```python
import pytest

@pytest.mark.integration
def test_ml_model():
    """Test ML model training.
    
    Requires: scikit-learn, tensorflow
    """
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("tensorflow not installed")
    
    # Test code here
```

## 🔀 Git Workflow

### Branch Naming

- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`
- Housekeeping: `housekeeping/description`
- Agent work: `agent/description`

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(ml): add GEMMA model integration
fix(websocket): resolve connection timeout issue
docs(readme): update Python 3.11 requirement
test(unit): add tests for signal generation
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linters locally
4. Commit your changes with descriptive messages
5. Push your branch to GitHub
6. Create a Pull Request
7. Address review feedback
8. Wait for CI checks to pass
9. Request review from maintainers

### Pull Request Checklist

- [ ] Python 3.11 verified
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] Ruff linting passes
- [ ] Mypy type checking passes
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts

## 🔒 Security

### Reporting Security Issues

If you discover a security vulnerability, please email the maintainers directly. Do not create a public issue.

### Security Guidelines

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Review code for security vulnerabilities
- Run bandit security scanner before submitting PR

## 📚 Documentation

### Code Documentation

- Add docstrings to public functions and classes
- Use Google-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

**Example:**
```python
def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float
) -> float:
    """Calculate position size based on risk management rules.
    
    Args:
        balance: Account balance in USD
        risk_percent: Risk percentage (0.01 = 1%)
        entry_price: Entry price in USD
        stop_loss: Stop loss price in USD
    
    Returns:
        Position size in contracts
    
    Example:
        >>> calculate_position_size(10000, 0.01, 50000, 49000)
        5.0
    """
    risk_amount = balance * risk_percent
    price_risk = abs(entry_price - stop_loss)
    return risk_amount / price_risk
```

### README Updates

When adding new features:
- Update the Features section
- Add usage examples
- Update configuration options
- Add troubleshooting tips if needed

## 🐛 Bug Reports

### Before Reporting

1. Check if the issue already exists
2. Verify Python 3.11 is being used
3. Try latest version from `main` branch
4. Reproduce the issue with minimal code

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version: 3.11.x
- OS: 
- Dependencies: (output of `pip list`)

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why this feature is needed

## Proposed Solution
How this could be implemented

## Alternatives Considered
Other approaches considered

## Additional Context
Any other relevant information
```

## 🎯 Current Priorities

### Phase 2 Focus Areas

1. **GEMMA Model Integration**
   - Model adapter implementation
   - Inference optimization
   - Performance benchmarking

2. **Monitoring Enhancements**
   - Monitoring hooks
   - Alert system improvements
   - Performance metrics

3. **Test Coverage**
   - Phase 2 specific tests
   - Integration test improvements
   - Coverage target: >80%

4. **Documentation**
   - API documentation
   - Architecture diagrams
   - Tutorial improvements

## 📞 Getting Help

- **Issues:** [GitHub Issues](https://github.com/SefaGH/bearish-alpha-bot/issues)
- **Discussions:** [GitHub Discussions](https://github.com/SefaGH/bearish-alpha-bot/discussions)
- **Documentation:** [README.md](README.md)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Acknowledgments

Thank you for contributing to Bearish Alpha Bot! Your contributions help make this project better for everyone.

---

**Last Updated:** 2025-11-11
**Python Version:** 3.11 (REQUIRED)
