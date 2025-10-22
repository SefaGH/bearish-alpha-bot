# GitHub Copilot Instructions for Bearish Alpha Bot

## Python Version Requirement

**CRITICAL: This project MUST use Python 3.11**

- ✅ Python 3.11.x is REQUIRED
- ❌ Python 3.12+ is NOT SUPPORTED

### Why Python 3.11 Only?

This project requires `aiohttp==3.8.6` for ccxt.pro WebSocket functionality. 
The aiohttp 3.8.6 package does not compile on Python 3.12+ due to internal API changes 
(specifically, the `PyLongObject.ob_digit` structure was changed).

### For Testing and CI/CD

When running tests or creating GitHub Actions workflows:
- Always use `python-version: "3.11"` in setup-python actions
- Never use Python 3.12 or higher
- Verify Python version before running tests: `python --version`

### Development Setup

To set up the development environment:
```bash
# Ensure Python 3.11 is installed
python --version  # Should show Python 3.11.x

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies Requiring Python 3.11

- `aiohttp==3.8.6` - WebSocket support for ccxt.pro
- `yarl<2.0` - Required for aiohttp 3.8.x
- `multidict<7.0` - Required for aiohttp 3.8.x

### Workflow Configuration

All GitHub Actions workflows should include:
```yaml
- uses: actions/setup-python@v4
  with:
    python-version: "3.11"
```

### Docker

The project Docker image uses `python:3.11-slim` as the base image.
Do not change this to a newer Python version.

### References

- `pyproject.toml`: `requires-python = ">=3.11,<3.12"`
- `requirements.txt`: Contains detailed explanation of version requirements
- `Dockerfile`: Uses `python:3.11-slim`
- `.python-version`: Specifies Python 3.11 for pyenv/asdf users
