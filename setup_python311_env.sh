#!/bin/bash
# Python 3.11 Integration Test Environment Setup Script
# This script sets up a clean Python 3.11 environment for running integration tests
# Addresses Issue #165 - Python Version Standardization to 3.11

set -e  # Exit on error

echo "========================================================================"
echo "Python 3.11 Integration Test Environment Setup"
echo "========================================================================"
echo ""

# Step 1: Verify Python 3.11 is available
echo "[Step 1] Verifying Python 3.11..."
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif [ -f "/opt/hostedtoolcache/Python/3.11.13/x64/bin/python3.11" ]; then
    PYTHON_CMD="/opt/hostedtoolcache/Python/3.11.13/x64/bin/python3.11"
else
    echo "❌ ERROR: Python 3.11 not found"
    echo "Please install Python 3.11 first"
    exit 1
fi

$PYTHON_CMD --version
echo "✅ Python 3.11 found"
echo ""

# Step 2: Verify Python 3.11 requirement
echo "[Step 2] Verifying Python 3.11 requirement..."
$PYTHON_CMD -c "import sys; assert sys.version_info[:2] == (3, 11), 'Python 3.11 required!'"
echo "✅ Python 3.11 requirement verified"
echo ""

# Step 3: Create clean virtual environment
echo "[Step 3] Creating clean Python 3.11 virtual environment..."
VENV_DIR="/tmp/venv311"
rm -rf $VENV_DIR
$PYTHON_CMD -m venv $VENV_DIR
echo "✅ Virtual environment created at $VENV_DIR"
echo ""

# Step 4: Activate virtual environment and upgrade pip
echo "[Step 4] Activating virtual environment and upgrading pip..."
source $VENV_DIR/bin/activate
python --version
pip install --upgrade pip > /dev/null 2>&1
echo "✅ Virtual environment activated and pip upgraded"
echo ""

# Step 5: Install core dependencies
echo "[Step 5] Installing core dependencies..."
echo "This may take a few minutes..."
pip install -q ccxt pandas numpy python-dotenv pyyaml requests \
    python-telegram-bot "aiohttp==3.8.6" "yarl<2.0" "multidict<7.0" \
    pytest pytest-asyncio pytest-timeout pytest-mock scikit-learn
echo "✅ Core dependencies installed"
echo ""

# Step 6: Verify aiohttp 3.8.6 installation (CRITICAL)
echo "[Step 6] Verifying aiohttp 3.8.6 installation..."
python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__} loaded successfully')"
echo ""

# Step 7: Verify ccxt.pro availability
echo "[Step 7] Verifying ccxt.pro availability..."
python -c "import ccxt; import ccxt.pro; print(f'✅ ccxt {ccxt.__version__} with ccxt.pro support')"
echo ""

# Step 8: Display environment summary
echo "========================================================================"
echo "Environment Setup Complete!"
echo "========================================================================"
echo ""
echo "Python Version: $(python --version)"
echo "Virtual Environment: $VENV_DIR"
echo "aiohttp: $(python -c 'import aiohttp; print(aiohttp.__version__)')"
echo "ccxt: $(python -c 'import ccxt; print(ccxt.__version__)')"
echo "pytest: $(python -c 'import pytest; print(pytest.__version__)')"
echo ""
echo "To use this environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run integration tests:"
echo "  pytest tests/integration/ -v -s --tb=short"
echo ""
echo "========================================================================"
