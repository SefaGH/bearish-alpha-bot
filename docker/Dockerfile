# ============================================================================
# Bearish Alpha Trading Bot - Docker Image
# ============================================================================
# Python 3.11 required for aiohttp==3.8.6 compatibility
# Python 3.12+ NOT supported!
# ============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="SefaGH"
LABEL description="Bearish Alpha Trading Bot"
LABEL version="1.0"
LABEL python-version="3.11"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app/scripts \
    TRADING_MODE=paper \
    CAPITAL_USDT=100 \
    LOG_LEVEL=INFO

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs && chmod 755 logs

# Create non-root user for security
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app

# Switch to non-root user
USER botuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; import os; sys.exit(0 if os.path.exists('logs') else 1)"

# Expose port (if web interface added in future)
# EXPOSE 8080

# Default command (can be overridden)
# Use vm_boot.py to translate env vars (TRADING_MODE/TRADING_DURATION/DEBUG_MODE)
# into launcher flags (--paper/--live/--duration) so Docker runs behave like VM runs.
CMD ["python", "-u", "vm_boot.py"]

# ============================================================================
# Build & Run Instructions
# ============================================================================
# Build:
#   docker build -t bearish-alpha-bot:latest .
#
# Run (paper mode):
#   docker run -e BINGX_KEY=xxx -e BINGX_SECRET=yyy bearish-alpha-bot:latest
#
# Run (live mode):
#   docker run -e TRADING_MODE=live -e EXECUTION_BACKEND=ccxt -e BINGX_ENV=prod -e BINGX_KEY=xxx -e BINGX_SECRET=yyy bearish-alpha-bot:latest
#
# Run with custom duration:
#   docker run -e TRADING_DURATION=600 bearish-alpha-bot:latest
#
# View logs:
#   docker logs -f <container_id>
# ============================================================================
