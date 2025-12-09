# Base Image - Python 3.11 for Azure compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including TA-Lib requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libc-dev \
    libgomp1 \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p logs data artifacts/gemma/final artifacts/ppo features/gemma/selected data/models/final data/cache/gemma \
    && touch data/state.json data/day_stats.json logs/.placeholder

# Set up Python paths inside the container
ENV PYTHONPATH="/app:/app/src:/app/scripts"
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port for Azure App Service
EXPOSE 8000

# Use vm_boot.py as the main entry point for VM + Docker
CMD ["python", "vm_boot.py"]