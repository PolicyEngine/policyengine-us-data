# Enhanced CPS Reproduction Environment
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    curl \
    wget \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./
COPY policyengine_us_data/__init__.py policyengine_us_data/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy rest of application
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed logs

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV POLICYENGINE_GITHUB_MICRODATA_AUTH_TOKEN=""
ENV CENSUS_API_KEY=""

# Default command
CMD ["python", "-c", "print('Enhanced CPS Docker environment ready. Run: make help')"]

# Expose port for dashboard
EXPOSE 8080

# Add labels
LABEL maintainer="PolicyEngine"
LABEL version="1.0"
LABEL description="Reproducible environment for Enhanced CPS dataset generation"