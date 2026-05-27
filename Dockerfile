# Multi-stage build — single Python environment for all submodules
# Base image already ships PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps needed by some Python packages (e.g. OpenCV-less builds, git for yfinance)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# torch/torchvision/torchaudio are already in the base image — skip them here
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt

# Copy project source (venvs, cache, data, and model checkpoints excluded via .dockerignore)
COPY . .

# Expose demo API port
EXPOSE 8000

# Environment variables (override at runtime with -e or --env-file)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Default: run the FastAPI demo server
# Override CMD to run training scripts, e.g.:
#   docker run ... python THGNN/train_ic_ranked.py
CMD ["uvicorn", "demo.api:app", "--host", "0.0.0.0", "--port", "8000"]
