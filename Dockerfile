FROM python:3.10-slim

# HF Spaces runs as non-root user (uid=1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# System deps for Faster-Whisper (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids downloading the full CUDA build)
RUN pip install --no-cache-dir \
    torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api ./api
COPY client ./client

# Download model from HF Hub at build time so runtime startup is instant.
# MODEL_REPO must be set as a build arg or HF Space secret.
# Example: your-hf-username/sachet-muril-v2
ARG MODEL_REPO=""
ARG HF_TOKEN=""
RUN if [ -n "$MODEL_REPO" ]; then \
      python -c "\
from huggingface_hub import snapshot_download; \
import os; \
snapshot_download( \
  repo_id='${MODEL_REPO}', \
  local_dir='/app/models/sachet_muril_v2', \
  token='${HF_TOKEN}' if '${HF_TOKEN}' else None \
)"; \
    fi

# Give ownership to the non-root user
RUN chown -R user:user /app
USER user

# HF Spaces requires port 7860
EXPOSE 7860

# Serve the app — StaticFiles mount is handled inside api/main.py
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
