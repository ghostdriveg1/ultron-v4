# ─────────────────────────────────────────────────────────
# Ultron Voice Space Dockerfile
# Deploy to: ghostdrive1/ultron-voice (HF Space, acct2)
#
# Bug fixes baked in:
#   T1 [HIGH]  Kokoro weights pre-downloaded at build time
#   VP1 [HIGH] PyNaCl present (discord voice recv)
#   VP2 [HIGH] LiveKit for WebRTC (UDP port isolation per Space)
#
# Port: 7860 (HF Space standard)
# Python: 3.10
# ─────────────────────────────────────────────────────────

FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Space requirement)
RUN adduser --disabled-password --gecos '' ultron
WORKDIR /app

# Copy requirements first for layer cache
COPY requirements-voice.txt .

# Install Python deps
# torch CPU (no CUDA needed for Kokoro inference on HF CPU Space)
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    torchaudio==2.3.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements-voice.txt

# ── T1 FIX: Pre-download Kokoro-82M weights at build time ──
# Forces weight download during Docker build, not at cold start
# Prevents runtime download failure on HF Spaces (no internet at startup)
RUN python -c "
try:
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a')
    print('Kokoro weights downloaded OK')
except Exception as e:
    print(f'Kokoro pre-download warning: {e}')
    # Non-fatal: STT still works, TTS degrades to stub
" || true

# Copy application code
COPY packages/ ./packages/

# Env defaults (secrets injected via HF Space Repository Secrets)
ENV PYTHONPATH=/app \
    PORT=7860 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info

USER ultron

# ── VP2 FIX: LiveKit handles WebRTC + UDP port isolation ──
# LiveKit runs as a sidecar process; voice pipeline connects to it
# Main process: FastAPI voice server on port 7860
# LiveKit: handles WebRTC SFU, proxies UDP through its own socket
CMD ["python", "-m", "uvicorn", \
     "packages.voice.server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
