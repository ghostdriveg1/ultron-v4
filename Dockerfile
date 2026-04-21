# Ultron Brain V4 — HuggingFace Spaces Docker
# python:3.11-slim is intentional: no CUDA, minimal image, fast build
FROM python:3.11-slim

# System deps: curl (health checks), git (optional tooling), ffmpeg (voice Phase 5)
# Playwright chromium deps installed explicitly — --with-deps breaks on Debian trixie
# (ttf-unifont + ttf-ubuntu-font-family removed from trixie repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ffmpeg \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libasound2t64 libpango-1.0-0 libcairo2 \
    fonts-liberation fonts-unifont && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache: deps change rarely)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium binary only (no --with-deps, system deps above)
RUN playwright install chromium

# Copy source
COPY . .

# Non-root user (HF Space requirement: uid 1000)
RUN adduser --disabled-password --gecos '' --uid 1000 ultron && \
    chown -R ultron:ultron /app
USER ultron

# HF Space metadata: app_port required or Space never opens
# See README.md for the ---\napp_port: 7860\n--- block (MUST be in README)
EXPOSE 7860

# Start Brain (FastAPI) + Discord bot as background processes
# Single worker: avoids dual-KeyPool quota burn (bug M1 / BOT1)
# PYTHONPATH=/app: enables all packages.* imports
CMD ["bash", "-c", \
    "PYTHONPATH=/app uvicorn packages.brain.main:app --host 0.0.0.0 --port 7860 --workers 1 & \
    sleep 5 && \
    PYTHONPATH=/app python3 -c 'from packages.brain.discord_bot import run; run()' & \
    wait"]
