"""
packages/voice/server.py

Ultron V4 — Voice Space FastAPI Entrypoint
============================================
Standalone FastAPI app for the ultron-voice HF Space (separate from Brain).
Deployed to: ghostdrive1/ultron-voice (HF Spaces Docker, port 7860)

Endpoints:
  POST /stt     — Audio bytes -> text via Groq Whisper (whisper-large-v3-turbo)
  POST /tts     — Text -> WAV audio bytes via Kokoro TTS
  GET  /health  — Voice Space health check

Design:
  - No LLM calls. No Brain dependency. Pure voice pipeline.
  - STT: multipart POST to Groq audio/transcriptions API (OpenAI-compat)
  - TTS: Kokoro local model loaded at startup (pre-downloaded in Dockerfile)
  - Auth: X-Ultron-Token header (same token as Brain)
  - Both endpoints return JSON (STT) or raw WAV bytes (TTS)
  - Target latency: STT ~300-500ms, TTS ~200-400ms on CPU

Pipecat pattern used (github.com/pipecat-ai/pipecat):
  - stt.py: POST to https://api.groq.com/openai/v1/audio/transcriptions
            multipart form: file=("audio.wav", bytes, "audio/wav"), model, language
            response: {"text": "transcribed text"}
  - Kokoro: KPipeline(lang_code='a'), pipeline(text, voice='af_heart')
            yields (gs, ps, audio) tuples; audio is float32 numpy array
            convert float32 -> int16 -> WAV bytes for return

Pre-registered bugs:
  VS1 [HIGH]  Kokoro KPipeline() blocks startup (model load ~5-15s, sync).
              Fix: load in asyncio.get_event_loop().run_in_executor(None, _load_kokoro)
              during lifespan. Startup returns while model loads.
              If /tts called before model ready -> 503 "Model loading...".

  VS2 [HIGH]  Single GROQ_STT_KEY env — no pool rotation. If key hits rate limit,
              all STT requests fail until cooldown. Fix: use GROQ_STT_KEY_0..N
              indexed rotation (same pattern as brain KeyPool). Deferred to v26.

  VS3 [MED]   Kokoro output is float32 numpy array (range -1..1).
              Must convert to int16 (multiply by 32767, clip, astype np.int16)
              before writing WAV. Silent audio if conversion skipped.

  VS4 [MED]   Groq Whisper max file size = 25MB. Discord audio (~192kbps opus)
              is much smaller, but if raw PCM uploaded: 16kHz * 2bytes * 60s = 1.9MB.
              No risk at target usage. Add explicit 25MB limit guard anyway.

  VS5 [LOW]   Kokoro synthesis is synchronous (runs on CPU). For text > 200 chars,
              synthesis takes 2-4s and blocks the event loop. Fix: run in executor.
              Already handled in /tts via asyncio.get_event_loop().run_in_executor.

  VS6 [LOW]   WAV header requires sample_rate and n_channels. Kokoro default is
              24000 Hz mono. If Kokoro changes output rate, WAV will be distorted.
              Fix: always read sample_rate from Kokoro output, not hardcoded.

Tool calls used writing this file (v25):
    Github:get_file_contents x1 (pipecat/services/groq/stt.py — Whisper API pattern)
    Github:get_file_contents x1 (pipecat/services/kokoro/ — TTS pattern)
"""

from __future__ import annotations

import asyncio
import hmac
import io
import logging
import os
import struct
import time
import wave
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_STT_KEY: str = os.environ.get("GROQ_STT_KEY", os.environ.get("GROQ_API_KEY_0", ""))
GROQ_STT_URL: str = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
GROQ_STT_LANGUAGE: str = "en"

KOKORO_VOICE: str = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_LANG: str = os.environ.get("KOKORO_LANG", "a")   # 'a' = American English
KOKORO_SAMPLE_RATE: int = 24000                            # VS6: read from pipeline ideally

ULTRON_AUTH_TOKEN: str = os.environ.get("ULTRON_AUTH_TOKEN", "")
MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25MB — Groq Whisper limit (VS4)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_kokoro_pipeline = None   # set in lifespan
_model_loading: bool = False
_model_ready: bool = False
_startup_time: float = 0.0


# ---------------------------------------------------------------------------
# Kokoro loader (blocking — run in executor)
# ---------------------------------------------------------------------------

def _load_kokoro_sync():
    """Synchronous Kokoro model load. Called from executor to avoid blocking loop."""
    global _kokoro_pipeline, _model_ready
    try:
        from kokoro import KPipeline  # type: ignore
        _kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG)
        _model_ready = True
        logger.info(f"[Voice] Kokoro pipeline loaded. voice={KOKORO_VOICE} lang={KOKORO_LANG}")
    except ImportError:
        logger.warning("[Voice] Kokoro not installed — TTS will be unavailable. Install: pip install kokoro")
    except Exception as e:
        logger.error(f"[Voice] Kokoro load failed: {e}")


# ---------------------------------------------------------------------------
# WAV helper
# ---------------------------------------------------------------------------

def _to_wav_bytes(samples, sample_rate: int = KOKORO_SAMPLE_RATE) -> bytes:
    """
    Convert float32 numpy array to WAV bytes.
    VS3: multiply by 32767, clip, cast to int16.
    VS6: use sample_rate param (don't hardcode).
    """
    import numpy as np  # type: ignore
    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if not ULTRON_AUTH_TOKEN:
        return  # dev mode — no auth
    token = request.headers.get("X-Ultron-Token", "")
    if not hmac.compare_digest(token, ULTRON_AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ultron-Token.")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time, _model_loading
    _startup_time = time.monotonic()
    logger.info("[Voice] Ultron V4 Voice Space starting...")

    if not GROQ_STT_KEY:
        logger.warning("[Voice] GROQ_STT_KEY not set — STT will fail at runtime")

    # VS1: load Kokoro in executor to avoid blocking startup
    _model_loading = True
    loop = asyncio.get_event_loop()
    asyncio.create_task(
        loop.run_in_executor(None, _load_kokoro_sync)
    )
    logger.info("[Voice] Kokoro loading in background (non-blocking)...")

    elapsed = (time.monotonic() - _startup_time) * 1000
    logger.info(f"[Voice] Voice Space READY in {elapsed:.1f}ms (Kokoro loading in background)")

    yield

    logger.info("[Voice] Voice Space shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ultron V4 Voice",
    version="4.0.0",
    description="Ultron V4 — Voice pipeline. Groq Whisper STT + Kokoro TTS.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None        # override default voice
    speed: Optional[float] = 1.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    uptime = time.monotonic() - _startup_time
    return JSONResponse({
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "version": "4.0.0",
        "stt": {
            "provider": "groq",
            "model": GROQ_STT_MODEL,
            "key_set": bool(GROQ_STT_KEY),
        },
        "tts": {
            "provider": "kokoro",
            "model_ready": _model_ready,
            "voice": KOKORO_VOICE,
        },
    })


@app.post("/stt")
async def stt(request: Request, audio: UploadFile = File(...)) -> JSONResponse:
    """
    Convert uploaded audio file to text via Groq Whisper.

    Accepts: any audio format Groq supports (wav, mp3, ogg, flac, webm, m4a)
    Returns: {"text": "transcribed text", "latency_ms": float}

    VS4: enforces 25MB file size limit.
    VS2: single key — no rotation yet. Upgrade to pool in v26.
    """
    _check_auth(request)

    if not GROQ_STT_KEY:
        raise HTTPException(status_code=503, detail="GROQ_STT_KEY not set — STT unavailable.")

    t_start = time.monotonic()

    # Read audio bytes
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:  # VS4
        raise HTTPException(
            status_code=413,
            detail=f"Audio too large: {len(audio_bytes) // 1024}KB > 25MB limit."
        )
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    filename = audio.filename or "audio.wav"
    content_type = audio.content_type or "audio/wav"

    # POST to Groq Whisper (pipecat pattern: multipart form)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {GROQ_STT_KEY}"},
                files={"file": (filename, audio_bytes, content_type)},
                data={
                    "model": GROQ_STT_MODEL,
                    "language": GROQ_STT_LANGUAGE,
                    "response_format": "json",
                },
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq Whisper timed out.")
    except Exception as e:
        logger.error(f"[STT] Groq request failed: {e}")
        raise HTTPException(status_code=500, detail=f"STT request failed: {e}")

    if resp.status_code == 429:
        raise HTTPException(status_code=429, detail="Groq STT rate limited. Try again shortly.")
    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="Groq STT key invalid.")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Groq STT returned {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
        text = data.get("text", "").strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT response parse failed: {e}")

    latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(f"[STT] transcribed {len(audio_bytes)} bytes -> {len(text)} chars in {latency_ms:.0f}ms")

    return JSONResponse({
        "text": text,
        "latency_ms": round(latency_ms, 1),
        "model": GROQ_STT_MODEL,
    })


@app.post("/tts")
async def tts(body: TTSRequest, request: Request) -> Response:
    """
    Convert text to WAV audio via Kokoro TTS.

    Returns: WAV bytes (Content-Type: audio/wav)
    VS5: synthesis runs in executor to avoid blocking event loop.
    VS3: float32 -> int16 conversion inside _to_wav_bytes.
    VS1: returns 503 if model not yet loaded.
    """
    _check_auth(request)

    if not _model_ready:
        if _model_loading:
            raise HTTPException(status_code=503, detail="Kokoro model still loading. Try again in 15s.")
        raise HTTPException(status_code=503, detail="Kokoro TTS unavailable (model failed to load).")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text.")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars).")

    voice = body.voice or KOKORO_VOICE
    t_start = time.monotonic()

    # VS5: run blocking synthesis in thread executor
    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            None,
            _synthesize_sync,
            text,
            voice,
        )
    except Exception as e:
        logger.error(f"[TTS] Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}")

    latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(f"[TTS] synthesized {len(text)} chars -> {len(wav_bytes)} bytes in {latency_ms:.0f}ms voice={voice}")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Latency-Ms": str(round(latency_ms, 1)),
            "X-Voice": voice,
        },
    )


def _synthesize_sync(text: str, voice: str) -> bytes:
    """
    Blocking Kokoro synthesis. Run via executor.
    Concatenates all audio chunks from generator into single WAV.
    VS3: float32 -> int16 via _to_wav_bytes.
    VS6: sample_rate read from Kokoro pipeline attribute.
    """
    import numpy as np  # type: ignore

    pipeline = _kokoro_pipeline
    if pipeline is None:
        raise RuntimeError("Kokoro pipeline not initialized")

    # KPipeline(text, voice=...) returns generator of (graphemes, phonemes, audio_array)
    audio_chunks = []
    sample_rate = KOKORO_SAMPLE_RATE

    try:
        for gs, ps, audio in pipeline(text, voice=voice):
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)
                # VS6: try to read sample_rate from pipeline
                if hasattr(pipeline, "sample_rate"):
                    sample_rate = pipeline.sample_rate
    except Exception as e:
        raise RuntimeError(f"Kokoro synthesis error: {e}")

    if not audio_chunks:
        raise RuntimeError("Kokoro produced no audio")

    combined = np.concatenate(audio_chunks, axis=0)
    return _to_wav_bytes(combined, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "packages.voice.server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        workers=1,  # single worker — Kokoro model in global state
    )
