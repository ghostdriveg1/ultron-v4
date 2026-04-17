"""
packages/voice/tts.py

Ultron V4 — Kokoro TTS
=======================
Input:  Text string.
Output: WAV audio bytes (ready to send to Discord voice channel).

Kokoro:
  - Model: hexgrad/Kokoro-82M (HuggingFace, MIT license)
  - Runtime: CPU-only (~0.3s for 100 tokens, 1s for 300 tokens)
  - Sample rate: 24000 Hz, mono, float32 -> converted to int16 WAV
  - Voices: af_heart (default, warm female), am_adam (male), etc.
  - No API key required — runs locally in HF Space container.
  - 82M params — fits in HF Space 16GB RAM with room to spare.

Design:
  - Lazy-load Kokoro on first call (cold start ~3s, then <1s per call).
  - Returns WAV bytes. Caller sends to Discord.
  - Async wrapper around sync Kokoro inference (runs in thread executor).
  - Truncate text at 500 chars to keep latency <1.5s.

Future bug risks (pre-registered):
  T1 [HIGH]  Kokoro first-call download: hexgrad/Kokoro-82M downloads model weights
             (~330MB) on first import. HF Space must have internet access.
             Fix: pre-download in Dockerfile via `python -c "from kokoro import KPipeline"`.

  T2 [HIGH]  Thread executor + asyncio: Kokoro is sync.
             Fix: asyncio.Semaphore(2) on synthesize() — max 2 concurrent TTS calls.

  T3 [MED]   kokoro package not in requirements.txt yet (Phase 5 dep).
             Add: kokoro>=0.9.2, soundfile>=0.12.1 to requirements.txt Phase 5 section.

  T4 [LOW]   Voice name typos fail silently — fall back to default voice. Desired behavior.

Tool calls used writing this file:
    External knowledge: Kokoro-82M HuggingFace model card + kokoro PyPI docs
"""

from __future__ import annotations

import asyncio
import io
import logging
import struct
import wave
from typing import Optional

log = logging.getLogger("voice.tts")

MAX_TEXT_CHARS = 500          # Truncate to keep latency <1.5s
DEFAULT_VOICE  = "af_heart"   # Warm female voice
SAMPLE_RATE    = 24000        # Kokoro native sample rate


class KokoroTTS:
    """
    Kokoro TTS. Lazy-loads model on first call.
    Async wrapper around sync inference.
    """

    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> None:
        self._voice    = voice
        self._speed    = speed
        self._pipeline = None  # Lazy init
        self._lock     = asyncio.Lock()
        self._sem      = asyncio.Semaphore(2)  # T2 guard

    async def _ensure_loaded(self) -> None:
        """Load Kokoro pipeline on first call (T1: ~3s cold start)."""
        if self._pipeline is not None:
            return
        async with self._lock:
            if self._pipeline is not None:
                return
            try:
                log.info("[TTS] Loading Kokoro pipeline (cold start ~3s)...")
                loop = asyncio.get_event_loop()
                self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
                log.info("[TTS] Kokoro loaded.")
            except ImportError:
                log.error(
                    "[TTS] kokoro package not installed. "
                    "Add kokoro>=0.9.2 to requirements.txt (T3)."
                )
                raise

    @staticmethod
    def _load_pipeline() -> object:
        """Sync Kokoro load. Runs in thread executor."""
        from kokoro import KPipeline  # type: ignore
        return KPipeline(lang_code="a")  # "a" = American English

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """
        Synthesize text to WAV bytes.

        Returns:
            WAV audio bytes (24kHz, mono, int16). Empty bytes on failure.
        """
        if not text.strip():
            return b""

        if len(text) > MAX_TEXT_CHARS:
            log.warning(f"[TTS] Text truncated from {len(text)} to {MAX_TEXT_CHARS} chars")
            text = text[:MAX_TEXT_CHARS]

        v = voice or self._voice
        s = speed or self._speed

        async with self._sem:  # T2 guard
            await self._ensure_loaded()
            try:
                loop = asyncio.get_event_loop()
                audio_array = await loop.run_in_executor(
                    None,
                    lambda: self._synthesize_sync(text, v, s),
                )
                wav_bytes = self._array_to_wav(audio_array)
                log.info(
                    f"[TTS] Synthesized {len(text)} chars -> "
                    f"{len(wav_bytes)} bytes WAV"
                )
                return wav_bytes
            except Exception as e:
                log.error(f"[TTS] Synthesis failed: {e}")
                return b""

    def _synthesize_sync(self, text: str, voice: str, speed: float) -> list:
        """Sync Kokoro inference. Runs in thread executor."""
        import numpy as np  # type: ignore
        chunks = []
        for _, _, audio in self._pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                chunks.append(audio)
        if not chunks:
            return []
        return np.concatenate(chunks, axis=0).tolist()

    @staticmethod
    def _array_to_wav(audio_list: list) -> bytes:
        """Convert float32 list to 16-bit PCM WAV bytes."""
        if not audio_list:
            return b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            frames = b"".join(
                struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
                for s in audio_list
            )
            wf.writeframes(frames)
        return buf.getvalue()
