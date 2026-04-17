"""
packages/voice/stt.py

Ultron V4 — Whisper STT via Groq API
=====================================
Input:  Raw audio bytes (opus or PCM from Discord, or any audio format).
Output: Transcribed text string.

Groq Whisper API:
  - Model: whisper-large-v3-turbo (fastest, 216x realtime)
  - Endpoint: https://api.groq.com/openai/v1/audio/transcriptions
  - Free tier: 7,200 audio minutes/day per key.
  - Response time: ~0.3s for 10s audio.
  - Max file size: 25MB.

Design:
  - Accepts raw bytes + audio format string (default: "ogg" for Discord opus).
  - Returns transcription text. Empty string on failure.
  - Key rotation: uses KeyPool to get a Groq key. Retries once on 429.
  - ffmpeg NOT required — sends audio directly to Groq; Groq handles decode.

Future bug risks (pre-registered):
  V1 [HIGH]  Discord opus audio is in ogg/opus container. Groq accepts it directly
             BUT Discord sends it in chunks — must wait for VAD silence before sending.
             Fix: implement VAD in pipeline.py, not here. STT just receives complete audio.

  V2 [HIGH]  Groq Whisper 429: if user talks rapidly, 2nd request hits rate limit.
             Fix: asyncio.Semaphore(2) in pipeline.py upstream.

  V3 [MED]   If KeyPool has no Groq keys available (all cooldown), STT fails silently.
             Fix: log warning + return empty string.

  V4 [LOW]   Audio bytes >25MB = Groq 413. Truncate at 23MB.

Tool calls used writing this file:
    Github:get_file_contents x1 (ultron-v3 voice layer for Groq Whisper endpoint pattern)
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger("voice.stt")

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_MODEL    = "whisper-large-v3-turbo"
MAX_AUDIO_BYTES  = 23 * 1024 * 1024  # 23MB hard cap (V4 guard)


class WhisperSTT:
    """
    Groq Whisper STT. Stateless. Async.
    Instantiated once in voice pipeline.
    """

    def __init__(self, pool: object) -> None:
        """
        Args:
            pool: KeyPool instance. Used to get a Groq API key per call.
        """
        self._pool = pool

    async def transcribe(
        self,
        audio_bytes: bytes,
        audio_format: str = "ogg",    # Discord sends ogg/opus
        language: Optional[str] = None,
        prompt: Optional[str] = None,  # Context hint to Whisper (improves accuracy)
    ) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes:  Raw audio bytes.
            audio_format: Container format (ogg, mp3, wav, m4a, webm, etc.)
            language:     ISO-639-1 language code hint. None = auto-detect.
            prompt:       Short context string to bias Whisper.

        Returns:
            Transcribed text string. Empty string on any failure.
        """
        if not audio_bytes:
            return ""

        # V4 guard: truncate oversized audio
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            log.warning(f"[STT] Audio too large ({len(audio_bytes)} bytes) — truncating")
            audio_bytes = audio_bytes[:MAX_AUDIO_BYTES]

        # Get a Groq key from pool
        key_obj = self._get_groq_key()
        if not key_obj:
            log.warning("[STT] No Groq key available for Whisper — V3 guard triggered")
            return ""

        api_key = key_obj["key"]
        key_id  = key_obj["key_id"]

        filename  = f"audio.{audio_format}"
        form_data = {"model": WHISPER_MODEL, "response_format": "text"}
        if language:
            form_data["language"] = language
        if prompt:
            form_data["prompt"] = prompt[:224]  # Groq Whisper prompt max 224 tokens

        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                files = {"file": (filename, audio_bytes, f"audio/{audio_format}")}
                r = await c.post(
                    GROQ_WHISPER_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form_data,
                    files=files,
                )

                if r.status_code == 429:
                    log.warning(f"[STT] Groq 429 on key={key_id} — reporting failure")
                    self._pool.report_failure(key_id)
                    return ""

                r.raise_for_status()
                self._pool.report_success(key_id)

                text = r.text.strip()
                log.info(f"[STT] Transcribed {len(audio_bytes)} bytes -> {len(text)} chars")
                return text

        except httpx.HTTPStatusError as e:
            log.error(f"[STT] HTTP error {e.response.status_code}: {e.response.text[:200]}")
            self._pool.report_failure(key_id)
            return ""
        except Exception as e:
            log.error(f"[STT] Transcription failed: {e}")
            return ""

    def _get_groq_key(self) -> Optional[dict]:
        """Scan pool.general for first available Groq key."""
        import time
        try:
            for key_obj in self._pool.general:
                if key_obj.get("provider") == "groq":
                    if key_obj.get("failures", 0) < 3:
                        reset_at = key_obj.get("reset_at", 0)
                        if reset_at == 0 or time.time() > reset_at:
                            return key_obj
        except Exception:
            pass
        return None
