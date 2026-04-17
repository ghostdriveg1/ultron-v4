"""
packages/voice/pipeline.py

Ultron V4 — Voice Pipeline
===========================
Full push-to-talk Discord voice pipeline:
  Discord opus audio -> VAD silence detection -> WhisperSTT ->
  Brain /infer -> KokoroTTS -> WAV response

Target: <1.5s end-to-end from speech end to audio response start.

Discord voice integration notes:
  - discord.py voice receive requires VoiceClient + sink callback.
  - Opus codec: Discord sends 48kHz 16-bit stereo opus frames (20ms each).
  - VAD: accumulate frames until 500ms silence -> treat as end of utterance.
  - Groq Whisper accepts 48kHz ogg/opus directly. No resample needed for STT.

Design:
  - VoicePipeline is instantiated per voice channel connection.
  - on_audio_frame() called by discord.py sink for each 20ms opus frame.
  - Internal asyncio.Queue buffers frames. Background task drains queue.
  - Brain URL called via httpx (same auth as discord_bot.py).

Future bug risks (pre-registered):
  VP1 [HIGH]  discord.py voice receive is only in discord.py 2.x + PyNaCl.
              Must add: discord.py>=2.3.2, PyNaCl>=1.5.0 to requirements.txt Phase 5 section.

  VP2 [HIGH]  HF Space cannot open UDP ports for Discord voice gateway.
              Voice layer MUST run on a separate HF Space (ultron-voice).
              Current Brain Space handles text only.

  VP3 [MED]   VAD silence threshold (500ms = 25 frames) may cut off fast speakers.
              Fix: tune to 800ms (40 frames) based on real usage.

  VP4 [MED]   Brain /infer timeout (10s). If Brain is slow, TTS gets no text.
              Fix: local fallback "Brain offline. Try text commands."

  VP5 [LOW]   Kokoro WAV output is 24kHz. Discord voice expects 48kHz PCM.
              Fix: resample WAV before send using audioop.ratecv() (stdlib).

Tool calls used writing this file:
    External knowledge: discord.py v2 voice docs, Pipecat architecture
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

log = logging.getLogger("voice.pipeline")

# VAD constants
FRAME_DURATION_MS    = 20       # Discord sends 20ms opus frames
SILENCE_THRESHOLD_MS = 500      # End of utterance after 500ms silence
SILENCE_FRAMES       = SILENCE_THRESHOLD_MS // FRAME_DURATION_MS  # 25 frames
MAX_UTTERANCE_FRAMES = 30_000 // FRAME_DURATION_MS  # 30s hard cap


class VoicePipeline:
    """
    Per-channel voice pipeline. Push-to-talk Discord audio -> text -> audio response.

    Usage:
        pipeline = VoicePipeline(stt, tts, brain_url, auth_token, channel_id, user_id)
        await pipeline.start()
        # In discord.py sink callback:
        await pipeline.on_audio_frame(opus_bytes)
        # Pick up response:
        wav = pipeline.last_wav
        reply = pipeline.last_reply
    """

    def __init__(
        self,
        stt: object,
        tts: object,
        brain_url: str,
        auth_token: str,
        channel_id: str,
        user_id: str,
        username: str = "user",
    ) -> None:
        self._stt         = stt
        self._tts         = tts
        self._brain_url   = brain_url
        self._auth_token  = auth_token
        self._channel_id  = channel_id
        self._user_id     = user_id
        self._username    = username

        self._frame_queue: asyncio.Queue = asyncio.Queue()
        self._audio_buffer: list[bytes]  = []
        self._silence_count: int         = 0
        self._recording: bool            = False
        self._task: Optional[asyncio.Task] = None

        # Output — bot reads these after flush
        self.last_wav:   bytes = b""
        self.last_reply: str   = ""

    async def start(self) -> None:
        """Start VAD + transcription background task."""
        self._task = asyncio.create_task(self._vad_loop())
        log.info(f"[Voice] Pipeline started channel={self._channel_id}")

    async def stop(self) -> None:
        """Stop the pipeline cleanly."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info(f"[Voice] Pipeline stopped channel={self._channel_id}")

    async def on_audio_frame(self, opus_bytes: bytes) -> None:
        """Called by discord.py sink for each 20ms frame. Non-blocking."""
        await self._frame_queue.put(opus_bytes)

    # --- VAD loop -----------------------------------------------------------

    async def _vad_loop(self) -> None:
        """
        Drain frame queue with VAD.
        Silence detected -> flush buffer to STT -> Brain -> TTS.
        """
        while True:
            frame = await self._frame_queue.get()

            if frame and len(frame) > 0:
                self._audio_buffer.append(frame)
                self._silence_count = 0
                self._recording = True

                if len(self._audio_buffer) >= MAX_UTTERANCE_FRAMES:
                    log.warning("[Voice] Max utterance length — force flush")
                    await self._flush()
            else:
                if self._recording:
                    self._silence_count += 1
                    if self._silence_count >= SILENCE_FRAMES:
                        await self._flush()

    async def _flush(self) -> None:
        """Process buffered audio: STT -> Brain -> TTS."""
        if not self._audio_buffer:
            return

        audio_bytes = b"".join(self._audio_buffer)
        self._audio_buffer  = []
        self._silence_count = 0
        self._recording     = False

        t_start = time.monotonic()
        log.info(f"[Voice] Flushing {len(audio_bytes)} bytes for STT")

        # 1. STT
        text = await self._stt.transcribe(audio_bytes, audio_format="ogg")
        if not text.strip():
            log.info("[Voice] STT empty — skipping")
            return

        log.info(f"[Voice] STT: '{text[:80]}' ({time.monotonic()-t_start:.2f}s)")

        # 2. Brain /infer
        reply = await self._call_brain(text)
        if not reply:
            reply = "Brain offline. Try text commands."  # VP4 fallback

        log.info(f"[Voice] Brain reply: '{reply[:80]}' ({time.monotonic()-t_start:.2f}s)")

        # 3. TTS
        wav_bytes = await self._tts.synthesize(reply)
        log.info(
            f"[Voice] TTS done. Total latency: {time.monotonic()-t_start:.2f}s. "
            f"WAV={len(wav_bytes)} bytes"
        )

        # 4. Store for bot to pick up
        self.last_wav   = wav_bytes
        self.last_reply = reply

    async def _call_brain(self, message: str, timeout: float = 10.0) -> str:
        """Call Brain /infer. Returns reply string. Empty on failure."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(
                    f"{self._brain_url}/infer",
                    headers={
                        "X-Ultron-Token": self._auth_token,
                        "Content-Type":   "application/json",
                    },
                    json={
                        "message":    message,
                        "channel_id": self._channel_id,
                        "user_id":    self._user_id,
                        "username":   self._username,
                    },
                )
                r.raise_for_status()
                return r.json().get("reply", "")
        except Exception as e:
            log.error(f"[Voice] Brain call failed: {e}")
            return ""
