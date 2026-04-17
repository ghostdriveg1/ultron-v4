"""packages/voice — Ultron V4 Voice Layer.

Modules:
  stt.py      — Whisper STT via Groq API (push-to-talk Discord audio -> text)
  tts.py      — Kokoro TTS (text -> WAV bytes, CPU, ~0.3s for 100 tokens)
  pipeline.py — Discord audio pipeline: receive opus -> decode -> STT -> Brain -> TTS -> send WAV

Target latency: <1.5s end-to-end.
Separate HF Space (ultron-voice) in Phase 6.
"""
