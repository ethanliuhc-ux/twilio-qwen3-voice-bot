# SPDX-License-Identifier: Apache-2.0
"""ASR engine: calls local oMLX `/v1/audio/transcriptions`.

One ASRSession per call: feed() accumulates 16kHz PCM, finalize() transcribes
the whole segment and resets for the next turn.
"""

import asyncio
import io
import logging
import time
import wave

import httpx
import numpy as np

import config

logger = logging.getLogger(__name__)

# Energy-trim transcription input: drop leading/trailing silence, keep speech
# (with 0.3s padding on each side)
_TRIM = True

# ASR prompt-leak signatures: instructional prompt transcribed as speech (substring match)
_PROMPT_LEAKS = ("書き起こす", "口語会話", "音声通話",
                 # zh ASR prompt signatures (it leaked back as "speech" in live
                 # calls; Japanese-only signatures missed it)
                 "多语言电话会话", "蛇口", "止水栓", "漏水",
                 "appointment, reservation")


def _trim_silence(pcm16k: bytes) -> bytes:
    """Trim leading/trailing silence by energy (same 0.015 threshold as VAD), keeping 0.3s padding."""
    if not _TRIM:
        return pcm16k
    pcm = np.frombuffer(pcm16k, dtype=np.int16)
    if pcm.size == 0:
        return pcm16k
    frames = pcm.reshape(-1, 160)  # 10ms @ 16k
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1)) / 32768.0
    voiced = np.where(rms > 0.015)[0]
    if voiced.size == 0:
        return b""
    pad = 30  # 0.3s
    lo = max(0, voiced[0] - pad)
    hi = min(len(frames), voiced[-1] + pad + 1)
    return frames[lo:hi].tobytes()


def _pcm_to_wav_bytes(pcm16k: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(config.ASR_SAMPLE_RATE)
        w.writeframes(pcm16k)
    return buf.getvalue()


async def _transcribe(pcm16k: bytes) -> str:
    wav_bytes = await asyncio.to_thread(_pcm_to_wav_bytes, pcm16k)
    data = {"model": config.ASR_MODEL_ID}
    if config.ASR_LANGUAGE:
        data["language"] = config.ASR_LANGUAGE
    if config.ASR_PROMPT:
        data["prompt"] = config.ASR_PROMPT
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0)) as client:
        resp = await client.post(
            f"{config.OMLX_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.OMLX_API_KEY}"},
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"ASR HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("text", "") or ""


class ASRSession:
    """Per-call recognition state: buffers audio, transcribes on finalize."""

    def __init__(self, sid: str = "?"):
        self._pcm = bytearray()
        self._sid = sid
        self._busy = False

    async def feed(self, pcm16k: bytes) -> None:
        """Feed 16kHz s16le PCM. Transcription is handled solely by finalize (endpoint); this only buffers."""
        self._pcm.extend(pcm16k)

    async def finalize(self, end: int | None = None) -> str:
        """Transcribe the buffered segment; returns the text.

        end=None: transcribe everything and clear the buffer (chat mode).
        end=N: transcribe only the first N bytes and delete just those,
        keeping audio that arrived during transcription (continuous
        segmentation in translation mode). Returns "" when the trimmed speech
        is too short — the consumed bytes are still dropped (the caller
        already re-opened the next segment at that boundary); audio that
        arrived after `end` is never lost."""
        if self._busy:
            logger.info("[%s] ASR finalize skipped (busy)", self._sid)
            return ""
        segment = bytes(self._pcm[:end]) if end is not None else bytes(self._pcm)
        audio = _trim_silence(segment)
        dur_ms = len(audio) / 2 / config.ASR_SAMPLE_RATE * 1000
        if len(audio) < config.ASR_SAMPLE_RATE:  # less than 0.5s of speech after trimming
            logger.info("[%s] ASR skip: voiced %.0fms too short", self._sid, dur_ms)
            if end is not None:
                del self._pcm[:end]
            return ""
        if end is not None:
            del self._pcm[:end]
        else:
            self._pcm.clear()
        self._busy = True
        try:
            return await self._transcribe_and_guard(audio, dur_ms)
        finally:
            self._busy = False

    async def _transcribe_and_guard(self, audio: bytes, dur_ms: float) -> str:
        t0 = time.monotonic()
        try:
            text = await _transcribe(audio)
            # prompt-leak guard: drop text when the instructional prompt got transcribed as speech
            if any(p in text for p in _PROMPT_LEAKS):
                logger.warning("[%s] ASR prompt-leak dropped: %r", self._sid, text)
                return ""
            logger.info("[%s] ASR final(voice=%dms, took=%dms): %r",
                        self._sid, dur_ms, (time.monotonic() - t0) * 1000, text)
            return text
        except Exception as exc:
            logger.warning("ASR finalize failed: %s", exc)
            return ""

    async def reset(self):
        self._pcm.clear()
