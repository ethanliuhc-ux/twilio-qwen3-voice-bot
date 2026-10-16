# SPDX-License-Identifier: Apache-2.0
"""Per-call state machine: listening -> thinking -> speaking -> listening …

Responsibilities:
- Receive Twilio inbound μ-law, decode + VAD + feed ASR
- Silence detection marks end of user speech -> streaming LLM -> per-sentence
  TTS -> send to Twilio on a 20ms cadence
- Barge-in: sustained user speech while speaking -> send clear to stop playback
"""

import asyncio
import base64
import logging
import time

import numpy as np

import config
import asr_engine
from asr_engine import ASRSession
from audio_codec import ulaw8k_to_pcm16_16k, ulaw_to_pcm16
from llm_client import stream_chat
from tts_engine import synthesize_stream

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"


class EnergyVAD:
    """Simple energy VAD: frames whose RMS exceeds the threshold count as speech.

    Tracks two accumulators: total voiced time and the *echo-only* prefix of the
    current utterance (voiced frames that arrived while the bot was still playing
    its own audio, i.e. the line echo of the bot). A real barge-in must have
    voiced time beyond the echo prefix.
    """

    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._echo_ms = 0.0

    def feed(self, pcm16k: bytes, echo_present: bool = False) -> tuple[bool, float, float]:
        """Returns (is_speech, voiced_duration_ms, echo_prefix_ms)."""
        pcm = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
        if pcm.size == 0:
            return self._voice_ms > 0, self._voice_ms, self._echo_ms
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        frame_ms = len(pcm) / config.ASR_SAMPLE_RATE * 1000.0
        if rms > self.threshold:
            if self._silence_ms > 300:  # utterance restart: reset both accumulators
                self._voice_ms = 0.0
                self._echo_ms = 0.0
            self._voice_ms += frame_ms
            self._silence_ms = 0.0
            if echo_present:
                self._echo_ms += frame_ms
            else:
                # clean (non-echo) speech arrived: the echo prefix is over
                self._echo_ms = 0.0
        else:
            self._silence_ms += frame_ms
            if self._silence_ms > 300:  # brief pauses do not reset the speech accumulator
                self._voice_ms = 0.0
                self._echo_ms = 0.0
        return self._voice_ms > 0, self._voice_ms, self._echo_ms

    def reset(self):
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._echo_ms = 0.0


class CallSession:
    """One call. server.py holds this object for the lifetime of the Twilio WS."""

    def __init__(self, ws, send_json, *, custom_params: dict | None = None):
        self.ws = ws
        self._send_json = send_json
        self.custom_params = custom_params or {}
        self.stream_sid: str | None = None
        self.state = STATE_IDLE
        self.history: list[dict] = []
        self.asr = ASRSession(sid="")
        self.vad = EnergyVAD()
        self._interrupted = False
        self._turn_tasks: set[asyncio.Task] = set()
        self._last_activity = time.monotonic()
        self._closed = False
        # Wall-clock deadline until which inbound audio is still the bot's own
        # echo (outbound playback + tail). Used to gate barge-in and endpointing.
        self._outbound_until = 0.0
        # "chat" = normal assistant; "interpret" = real-time translation mode
        # (continuous segmentation, no barge-in, translation-only replies).
        self.mode = "chat"
        # Target language the user asked to translate INTO ("zh"/"ja"/"en");
        # None = auto-pick from the transcript's language.
        self.interp_target: str | None = None
        # Interpret mode serializes playback through one pump task so two
        # speak() loops can never interleave chunks into the Twilio stream.
        # Items are (enqueue_monotonic, text); stale items are dropped so the
        # bot re-syncs with the caller instead of speaking hours-old text.
        self._interp_q: asyncio.Queue[tuple[float, str]] = asyncio.Queue()
        self._interp_pump_task: asyncio.Task | None = None

    # -- Twilio send ------------------------------------------------------

    async def _send_media(self, ulaw: bytes):
        msg = {
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"track": "outbound", "payload": base64.b64encode(ulaw).decode()},
        }
        await self._send_json(msg)

    async def _send_mark(self, name: str):
        await self._send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": name},
        })

    async def _send_clear(self):
        await self._send_json({"event": "clear", "streamSid": self.stream_sid})

    # -- Lifecycle ---------------------------------------------------------

    async def start(self):
        self.state = STATE_LISTENING
        self._last_activity = time.monotonic()
        self.history.append({"role": "assistant", "content": config.WELCOME_TEXT})
        await self.speak(config.WELCOME_TEXT)

    async def close(self):
        self._closed = True
        # Drain queued translations without speaking (call is ending).
        while not self._interp_q.empty():
            try:
                self._interp_q.get_nowait()
                self._interp_q.task_done()
            except asyncio.QueueEmpty:
                break
        for t in self._turn_tasks:
            if not t.done():
                t.cancel()

    # -- Inbound events ---------------------------------------------------------

    async def on_inbound_media(self, ulaw: bytes):
        """Twilio inbound audio (8kHz μ-law)."""
        if self._closed:
            return
        now = time.monotonic()
        if self._last_activity < self._outbound_until and now >= self._outbound_until:
            # Bot just stopped playing: the voice accumulated so far was the bot's
            # own echo, not the user. Wipe both accumulators so the endpointing
            # loop can't mistake the echo tail for a user turn.
            self.vad.reset()
        self._last_activity = now
        pcm16k = ulaw8k_to_pcm16_16k(ulaw)  # pure numpy, microseconds, no thread needed
        echo_present = self._last_activity < self._outbound_until
        is_speech, voiced_ms, echo_ms = self.vad.feed(pcm16k, echo_present)

        # Barge-in: voiced past the threshold AND sustained beyond what the
        # bot's own echo could explain (echo prefix shorter than voiced time).
        # Disabled in interpret mode: the caller speaks continuously there,
        # so the bot finishes each translation like a real interpreter (their
        # speech is still captured and translated after playback).
        if (self.state == STATE_SPEAKING
                and self.mode == "chat"
                and voiced_ms >= config.BARGE_IN_MS
                and echo_ms < voiced_ms):
            await self._barge_in()
            return
        # Periodic VAD status log to debug "didn't hear anything" issues
        self._vad_log_at = getattr(self, "_vad_log_at", 0.0)
        if self.state == STATE_LISTENING and now - self._vad_log_at > 2.0:
            self._vad_log_at = now
            logger.debug("[%s] VAD voiced=%.0fms rms-chunk=%d",
                         self.stream_sid, voiced_ms, len(pcm16k))

        # Always buffer into ASR: when the user barges in, the speech they spoke
        # *during* playback must be recognized, not lost. The bot's own echo
        # mixed into the buffer is handled by the echo-free re-buffer in
        # _barge_in(); the buffer is also cleared on every turn finalize.
        await self.asr.feed(pcm16k)

    async def on_mark(self, name: str):
        """Twilio confirms a mark finished playing (or replays unplayed marks after clear)."""
        # Simplified: log only; playback progress advances naturally with the tick cadence
        logger.debug("[%s] mark done: %s", self.stream_sid, name)

    async def on_dtmf(self, digit: str):
        logger.info("[%s] DTMF: %s", self.stream_sid, digit)

    async def on_stop(self):
        await self.close()

    # -- Barge-in -------------------------------------------------------------

    async def _barge_in(self):
        logger.info("[%s] barge-in, clearing buffer", self.stream_sid)
        # Echo-free re-buffer: the ASR buffer currently holds the bot's own
        # voice mixed with the user's interruption. Replace it with just the
        # frames arriving from here on, so the turn that gets finalized after
        # this barge-in contains only the user's speech.
        await self.asr.reset()
        # Echo deadline: inbound audio is still dominated by the bot's echo
        # until the last already-streamed chunk has played out (+ tail).
        # Barge-in won't re-fire inside this window; endpointing won't open a
        # ghost turn from the echo tail.
        self._outbound_until = time.monotonic() + config.ECHO_TAIL_SEC
        # _interrupted is owned by the speaking turn loop (on_user_turn):
        # it stops speak() loops and suppresses the history append. Do NOT
        # clear it here — clearing would let the interrupted turn resume
        # sending the rest of its sentences.
        self._interrupted = True
        await self._send_clear()
        # Don't cancel any task: speak / on_user_turn exit on their own via the
        # _interrupted flag; cancel would propagate through receive_text and kill
        # the whole WebSocket.
        # Do NOT reset the VAD: the user is *currently speaking* — wiping the
        # accumulator loses the speech that is already buffered for ASR.
        # Switch to LISTENING immediately: the endpointing loop starts tracking
        # the user's turn right away (the echo re-buffer above keeps the bot's
        # own voice out of the transcript, e.g. "你好。你好，能听见吗？").
        self.state = STATE_LISTENING

    # -- Speak -------------------------------------------------------------

    async def speak(self, text: str):
        """Synthesize and play a text at steady pace; returns early if interrupted.

        TTS generation is faster than real time (RTF<1); pacing sends against an
        absolute clock so the Twilio buffer never overflows and audio plays for
        its true duration. Only one speak may run at a time (turns are naturally
        serial). Sentence history bookkeeping is the caller's job
        (on_user_turn / start).
        """
        if self._closed:
            return
        self.state = STATE_SPEAKING
        self._interrupted = False  # new sentence starts, clear the stale interrupt flag
        t0 = time.monotonic()
        # Inbound audio is the bot's own echo until the last scheduled chunk
        # has played out (+ tail for network/jitter).
        self._outbound_until = t0 + config.ECHO_TAIL_SEC
        sent = 0
        try:
            async for chunk in synthesize_stream(text):
                if self._interrupted or self._closed:
                    break
                await self._send_media(chunk)
                sent += 1
                # Pace against absolute clock: chunk n goes out at t0 + n*20ms
                ahead = t0 + sent * config.SEND_TICK_SEC - time.monotonic()
                if ahead > 0:
                    await asyncio.sleep(ahead)
        except asyncio.CancelledError:
            logger.info("[%s] speak cancelled after %d chunks", self.stream_sid, sent)
            raise
        except Exception:
            logger.exception("[%s] speak failed", self.stream_sid)
        finally:
            # On barge-in, _barge_in() already switched to LISTENING; don't
            # clobber it here (the interrupting speech must keep flowing to ASR).
            if self.state == STATE_SPEAKING and not self._closed:
                self.state = STATE_LISTENING
        if not self._interrupted and not self._closed:
            logger.info("[%s] spoken %.1fs audio in %.1fs wall",
                        self.stream_sid, sent * 0.02, time.monotonic() - t0)
            self.vad.reset()
        # Refine the echo deadline to the actual last chunk sent (early exit on
        # barge-in would otherwise leave a stale far-future window).
        self._outbound_until = max(t0 + sent * config.SEND_TICK_SEC, time.monotonic()) \
            + config.ECHO_TAIL_SEC

    # -- Turn handling ---------------------------------------------------------

    async def handle_turn(self, user_text: str):
        """Task entry for on_user_turn: swallows CancelledError so it never reaches the event loop."""
        try:
            await self.on_user_turn(user_text)
        except asyncio.CancelledError:
            logger.info("[%s] turn cancelled", self.stream_sid)
        except Exception:
            logger.exception("[%s] turn failed", self.stream_sid)

    async def _detect_mode_command(self, text: str) -> tuple[str | None, str | None]:
        """Keyword pre-filter + one small LLM call: ('enter'|'exit'|None, target).

        Cheap guard first so normal turns pay zero extra latency. `target` is
        the language the user asked to translate into ('zh'/'ja'/'en'), if
        stated."""
        low = text.lower()
        triggers = ("翻译", "translation", "翻訳", "通訳", "同传", "同声传译")
        exits = ("退出", "结束翻译", "exit", "stop translat", "no longer translat")
        if not any(k in low for k in triggers + exits):
            return None, None
        try:
            reply = ""
            async for sent in stream_chat(
                [
                    {"role": "system", "content":
                        "分类用户这句话。只输出一行：意图|目标语言。"
                        "意图：ENTER_TRANSLATE（用户想开启实时翻译/同声传译模式）、"
                        "EXIT_TRANSLATE（用户想退出翻译模式）、"
                        "CHAT（普通对话，包括提到'翻译'这个词但并非要求开启模式，例如'这个词用英语怎么说'）。"
                        "目标语言：用户希望译文使用的语言：zh、ja、en 或 none"
                        "（例如'翻译成汉语/中文'→zh，'译成日语'→ja，'translate into English'→en，"
                        "未说明→none）。"},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=16,
            ):
                reply += sent
            reply = reply.strip().upper()
            intent = reply.split("|")[0].strip()
            lang_part = reply.split("|")[1].strip().lower() if "|" in reply else "none"
            target = {"ZH": "zh", "JA": "ja", "EN": "en"}.get(lang_part.upper())
            if "ENTER_TRANSLATE" in intent:
                return "enter", target
            if "EXIT_TRANSLATE" in intent:
                return "exit", None
        except Exception:
            logger.exception("[%s] mode classify failed", self.stream_sid)
        return None, None

    async def on_user_turn(self, user_text: str):
        """User finished a sentence (triggered by endpointing)."""
        if self._closed or not user_text.strip():
            logger.info("[%s] turn skipped (empty)", self.stream_sid)
            return
        # Interpret mode: the caller speaks continuously, so every segment is
        # real content — never drop it because the bot is mid-playback.
        if self.state == STATE_SPEAKING and self.mode != "interpret":
            logger.info("[%s] turn skipped (bot still speaking)", self.stream_sid)
            return
        # Mode switching (spoken): recognized in both modes; a classify
        # failure is treated as a normal turn so chat never breaks.
        # Skipped while the bot is playing (interpret mode): the echo of the
        # bot's own voice can contain the trigger word and flip modes by
        # accident; the caller can just say it again after playback.
        if self.state != STATE_SPEAKING:
            command, target = await self._detect_mode_command(user_text)
        else:
            command, target = None, None
        if command == "enter":
            logger.info("[%s] >>> user: %s", self.stream_sid, user_text)
            self.mode = "interpret"
            self.interp_target = target
            await self.speak(config.TRANSLATE_ON_TEXT)
            return
        if command == "exit" and self.mode == "interpret":
            logger.info("[%s] >>> user: %s", self.stream_sid, user_text)
            self.mode = "chat"
            await self.speak(config.TRANSLATE_OFF_TEXT)
            return
        if self.mode == "interpret":
            await self._interpret_turn(user_text)
            return
        # Barge-in path: the user interrupted the previous reply; their speech
        # since then is real and must be processed — clear the stale flag.
        # (speak() clears it at the start of each sentence, so after the last
        # sentence exits on the flag the value here is stale.)
        self._interrupted = False
        # Prevent concurrent turns: if the previous turn is still winding down, wait for it (max 5s)
        # Note: the current task is itself in _turn_tasks; skip self
        cur = asyncio.current_task()
        for t in list(self._turn_tasks):
            if t is cur or t.done():
                continue
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
        logger.info("[%s] >>> user: %s", self.stream_sid, user_text)
        self.history.append({"role": "user", "content": user_text})
        self.state = STATE_THINKING
        reply_buf = ""
        try:
            async for sent in stream_chat(
                [{"role": "system", "content": config.SYSTEM_PROMPT}, *self.history[-20:]]
            ):
                if self._interrupted or self._closed:
                    break
                reply_buf += sent
                logger.info("[%s] <<< bot: %s", self.stream_sid, sent)
                await self.speak(sent)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] LLM failed", self.stream_sid)
            await self.speak("すみません、ただいま不具合が発生しています。しばらくしてからもう一度おかけ直しください。")
            return
        if not self._closed:
            if self._interrupted:
                # Barge-in: the user is speaking right now. Keep their buffered
                # speech in ASR/VAD so the endpointing loop captures this new
                # turn; don't reset and don't overwrite the state they set.
                return
            # Partial sentences spoken before an error still count as history.
            if reply_buf:
                self.history.append({"role": "assistant", "content": reply_buf})
            await self.asr.reset()
            self.state = STATE_LISTENING
            self.vad.reset()

    async def _interp_pump(self):
        """Single serialization point for interpret-mode playback.

        Only one pump runs per session, so two speak() loops can never
        interleave their μ-law chunks into the same Twilio stream (the
        stutter seen in live calls). Between translations it waits only
        until the already-scheduled audio has played out (the chunk clock),
        not a fixed echo tail — the ASR buffer is protected by
        _outbound_until, not by dead air. Stale queue items (the caller
        spoke on while the bot fell behind) are dropped so the bot
        re-syncs instead of reading 30s-old text."""
        try:
            while not self._closed:
                queued_at, text = await self._interp_q.get()
                if self._closed:
                    break
                # Re-sync: drop translations overtaken by newer content.
                if time.monotonic() - queued_at > config.TRANSLATE_STALE_SEC:
                    logger.info("[%s] interp stale drop (%.1fs late): %r",
                              self.stream_sid, time.monotonic() - queued_at,
                              text[:60])
                    continue
                await self.speak(text)
                # Wait only for the already-scheduled audio to play out
                # (speak() set _outbound_until to last-chunk + tail).
                wait = self._outbound_until - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] interp pump failed", self.stream_sid)

    async def _interpret_turn(self, user_text: str):
        """Translation mode: translate and enqueue; no chat history, no barge-in.

        The caller keeps talking while we speak — their speech accumulates in
        the ASR buffer and the endpointing loop opens the next segment.
        Playback goes through _interp_pump so translations never overlap."""
        # Prompt-leak guard (PROJECT_SUMMARY trap #2): the ASR prompt can be
        # transcribed as speech; in chat mode finalize() drops it, but
        # interpret turns bypass finalize's guard, so check here. Without this
        # the bot speaks its own prompt, transcribes its own playback, and
        # loops.
        if any(p in user_text for p in asr_engine._PROMPT_LEAKS):
            logger.warning("[%s] interp prompt-leak dropped: %r", self.stream_sid, user_text[:60])
            return
        logger.info("[%s] >>> caller: %s", self.stream_sid, user_text)
        if self.interp_target:
            lang_name = {"zh": "简体中文", "ja": "日语", "en": "英语"}.get(
                self.interp_target, "简体中文")
            prompt = (f"你是同声传译。把用户接下来输入的每一句话翻译成{lang_name}。"
                      f"无论输入是什么语言，输出永远只能是{lang_name}。"
                      f"只输出译文本身，不要任何解释、引号、前缀或标注。")
        else:
            prompt = config.TRANSLATE_PROMPT
        try:
            async for sent in stream_chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": user_text}],
                temperature=0.3,
            ):
                if self._closed:
                    break
                logger.info("[%s] <<< interp: %s", self.stream_sid, sent)
                await self._interp_q.put((time.monotonic(), sent))
            if self._interp_pump_task is None or self._interp_pump_task.done():
                self._interp_pump_task = asyncio.create_task(self._interp_pump())
                self._turn_tasks.add(self._interp_pump_task)
                self._interp_pump_task.add_done_callback(self._turn_tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] translate failed", self.stream_sid)
        finally:
            # Do NOT flip to LISTENING here: this translation's audio is still
            # draining through Twilio's buffer and its echo would open ghost
            # segments. speak() already set state=LISTENING after its last
            # chunk; _outbound_until gates the echo tail.
            if not self._closed and self.state == STATE_THINKING:
                self.state = STATE_LISTENING
