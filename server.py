# SPDX-License-Identifier: Apache-2.0
"""Twilio Media Streams phone voice bot server.

- POST /twiml   TwiML for incoming calls: <Connect><Stream> points at /media here
- WS   /media   Twilio Media Streams bidirectional WebSocket
"""

import asyncio
import base64
import json
import logging
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import config
from asr_engine import ASRSession
from audio_codec import ulaw8k_to_pcm16_16k
from call_session import CallSession, STATE_LISTENING, STATE_SPEAKING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("server")

app = FastAPI(title="qwen3-tts-twilio-bot")

# streamSid -> CallSession
_sessions: dict[str, CallSession] = {}


def _public_ws_url(request: Request) -> str:
    """wss URL of this service's /media (behind ngrok: use forwarded headers)."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost")
    proto = request.headers.get("x-forwarded-proto", "https")
    ws_proto = "wss" if proto in ("https", "wss") else "ws"
    return f"{ws_proto}://{host}/media"


@app.get("/health")
async def health():
    return {"ok": True, "sessions": len(_sessions)}


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    # GET allowed only as a sanity check; Twilio always POSTs.
    # (A GET here is usually a browser misconfig: Twilio would treat the
    # 405 body as invalid TwiML and the call fails / WebSocket gets 403.)
    ws_url = _public_ws_url(request)
    logger.info("incoming call, stream url=%s", ws_url)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>"""
    return Response(content=xml, media_type="application/xml")


async def _finalize_and_dispatch(session: CallSession, end: int | None):
    """Finalize ASR off the polling loop and dispatch the turn task.

    Runs as its own task so the endpointing loop keeps polling (and keeps
    tracking voice/echo) while transcription + playback proceed. The
    consumed/empty outcomes are normal in interpret mode (too-short
    segments), so log them at debug level."""
    final = await session.asr.finalize(end)
    if final.strip():
        session._interrupted = False
        task = asyncio.create_task(session.handle_turn(final))
        session._turn_tasks.add(task)
        task.add_done_callback(session._turn_tasks.discard)
    else:
        logger.debug("[%s] endpoint: ASR empty, no turn", session.stream_sid)


async def _endpointing_loop(session: CallSession):
    """Poll for user turn end: in listening state, once VAD silence exceeds the
    threshold and speech was accumulated -> finalize ASR and start a new turn.

    Speech that began while the bot was still playing (pure echo) does not open
    a turn; only speech that started after the echo window does.

    In interpret mode segmentation is continuous: a segment also fires after
    TRANSLATE_MIN_VOICED_MS of speech without waiting for silence, and
    finalize consumes only that segment's bytes so speech arriving during
    transcription opens the next segment."""
    last_voice_at = time.monotonic()
    prev_voice_ms = 0.0
    turn_open = False
    seg_bytes = 0          # bytes of the current segment in the ASR buffer
    seg_voiced_ms = 0.0    # voiced time of the current segment
    finalize_task: asyncio.Task | None = None
    while not session._closed:
        await asyncio.sleep(0.1)
        now = time.monotonic()
        state = session.state
        voice_ms = session.vad._voice_ms
        # Voice arrived since the last poll (works in every state, including
        # right after a barge-in flipped the state to LISTENING):
        new_voice = voice_ms > prev_voice_ms
        if new_voice:
            last_voice_at = time.monotonic()
            if session.mode == "interpret":
                # Attribute the delta to the current segment; a VAD restart
                # (silence >300ms reset) starts a fresh segment.
                if voice_ms > prev_voice_ms + seg_voiced_ms + 1:
                    seg_voiced_ms = voice_ms
                    seg_bytes = len(session.asr._pcm)
                else:
                    seg_voiced_ms += voice_ms - prev_voice_ms
            # Open a turn only when the voiced frame was outside the echo window
            # (or the bot already stopped playing): echo_prefix < voiced means a
            # clean speech frame arrived.
            # Interpret mode: never open a segment while the bot is still
            # playing (or inside its echo tail) — the bot's own playback echo
            # would open ghost segments and dispatch a second concurrent
            # turn. The caller's real speech keeps accumulating in the ASR
            # buffer and the next segment picks it up after playback ends.
            if (state == STATE_LISTENING
                    and not turn_open
                    and session.vad._echo_ms < session.vad._voice_ms
                    and not (session.mode == "interpret"
                            and now < session._outbound_until)):
                turn_open = True
                if session.mode == "interpret":
                    seg_bytes = len(session.asr._pcm)
                    seg_voiced_ms = voice_ms
        prev_voice_ms = voice_ms
        if state != STATE_LISTENING:
            continue
        if session.mode == "interpret":
            # Continuous segmentation: finalize after enough speech, no silence
            # needed. The segment consumes its own bytes so the next segment
            # starts clean; no re-fire spam on stale windows.
            if finalize_task is not None and not finalize_task.done():
                continue
            if turn_open and seg_voiced_ms >= config.TRANSLATE_MIN_VOICED_MS:
                logger.info("[%s] interp segment: voiced=%.0fms, finalizing",
                            session.stream_sid, seg_voiced_ms)
                turn_open = False
                prev_voice_ms = 0.0
                # Advance the segment boundary NOW: if finalize later returns
                # "" (too short), the loop must not re-fire on the same
                # bytes every poll (seen in live calls as a 100ms skip spam).
                # The caller's speech keeps accumulating past this point and
                # opens the next segment.
                fire_bytes = seg_bytes
                seg_bytes = len(session.asr._pcm)
                seg_voiced_ms = 0.0
                finalize_task = asyncio.create_task(
                    _finalize_and_dispatch(session, fire_bytes))
            continue
        if voice_ms > 0:
            continue
        silence = time.monotonic() - last_voice_at
        if turn_open and silence > config.TURN_END_SILENCE_SEC:
            logger.info("[%s] endpoint: silence=%.1fs, finalizing ASR",
                        session.stream_sid, silence)
            turn_open = False
            finalize_task = asyncio.create_task(_finalize_and_dispatch(session, None))
            last_voice_at = time.monotonic()


@app.websocket("/twiml")
async def twiml_ws_fallback(ws: WebSocket):
    """Safety net: some Twilio numbers are configured to open the Media Stream
    on the webhook URL itself (GET /twiml), not /media. Accept here and log so
    the call still connects instead of getting a 403."""
    logger.warning("Media Stream opened on /twiml (fallback); webhook should point <Stream> at /media")
    await media_stream(ws)


@app.websocket("/media")
async def media_stream(ws: WebSocket):
    await ws.accept()
    session: CallSession | None = None
    stream_sid: str | None = None
    send_lock = asyncio.Lock()

    async def send_json(msg: dict):
        async with send_lock:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                pass  # call already ended, ignore send failures

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                logger.info("twilio websocket connected")

            elif event == "start":
                stream_sid = msg["start"]["streamSid"]
                session = CallSession(ws, send_json,
                                     custom_params=msg["start"].get("customParameters", {}))
                session.stream_sid = stream_sid
                session.asr = ASRSession(sid=stream_sid[:12])
                _sessions[stream_sid] = session
                asyncio.create_task(_endpointing_loop(session))
                await session.start()

            elif event == "media":
                if session:
                    ulaw = base64.b64decode(msg["media"]["payload"])
                    try:
                        await session.on_inbound_media(ulaw)
                    except (asyncio.CancelledError, WebSocketDisconnect):
                        raise
                    except Exception:
                        logger.exception("inbound media handling failed")

            elif event == "mark":
                if session:
                    await session.on_mark(msg["mark"]["name"])

            elif event == "dtmf":
                if session:
                    await session.on_dtmf(msg["dtmf"]["digit"])

            elif event == "stop":
                logger.info("stream stopped: %s", stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("twilio websocket disconnected: %s", stream_sid)
    except Exception:
        logger.exception("media stream error: %s", stream_sid)
    finally:
        if session:
            await session.close()
        if stream_sid:
            _sessions.pop(stream_sid, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level="info",
        timeout_keep_alive=30,
    )
