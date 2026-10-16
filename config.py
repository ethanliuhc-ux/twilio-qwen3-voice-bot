# SPDX-License-Identifier: Apache-2.0
"""Central configuration: environment variables + constants."""

import os
from pathlib import Path


def _load_dotenv():
    """Minimal .env loader (KEY=VALUE, # comments); never overrides existing env vars."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# oMLX (LLM chat)
# ---------------------------------------------------------------------------
OMLX_BASE_URL = os.environ.get("OMLX_BASE_URL", "http://localhost:8000/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3.8-Flash-Next-oQ4e-mtp")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "你是一个电话语音助手，精通日语、普通话和英语。"
    "始终使用用户最后一句话所使用的同一种语言回答（用户说日语就用日语、"
    "说中文就用中文、说英语就用英语），不要主动切换语言。"
    "回答要简短、口语化、自然，每次不超过三句话。"
    "如果用户发音含糊，关键信息（地名、号码、事由）不确定时，"
    "先用同一种语言反问确认，不要猜测。"
    "不要使用 Markdown、列表或表情符号，因为回复会被转成语音朗读。",
)

# ---------------------------------------------------------------------------
# oMLX audio models (TTS / ASR both served by oMLX over HTTP)
# ---------------------------------------------------------------------------
TTS_MODEL_ID = os.environ.get("TTS_MODEL_ID", "Qwen3-TTS-12Hz-1.7B-Base-fp32")
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "Qwen3-ASR-1.7B-bf16")
# Base TTS needs a reference audio for voice cloning (local wav path, auto base64-uploaded to oMLX)
REF_AUDIO_PATH = os.environ.get("REF_AUDIO_PATH", "")
REF_TEXT = os.environ.get("REF_TEXT", "")

# TTS speed multiplier (1.0 = normal, >1 faster)
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.25"))

# Per-language reference audio: synthesize in the language whose native ref is
# configured, to avoid cross-language accent contamination.
# Languages left unset fall back to REF_AUDIO_PATH/REF_TEXT.
REF_AUDIO_PATH_JA = os.environ.get("REF_AUDIO_PATH_JA", "")
REF_TEXT_JA = os.environ.get("REF_TEXT_JA", "")
REF_AUDIO_PATH_ZH = os.environ.get("REF_AUDIO_PATH_ZH", "")
REF_TEXT_ZH = os.environ.get("REF_TEXT_ZH", "")
REF_AUDIO_PATH_EN = os.environ.get("REF_AUDIO_PATH_EN", "")
REF_TEXT_EN = os.environ.get("REF_TEXT_EN", "")

REF_AUDIO_PATHS = {
    "ja": REF_AUDIO_PATH_JA,
    "zh": REF_AUDIO_PATH_ZH,
    "en": REF_AUDIO_PATH_EN,
}
REF_TEXTS = {
    "ja": REF_TEXT_JA,
    "zh": REF_TEXT_ZH,
    "en": REF_TEXT_EN,
}

# ASR tuning: auto = language detection (ja/zh/en mixed OK); can pin ja/zh/en
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "auto")
# Empty by default: Qwen3-ASR transcribes a long instructional prompt back as
# "speech" when the audio window is mostly silence/echo — live calls showed
# this exact string echoed and every such segment dropped (content loss).
# Set it via env only if you accept that trade-off.
ASR_PROMPT = os.environ.get("ASR_PROMPT", "")

# ---------------------------------------------------------------------------
# Audio format constants
# ---------------------------------------------------------------------------
TWILIO_SAMPLE_RATE = 8000          # Twilio Media Streams is fixed 8kHz μ-law
ASR_SAMPLE_RATE = 16000            # mlx-qwen3-asr expects 16kHz
TTS_SAMPLE_RATE = 24000            # mlx-audio qwen3_tts outputs 24kHz

# Send pacing: one 160-byte μ-law block every 20ms (Twilio's recommended granularity)
SEND_TICK_SEC = 0.02
ULAW_BYTES_PER_TICK = TWILIO_SAMPLE_RATE // 50  # 160

# ---------------------------------------------------------------------------
# Dialog state machine / VAD
# ---------------------------------------------------------------------------
# While the bot is speaking, N ms of continuous speech energy counts as barge-in
BARGE_IN_MS = int(os.environ.get("BARGE_IN_MS", "400"))
# How long after the last outbound chunk inbound audio is still treated as the
# bot's own echo (line/speaker echo + network tail). Barge-in and endpointing
# ignore voiced frames inside this window.
ECHO_TAIL_SEC = float(os.environ.get("ECHO_TAIL_SEC", "0.6"))
# After the bot finishes, how long silence means the user is done (with ASR endpointing)
TURN_END_SILENCE_SEC = float(os.environ.get("TURN_END_SILENCE_SEC", "0.8"))

# ---------------------------------------------------------------------------
# Real-time translation (interpret) mode
# ---------------------------------------------------------------------------
# In interpret mode a segment is finalized after this much continuous speech
# (no trailing silence required) so translation runs while the caller talks.
# 2500ms balances latency vs fragmentation: 4000 let segments grow to 25s+
# (bot fell 20-30s behind the caller); 1500 chopped into fragments.
TRANSLATE_MIN_VOICED_MS = int(os.environ.get("TRANSLATE_MIN_VOICED_MS", "2500"))
# A queued translation older than this (from caller speech that has long been
# overtaken) is dropped instead of spoken — an interpreter re-syncs rather
# than reading 30s-old text. Bounds the backlog when output speech is longer
# than the input speech it translates.
TRANSLATE_STALE_SEC = float(os.environ.get("TRANSLATE_STALE_SEC", "8.0"))
TRANSLATE_PROMPT = os.environ.get(
    "TRANSLATE_PROMPT",
    "你是同声传译。把用户接下来输入的话翻译成目标语言。"
    "目标语言：如果输入是日语则翻译成简体中文，如果输入是中文则翻译成日语，"
    "如果输入是英语则翻译成简体中文。"
    "只输出译文本身，不要任何解释、引号、前缀或标注。",
)
TRANSLATE_ON_TEXT = os.environ.get(
    "TRANSLATE_ON_TEXT",
    "好的，已进入实时翻译模式。请连续讲话，我会逐句翻译，您不用停顿。"
    "如需退出，请说退出翻译。",
)
TRANSLATE_OFF_TEXT = os.environ.get(
    "TRANSLATE_OFF_TEXT",
    "好的，已退出翻译模式，我们继续正常对话。",
)

# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "8765"))

# Welcome line for incoming calls ({name} can be replaced via customParameter, not enabled yet)
WELCOME_TEXT = os.environ.get(
    "WELCOME_TEXT",
    "This is a voice assistant. I speak Japanese, Chinese, and English. "
    "How can I help you?",
)
