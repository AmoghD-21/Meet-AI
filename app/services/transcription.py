"""
app/services/transcription.py
==============================
Speech-to-text transcription service using Groq Whisper large-v3.

Groq's LPU (Language Processing Unit) hardware makes Whisper inference
significantly faster than standard GPU-based deployments, making it
practical for near-real-time transcription of 15-second audio chunks.

The Groq client is initialised once at module level as a singleton.
This avoids per-request connection overhead and is safe for concurrent use.

Functions:

transcribe_chunk(audio_path) -> str
    Used by the WebSocket pipeline for live recording.
    Sends a WAV file to Groq's audio.transcriptions.create API.
    Uses response_format="text" — returns a plain string directly.
    Handles both str and object response types for SDK compatibility.
    Raises TranscriptionException on any failure.

transcribe_audio(audio_path) -> dict
    Used by the /api/upload endpoint for pre-recorded files.
    Uses response_format="verbose_json" — returns richer data:
      - text     : full transcript string
      - language : detected language code
      - segments : list of timed segments (if available)
      - words    : word count

Both functions raise TranscriptionException (from app.core.exceptions)
so callers can handle transcription failures independently of other errors.

Author : MeetingAI
"""

import os
import sys
from groq import Groq
from app.core.config import settings
from app.core.exceptions import TranscriptionException
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

# initialize Groq client
try:
    client = Groq(api_key=settings.GROQ_API_KEY)
    logger.info("Groq transcription client initialized")
except Exception as e:
    raise Exception(f"Failed to initialize Groq client: {e}")


def transcribe_chunk(audio_path: str) -> str:
    """
    Transcribes audio using Groq Whisper large-v3.
    Fast, accurate, no local GPU needed.
    """
    try:
        logger.info(f"Starting transcription: {audio_path}")

        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3",
                language="en",
                response_format="text"
            )

        text = result.strip() if isinstance(result, str) else result.text.strip()

        logger.info(f"Transcription complete — words: {len(text.split())}")
        return text

    except Exception as e:
        raise TranscriptionException(
            f"Groq transcription failed: {str(e)}", sys
        )


def transcribe_audio(audio_path: str) -> dict:
    """
    Full transcription for uploaded files.
    Returns dict with text, language, segments.
    """
    try:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3",
                language="en",
                response_format="verbose_json"
            )

        return {
            "text": result.text,
            "language": result.language,
            "segments": result.segments if hasattr(result, 'segments') else [],
            "words": len(result.text.split())
        }

    except Exception as e:
        raise TranscriptionException(
            f"Groq transcription failed: {str(e)}", sys
        )