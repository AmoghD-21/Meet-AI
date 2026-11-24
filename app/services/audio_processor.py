"""
app/services/audio_processor.py
================================
Audio format conversion service using ffmpeg.

The browser's MediaRecorder API produces audio in WebM/Opus format.
Groq Whisper requires WAV format at 16 kHz, mono channel, 16-bit signed PCM.
This module bridges that gap using ffmpeg as a subprocess.

Main function — convert_webm_to_wav(audio_bytes, chunk_id):
  1. Validates that the audio bytes are non-empty and > 100 bytes.
  2. Saves raw bytes to uploads/ as chunk_{id}.webm.
  3. Runs ffmpeg with arguments:
       -ar 16000     → 16 kHz sample rate (Whisper requirement)
       -ac 1         → mono channel
       -sample_fmt s16 → 16-bit signed integer PCM
       -y            → overwrite output file without prompting
  4. Cleans up the .webm input file immediately (even on failure).
  5. Checks ffmpeg return code — raises AudioProcessingException on failure.
  6. Returns the path to the .wav output file.

Cleanup function — cleanup_chunk(file_path):
  Deletes a temporary audio file. Errors are logged as warnings
  (not fatal — a leftover temp file is not a critical failure).

KNOWN LIMITATION:
  subprocess.run() is synchronous and briefly blocks the asyncio event loop.
  For production, replace with an async ffmpeg wrapper (e.g. aioconsole).

Author : MeetingAI
"""

import sys
import os
import uuid
import subprocess
import io
from app.utils.logger import get_logger
from app.core.exceptions import AudioProcessingException
from app.core.config import settings

logger = get_logger(__name__)


def convert_webm_to_wav(audio_bytes: bytes, chunk_id: str = None) -> str:

    if chunk_id is None:
        chunk_id = str(uuid.uuid4())[:8]

    logger.info(f"Processing chunk {chunk_id} — size: {len(audio_bytes)} bytes")

    if not audio_bytes or len(audio_bytes) < 100:
        raise AudioProcessingException(
            f"Chunk {chunk_id} too small or empty: {len(audio_bytes)} bytes",
            sys
        )

    input_path = os.path.join(settings.UPLOAD_FOLDER, f"chunk_{chunk_id}.webm")
    output_path = os.path.join(settings.UPLOAD_FOLDER, f"chunk_{chunk_id}.wav")

    try:
        # save raw bytes
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        logger.debug(f"Saved raw chunk to {input_path}")

        # convert with ffmpeg
        command = [
            "ffmpeg", "-i", input_path,
            "-ar", str(settings.SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-y",
            output_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # clean up input regardless of result
        cleanup_chunk(input_path)

        if result.returncode != 0:
            error_detail = result.stderr.decode("utf-8", errors="ignore")
            raise AudioProcessingException(
                f"ffmpeg failed for chunk {chunk_id}: {error_detail}",
                sys
            )

        logger.info(f"Chunk {chunk_id} converted successfully")
        return output_path

    except AudioProcessingException:
        raise

    except Exception as e:
        cleanup_chunk(input_path)  # clean up on unexpected error too
        raise AudioProcessingException(
            f"Unexpected error processing chunk {chunk_id}: {str(e)}",
            sys
        )


def cleanup_chunk(file_path: str) -> None:
    """
    Deletes temporary audio file after transcription is done.

    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {str(e)}")
