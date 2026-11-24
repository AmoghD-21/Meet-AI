"""
app/api/upload.py
=================
REST API endpoints for pre-recorded file upload and processing.

Provides an alternative entry point to MeetingAI for users who want to
transcribe and summarise an existing audio or video file rather than
recording live.

Endpoints:
    POST   /api/upload              Upload and process an audio/video file
    GET    /api/session/{id}        Retrieve stored session data
    DELETE /api/session/{id}        Delete session data (privacy compliance)

Supported file types:
    Audio : .mp3, .wav, .m4a, .flac, .aac, .ogg
    Video : .mp4, .avi, .mov, .mkv, .webm
    PDF   : .pdf  (not yet implemented — raises FileProcessingException)

Upload processing pipeline:
  1. Validate file extension against allow-list
  2. Save file with UUID-prefixed filename (security: prevents path traversal)
  3. Transcribe using Groq Whisper (verbose_json — returns language + segments)
  4. Generate structured summary (direct or MapReduce based on length)
  5. Store result in in-memory session_store dict
  6. Clean up uploaded file in finally block
  7. Return summary, word count, model used, and strategy

Limitations:
  - session_store is an in-memory dict — not thread-safe and lost on restart.
    Replace with a database or Redis for production use.
  - PDF extraction is stubbed out (raises FileProcessingException).

Author : MeetingAI
"""

import sys
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from werkzeug.utils import secure_filename
from app.services.transcription import transcribe_audio
from app.services.summary import generate_summary
from app.core.config import settings
from app.core.exceptions import (
    TranscriptionException,
    SummaryGenerationException,
    FileProcessingException
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# in-memory session store
# replace with database later
session_store = {}

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_PDF   = {".pdf"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file upload — audio, video, or PDF.
    Returns session ID and processing status.
    """
    session_id = str(uuid.uuid4())
    save_path = None

    logger.info(f"Upload received — file: {file.filename}, session: {session_id}")

    try:
        # validate file
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALLOWED_AUDIO | ALLOWED_VIDEO | ALLOWED_PDF:
            raise FileProcessingException(
                f"Unsupported file type: {ext}", sys
            )

        # save file
        unique_filename = f"{session_id}_{filename}"
        save_path = os.path.join(settings.UPLOAD_FOLDER, unique_filename)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File saved: {save_path}")

        # transcribe
        if ext in ALLOWED_PDF:
            # PDF handling — plug in your existing pdf_extractor here
            raise FileProcessingException(
                "PDF support coming soon", sys
            )
        else:
            result = transcribe_audio(save_path)
            transcript = result["text"]

        logger.info(f"Transcription complete — words: {len(transcript.split())}")

        # generate summary
        summary_result = generate_summary(transcript)

        # store session
        session_store[session_id] = {
            "file": save_path,
            "transcript": transcript,
            "summary": summary_result["summary"],
            "session_id": session_id
        }

        logger.info(f"Upload processed successfully — session: {session_id}")

        return {
            "status": "completed",
            "session_id": session_id,
            "summary": summary_result["summary"],
            "words_transcribed": len(transcript.split()),
            "model_used": summary_result["model"],
            "strategy": summary_result.get("strategy", "direct")
        }

    except FileProcessingException as e:
        logger.error(f"File processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except TranscriptionException as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")

    except SummaryGenerationException as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail="Summary generation failed")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        # clean up uploaded file after processing
        if save_path and os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.debug(f"Cleaned up: {save_path}")
            except Exception:
                pass


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Returns stored session data."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_store[session_id]


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Deletes session data — privacy compliance."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    del session_store[session_id]
    logger.info(f"Session {session_id} deleted")
    return {"status": "deleted", "session_id": session_id}