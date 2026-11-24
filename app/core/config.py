"""
app/core/config.py
==================
Centralised configuration management for MeetingAI.

All application settings are defined in a single Pydantic `BaseSettings` class.
Every setting can be overridden by an environment variable or a `.env` file —
no magic strings are scattered across the codebase.

Key setting groups:
- Folder paths  : UPLOAD_FOLDER, OUTPUT_FOLDER, TRANSCRIPT_FOLDER, LOG_FOLDER, STATIC_FOLDER
- Audio         : CHUNK_DURATION_MS, SAMPLE_RATE (16 kHz — Whisper requirement), AUDIO_FORMAT
- Transcription : WHISPER_MODEL (whisper-large-v3 via Groq cloud)
- LLM / Summary : GROQ_MODEL (llama-3.3-70b-versatile), GROQ_API_KEY
- MCP           : MCP_CONFIG_FILE — absolute path to mcp_server_file.json
- App           : APP_HOST, APP_PORT, DEBUG

Usage:
    from app.core.config import settings
    settings.GROQ_API_KEY   # access any setting

The `settings` singleton is created at the bottom of this file and imported
everywhere in the app. All configured folders are also created at import time
via `os.makedirs(exist_ok=True)`.

Author : MeetingAI
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Project root — folder containing pyproject.toml
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    
    # ── Folder Paths ──────────────────────────────────────
    UPLOAD_FOLDER: str = str(ROOT_DIR / "uploads")
    OUTPUT_FOLDER: str = str(ROOT_DIR / "output")
    TRANSCRIPT_FOLDER: str = str(ROOT_DIR / "data" / "transcripts")
    LOG_FOLDER: str = str(ROOT_DIR / "logs")
    STATIC_FOLDER: str = str(ROOT_DIR / "static")
    MCP_CONFIG_FILE: str = os.getenv("MCP_CONFIG_FILE", r"C:\Users\Lenovo\meeting-summarizer\app\services\rag\mcp_server_file.json")
    

    # ── Audio Settings ────────────────────────────────────
    CHUNK_DURATION_MS: int = 5000       # how often browser sends audio chunk
    SAMPLE_RATE: int = 16000            # whisper expects 16khz
    AUDIO_FORMAT: str = "wav"           # format after conversion
    MAX_AUDIO_SIZE_MB: int = 25         # max upload size

    # ── Whisper / Transcription ───────────────────────────
    WHISPER_MODEL: str = "whisper-large-v3"         # base/small/medium
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"  # most efficient on cpu
    WHISPER_LANGUAGE: str = "en"

    # ── LLM / Summary ────────────────────────────────────
    LLM_PROVIDER: str = "groq"          # groq or ollama
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── App Settings ──────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Single instance used across entire app
settings = Settings()

# Ensure all folders exist on startup
for folder in [
    settings.UPLOAD_FOLDER,
    settings.OUTPUT_FOLDER,
    settings.TRANSCRIPT_FOLDER,
    settings.LOG_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)