"""
app/utils/logger.py
====================
Centralised logging configuration for MeetingAI.

Sets up the Python root logger with two handlers at module import time:

RotatingFileHandler (DEBUG level):
    - Writes ALL log messages (DEBUG and above) to a timestamped file in /logs.
    - File name format: MM_DD_YYYY_HH_MM_SS.log
    - Max file size: 5 MB per file.
    - Backup count: 3 (keeps last 3 rotated files alongside the current one).
    - Useful for post-incident debugging and audit trails.

StreamHandler (INFO level):
    - Writes INFO and above to stdout (console).
    - UTF-8 encoding with line buffering (buffering=1) for real-time output.
    - Hides DEBUG noise from the terminal during normal operation.

Log format:
    [ YYYY-MM-DD HH:MM:SS,mmm ] <module_name> - <LEVEL> - <message>

Usage (in every module):
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Session started")
    logger.error("Transcription failed: ...")
    logger.debug("Chunk 3 — 4096 bytes")

configure_logger() is called automatically at import time (bottom of this file).
Call get_logger(__name__) in every module — do NOT call configure_logger() again.

Author : MeetingAI
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

# Constants
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5MB
BACKUP_COUNT = 3

# Find project root — the folder containing pyproject.toml
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
log_dir_path = PROJECT_ROOT / LOG_DIR
log_dir_path.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_path / LOG_FILE

def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
    )

    # File handler — captures everything
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=MAX_LOG_SIZE, 
        backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler — only INFO and above
    console_handler = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    )
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Call this in every module.
    Usage: logger = get_logger(__name__)
    """
    return logging.getLogger(name)

# Configure once at import time
configure_logger()