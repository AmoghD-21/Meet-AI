"""
app/core/exceptions.py
======================
Custom exception hierarchy for MeetingAI.

Design:
- `MyException` is the base class for all custom exceptions.
  Its `__init__` calls `error_message_detail()` which extracts the Python
  source filename and line number from `sys.exc_info()` traceback, producing
  highly precise log messages of the form:
      "Error in [<file>] at line [<N>]: <message>"

- Domain-specific subclasses inherit from `MyException` with no additional
  logic (the `pass` body is intentional — the type itself carries the domain).

Exception classes:
    AudioProcessingException    — ffmpeg conversion failures
    TranscriptionException      — Groq Whisper API failures
    SummaryGenerationException  — Groq LLM summarization failures
    FileProcessingException     — File upload / parsing failures
    WebSocketException          — WebSocket connection failures
    ModelLoadException          — Critical: AI model failed to load (app cannot start)

Usage:
    raise TranscriptionException("Groq returned empty response", sys)

Author : MeetingAI
"""

import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    # fix — handle case where traceback is None
    if exc_tb is None:
        error_message = f"Error: {str(error)}"
        logging.error(error_message)
        return error_message

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {str(error)}"
    )
    logging.error(error_message)
    return error_message

class MyException(Exception):
    """
    Custom exception class for handling errors.
    """
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the Exception with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        # Call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message

class AudioProcessingException(MyException):
    pass

class TranscriptionException(MyException):
    pass

class WebSocketException(MyException):
    pass

class SummaryGenerationException(MyException):
    pass

class ModelLoadException(MyException):
    pass
class FileProcessingException(MyException):
    """
    Raised when uploaded file processing fails.
    Example: unsupported format, file corrupted, PDF extraction fails
    """
    pass


class WebSocketException(MyException):
    """
    Raised when WebSocket connection has issues.
    Example: client disconnected unexpectedly, message too large
    """
    pass


class ModelLoadException(MyException):
    """
    Raised when AI model fails to load.
    Example: insufficient memory, model files missing
    This is CRITICAL — app cannot function without the model
    """
    pass