"""
app/services/rag/__init__.py
=============================
Public API for the RAG (Retrieval-Augmented Generation) subsystem.

Exports the three primary components used by the WebSocket handler (audio.py):

    MeetingVectorStore
        Per-session in-memory ChromaDB vector store.
        Holds all embedded transcript sentences for one recording session.

    TranscriptIngestionPipeline
        Background producer-consumer pipeline.
        Accepts transcript text non-blockingly and indexes it into the store.

    run_agent
        Async function that runs the LangGraph ReAct agent for one Q&A turn.
        Lazily builds the agent graph on first call and caches it on the store.

Usage in audio.py:
    from app.services.rag import MeetingVectorStore, TranscriptIngestionPipeline, run_agent

Author : MeetingAI
"""

from app.services.rag.store import MeetingVectorStore
from app.services.rag.ingestion import TranscriptIngestionPipeline
from app.services.rag.agent import run_agent

__all__ = ["MeetingVectorStore", "TranscriptIngestionPipeline", "run_agent"]