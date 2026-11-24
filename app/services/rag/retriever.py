"""
app/services/rag/retriever.py
==============================
Semantic retrieval layer over the per-session ChromaDB vector store.

MeetingRetriever wraps MeetingVectorStore and provides a clean interface
for the LangGraph agent's search_transcript tool.

How it works:
    1. Checks if the vector store has any indexed sentences (count == 0 → early return).
    2. Calls MeetingVectorStore.similarity_search(query, k) which:
         a. Embeds the query using the shared FastEmbed ONNX model.
         b. Performs HNSW cosine similarity search.
         c. Returns the top-k LangChain Document objects.
    3. Formats results as "[chunk {id}] {sentence}" lines.
       The chunk ID prefix gives the LLM chronological context — it can
       distinguish early-meeting content from late-meeting content.
    4. Returns a RetrievalResult dataclass.

RetrievalResult dataclass:
    found     : bool            — False if store is empty or no results
    context   : str             — Formatted string ready for LLM consumption
    documents : list[Document]  — Raw LangChain Document objects
    count     : int             — Number of results returned

Default k=5 (top-5 results). Configurable at construction time.

The retriever is instantiated inside create_tools() (rag/tools.py) with a
closure over the session's MeetingVectorStore, ensuring each agent instance
searches only its own session's data.

Author : MeetingAI
"""

from dataclasses import dataclass
from langchain_core.documents import Document
from app.services.rag.store import MeetingVectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    found: bool
    context: str
    documents: list[Document]
    count: int


class MeetingRetriever:
    """
    LangChain-style retriever wrapper.
    Same pattern as your medical app retriever.
    """

    def __init__(self, store: MeetingVectorStore, k: int = 5):
        self._store = store
        self._k = k

    def retrieve(self, query: str) -> RetrievalResult:
        if self._store.count == 0:
            return RetrievalResult(
                found=False,
                context="",
                documents=[],
                count=0
            )

        docs = self._store.similarity_search(query, k=self._k)

        if not docs:
            return RetrievalResult(
                found=False,
                context="",
                documents=[],
                count=0
            )

        # format context with chunk numbers for chronological awareness
        lines = [
            f"[chunk {doc.metadata.get('chunk_id', 0)}] {doc.page_content}"
            for doc in docs
        ]

        return RetrievalResult(
            found=True,
            context="\n".join(lines),
            documents=docs,
            count=len(docs)
        )