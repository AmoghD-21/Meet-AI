"""
app/services/rag/tools.py
==========================
LangChain tool definitions for the LangGraph ReAct agent.

This module defines all tools available to the agent and the MCP tool loader.
Tools are created as closures over the session's MeetingVectorStore, ensuring
each session's agent can only access its own transcript data.

Tools:

search_transcript(query: str) -> str
    Searches the session's in-memory ChromaDB for semantically similar content.
    Uses MeetingRetriever with k=5 (top-5 cosine similarity results).
    Returns formatted context with [chunk N] prefixes for chronological awareness.
    Returns a "not found" message with current indexed count if empty.
    The LLM is instructed (via docstring + system prompt) to call this FIRST
    for any meeting-related question.

web_search(query: str) -> str
    Performs a live web search using Tavily (max 3 results).
    Returns formatted "Source: URL\nContent: ..." blocks.
    Used for external facts not present in the meeting transcript.

_get_mcp_tools() -> list  [async]
    Loads Google Calendar tools from the MCP server defined in mcp_server_file.json.
    Uses MultiServerMCPClient from langchain-mcp-adapters to spawn the npx
    child process and fetch tools via the MCP stdio protocol.
    Converts MCP protocol tools into standard LangChain BaseTool objects.
    Returns [] on any failure (calendar is optional — agent degrades gracefully).

create_tools(store, mcp_tools=None) -> list
    Factory function called by _build_graph() in agent.py.
    Returns [search_transcript, web_search] + mcp_tools.
    MCP tools are passed in separately (already awaited before graph build).

Tool docstrings are critical:
    The LLM reads the @tool function's docstring to decide when to call it.
    Docstrings are written to be unambiguous about scope and priority.

Author : MeetingAI
"""

from langchain_core.tools import tool
from app.services.rag.retriever import MeetingRetriever
from app.services.rag.store import MeetingVectorStore
from app.utils.logger import get_logger
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.core.config import settings
import json

logger = get_logger(__name__)

async def _get_mcp_tools() -> list:
    """Fetch MCP tools async. Returns empty list if MCP unavailable."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        import json
        with open(settings.MCP_CONFIG_FILE) as f:
            servers = json.load(f)
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
        logger.info(f"MCP tools loaded: {[t.name for t in tools]}")
        return tools
    except Exception as e:
        logger.warning(f"MCP tools unavailable: {e}")
        return []

def create_tools(store: MeetingVectorStore , mcp_tools: list = None) -> list:
    retriever = MeetingRetriever(store, k=5)

    @tool
    def search_transcript(query: str) -> str:
        """
        Search the meeting transcript for relevant information.
        Use for ANY question about what was said, discussed,
        decided, assigned, or mentioned during the meeting.
        Always use this first for meeting-related questions.
        """
        result = retriever.retrieve(query)

        if not result.found:
            return (
                f"No relevant content found. "
                f"{store.count} sentences indexed so far."
            )

        logger.info(f"search_transcript: '{query[:40]}' - {result.count} results")
        return f"Relevant meeting content:\n{result.context}"

    @tool
    def web_search(query: str):
        """
        use this tool to search the web for information NOT in the meeting.
        also if any question explicitly asks for "web search" or "search the web" or "external information".
        """
        try:
            tavily = TavilySearchResults(max_results=3)
            results = tavily.invoke(query)
            
            
            formatted_results = []
            for r in results:
                formatted_results.append(
                    f"Source: {r.get('url')}\nContent: {r.get('content')}"
                )
            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Error searching web: {e}"

    return [search_transcript, web_search]+(mcp_tools or [])