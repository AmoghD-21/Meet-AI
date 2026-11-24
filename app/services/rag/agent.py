"""
app/services/rag/agent.py
==========================
LangGraph ReAct agent for intelligent meeting Q&A.

Implements a stateful tool-calling agent using LangGraph's StateGraph.
The agent can answer questions about the meeting transcript, search the web,
and manage Google Calendar events — all within a single conversation turn.

Agent LLM:
    GPT-4o via GitHub Models (Azure inference endpoint).
    Used over Groq LLaMA for superior tool-calling reliability.
    Temperature 0.3 for consistent, factual responses.

Graph Structure (ReAct pattern):
    START → call_llm → tools_condition → tools → call_llm → ...
                                       → END (when no tool call)

    - call_llm   : Invokes GPT-4o with current message history.
    - tools      : ToolNode — executes the tool chosen by the LLM.
    - tools_condition : Prebuilt LangGraph router — checks if the last
                         AIMessage contains tool_calls. Routes to "tools"
                         if yes, END if no.
    - The tools → call_llm back-edge creates the ReAct loop.
    - recursion_limit=10 caps the loop at 10 node visits (5 tool calls max).

AgentState:
    TypedDict with a single key `messages: Annotated[list, add_messages]`.
    add_messages is LangGraph's reducer — safely merges messages from
    multiple nodes without overwriting.

System Prompt:
    Generated dynamically with today's date (for calendar date calculations).
    Instructs the LLM on tool selection priority:
      1. search_transcript — for ANY meeting content question
      2. web_search        — for external facts
      3. Calendar tools    — for create/list/update/delete events

Session Caching:
    The compiled graph is attached to the MeetingVectorStore instance as
    `store._agent`. Built lazily on the first Q&A request of the session.
    MCP server startup (5-8s) only happens once per session this way.

Async:
    run_agent() uses ainvoke (not invoke) because MCP tools are async-only.
    The entire agent execution is awaitable and non-blocking.

Author : MeetingAI
"""

import asyncio
import os
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from app.services.rag.store import MeetingVectorStore
from app.services.rag.tools import create_tools , _get_mcp_tools
from app.core.config import settings
from datetime import datetime
from app.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

def get_system_prompt():
    today = datetime.now().strftime("%A, %B %d, %Y")
    return f"""You are an intelligent meeting assistant.
    Today's date is {today}. Use this for any date calculations like "upcoming Sunday", "next Monday", etc.

    Tool selection:
    - search_transcript — anything about meeting content
    - web_search — external facts not in the meeting
    - create_event — create a Google Calendar event
    - list_events — list Google Calendar events
    - update_event — update an existing calendar event
    - delete_event — delete a calendar event

    Always use search_transcript first for meeting questions.
    For calendar operations, calculate exact dates based on today's date.
    Be concise. Base answers only on tool results."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def _build_graph(store: MeetingVectorStore, mcp_tools: list = None):
    tools = create_tools(store)
    tools.extend(mcp_tools)  # Add MCP tools to the list
    tool_node = ToolNode(tools)

    llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("GITHUB_TOKEN"), 
    base_url="https://models.inference.ai.azure.com"
    ).bind_tools(tools)

    def call_llm(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("tools", tool_node)        # must be named "tools" for tools_condition
    graph.set_entry_point("call_llm")
    graph.add_conditional_edges("call_llm", tools_condition)  # prebuilt condition
    graph.add_edge("tools", "call_llm")
    return graph.compile()


async def run_agent(question: str, store: MeetingVectorStore) -> str:
    try:
        if not hasattr(store, '_agent'):
            mcp_tools = await _get_mcp_tools()
            store._agent = _build_graph(store, mcp_tools)

        # ainvoke — required because MCP tools are async-only
        result = await store._agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=get_system_prompt()),
                    HumanMessage(content=question)
                ]
            },
            config={"recursion_limit": 10}
        )

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content

        return "Could not generate an answer."

    except Exception as e:
        logger.error(f"Agent error: {e}")
        return "Agent error. Please try again."