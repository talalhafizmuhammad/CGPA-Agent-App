"""
agent.py - builds the CGPA agent (create_agent version).

Differences from the old version:
  * tools are NOT imported from tools.py any more - they come from the MCP server
    (see load_tools), so this file has no idea what language the tools are in.
  * the agent is compiled with a checkpointer (memory), so it remembers a thread.
  * a tool-error middleware turns "server is down" into a ToolMessage the model reads
    instead of an exception that crashes the chat (Task 6).
"""

import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

warnings.filterwarnings("ignore", message=".*langchain.mcp.*beta.*")
from langchain.mcp import MCPAdapter  # noqa: E402
from fastmcp import Client  # noqa: E402

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
KEY = os.getenv("FASTMCP_KEY") or None  # only needed if the host requires auth

SYSTEM_PROMPT = (Path(__file__).parent / "prompt.txt").read_text()


async def load_tools():
    """Connect to the MCP server, list its tools, disconnect.

    Each returned tool opens a fresh connection when called, so a server restart
    between questions is fine.
    """
    target = Client(MCP_URL, auth=KEY) if KEY else MCP_URL
    return await MCPAdapter(target).list_tools()


@wrap_tool_call
async def tool_errors_to_model(request, handler):
    """If a tool call blows up (server down, timeout, auth...), answer the tool call
    with an error ToolMessage. Every tool call therefore always gets an answer, so
    the saved history is never left with a dangling call, and the model can tell
    the student what happened."""
    try:
        return await handler(request)
    except Exception as err:  # noqa: BLE001 - any failure must become a ToolMessage
        return ToolMessage(
            content=(
                f"Error: the CGPA tool server could not be reached or failed "
                f"({type(err).__name__}). The tool did not run and returned no result. "
                f"Tell the student the calculator is unavailable right now; do not guess a number."
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


_BAD_TOOL_CALL = ("tool call validation failed", "tool_use_failed", "failed to call a function")


@wrap_model_call
async def retry_bad_tool_call(request, handler):
    """Groq sometimes rejects the model's own tool call (e.g. empty arguments) with a
    400 before it ever reaches a tool. Retry once; if it fails again, ask the student
    for the details instead of crashing the chat. Other errors are re-raised."""
    for _ in range(2):
        try:
            return await handler(request)
        except Exception as err:  # noqa: BLE001
            if not any(k in str(err).lower() for k in _BAD_TOOL_CALL):
                raise
    return ModelResponse(
        result=[
            AIMessage(
                "I couldn't set that calculation up yet. Could you give me the numbers it "
                "needs, for example your current CGPA and completed credit hours?"
            )
        ]
    )


def build_agent(tools, checkpointer, model=None):
    if model is None:
        from llm import llm as model

    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[tool_errors_to_model, retry_bad_tool_call],
        checkpointer=checkpointer,
    )