"""
test_offline.py - end-to-end check of memory, streaming and server-down handling with a
scripted fake model (no Groq key needed). Needs the MCP server running on MCP_URL.
The fake model behaves like Groq in one way that matters: it REJECTS a history that has
a tool call with no answer.

    python test_offline.py
"""

import asyncio
import os
import tempfile
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_agent, load_tools
from main import run_turn
from memory import get_messages, list_threads, repair_dangling


class FakeGroq(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kw):
        return self

    def _next(self, messages) -> AIMessage:
        # strict like Groq: every tool call must be answered before the next message
        answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    if tc["id"] not in answered:
                        raise ValueError(f"400: tool call {tc['id']} has no tool message")
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(f"Tool said: {last.content}")
        if "remaining" in last.text:
            return AIMessage("", tool_calls=[{
                "name": "get_remaining_credit_hours", "args": {"current_semester": 4},
                "id": f"call_{len(messages)}"}])
        n = sum(isinstance(m, HumanMessage) for m in messages)
        return AIMessage(f"You have sent {n} message(s) in this chat.")

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(self, messages, stop=None, run_manager=None, **kw):
        msg = self._next(messages)
        if msg.tool_calls:
            yield ChatGenerationChunk(message=AIMessageChunk(
                content="", tool_call_chunks=[{
                    "name": msg.tool_calls[0]["name"], "args": '{"current_semester": 4}',
                    "id": msg.tool_calls[0]["id"], "index": 0}]))
        else:
            for w in msg.text.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=w + " "))


async def main():
    tools = await load_tools()
    print("tools:", [t.name for t in tools])
    db = os.path.join(tempfile.mkdtemp(), "t.sqlite")
    cfg = {"configurable": {"thread_id": "talal"}}

    # 1. memory + streaming
    async with AsyncSqliteSaver.from_conn_string(db) as mem:
        app = build_agent(tools, mem, model=FakeGroq())
        await run_turn(app, cfg, "hello")
        await run_turn(app, cfg, "remaining credit hours?")
    # 2. new connection = "closed the program" -> history must survive
    async with AsyncSqliteSaver.from_conn_string(db) as mem:
        app = build_agent(tools, mem, model=FakeGroq())
        assert await list_threads(mem) == ["talal"], await list_threads(mem)
        n = len(await get_messages(app, cfg))
        assert n == 6, n  # human, ai, human, ai(tool call), tool, ai
        await run_turn(app, cfg, "again")
        msgs = await get_messages(app, cfg)
        assert "3 message" in msgs[-1].text, msgs[-1].text
        print("memory survives restart: OK")

        # 3. server down (caller stops it before this step)
        print("--- now STOP the server, press Enter ---") if os.getenv("MANUAL") else None
        os.system("pkill -f 'python server.py'")
        await asyncio.sleep(1)
        await run_turn(app, cfg, "remaining credit hours?")  # must NOT crash
        msgs = await get_messages(app, cfg)
        err = [m for m in msgs if isinstance(m, ToolMessage) and m.status == "error"]
        assert err, "expected an error ToolMessage"
        await run_turn(app, cfg, "still there?")  # history is valid, chat not stuck
        print("server down handled: OK")

        # 4. orphan tool call (e.g. Ctrl-C mid-run, or saved by the OLD code)
        await app.aupdate_state(cfg, {"messages": [AIMessage("", tool_calls=[
            {"name": "calculate_new_cgpa", "args": {}, "id": "orphan_1"}])]})
        try:
            await run_turn(app, cfg, "hi")
            raise SystemExit("expected the strict model to reject the orphan")
        except ValueError:
            print("orphan breaks the chat without repair (as described in Task 6)")
        assert await repair_dangling(app, cfg) == 1
        await run_turn(app, cfg, "hi after repair")
        print("orphan repaired: OK")


if __name__ == "__main__":
    asyncio.run(main())
