"""
main.py - terminal chat for the CGPA agent.

  * Memory   : AsyncSqliteSaver (chats.sqlite). Your name is the thread_id.
  * Streaming: astream with ["updates", "messages"] - tokens print as they arrive,
               tool calls / results print as each node finishes.
  * Commands : /switch  /history  /forget  /quit
Run the MCP server first (cd ../server && python server.py), then: python main.py
"""

import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import MCP_URL, build_agent, load_tools
from memory import list_threads, repair_dangling, show_history

DB_PATH = str(Path(__file__).parent / "chats.sqlite")
REPORTS_DIR = Path(__file__).parent / "reports"


def save_local_copy(args: dict) -> None:
    """The server's save_report writes on the SERVER's disk. Keep a copy here too."""
    name = Path(str(args.get("filename") or "report")).name or "report"
    if not name.lower().endswith(".txt"):
        name += ".txt"
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / name).write_text(str(args.get("content", "")))
    print(f"\n  (local copy: reports/{name})", end="")


async def ask_name(memory) -> str:
    threads = await list_threads(memory)
    print("Saved chats:", ", ".join(threads) if threads else "(none yet)")
    while True:
        name = (await asyncio.to_thread(input, "Who is chatting? ")).strip()
        if name:
            return name


async def run_turn(app, config, text: str) -> None:
    """Stream one turn. Only the NEW message is sent; the checkpointer loads the rest."""
    inputs = {"messages": [HumanMessage(text)]}
    streaming_line = False  # are we in the middle of a streamed answer line?

    async for chunk in app.astream(
        inputs, config, stream_mode=["updates", "messages"], version="v2"
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]  # one piece of the model's answer
            if metadata["langgraph_node"] == "model" and token.text:
                if not streaming_line:
                    print("agent: ", end="", flush=True)
                    streaming_line = True
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            for node, update in chunk["data"].items():  # what that node returned
                if not update or "messages" not in update:
                    continue
                for m in update["messages"]:
                    if node == "model":
                        for tc in getattr(m, "tool_calls", []):
                            if streaming_line:
                                print()
                                streaming_line = False
                            print(f"  [calling {tc['name']} {tc['args']}]", flush=True)
                            if tc["name"] == "save_report":
                                save_local_copy(tc["args"])
                    elif node == "tools":
                        if streaming_line:
                            print()
                            streaming_line = False
                        print(f"  [result: {m.text}]", flush=True)
    print()


async def main() -> None:
    try:
        tools = await load_tools()
    except Exception as err:  # noqa: BLE001
        print(f"Cannot reach the MCP server at {MCP_URL} ({type(err).__name__}: {err}).")
        print("Start it (cd ../server && python server.py) or fix MCP_URL / FASTMCP_KEY in .env.")
        return
    print(f"Connected to {MCP_URL} - {len(tools)} tools: {', '.join(t.name for t in tools)}")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        app = build_agent(tools, memory)

        name = await ask_name(memory)
        config = {"configurable": {"thread_id": name}}
        print(f"Hi {name}. Commands: /switch  /history  /forget  /quit\n")

        while True:
            try:
                text = (await asyncio.to_thread(input, f"{name}: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/switch":
                name = await ask_name(memory)
                config = {"configurable": {"thread_id": name}}
                continue
            if text == "/history":
                await show_history(app, config)
                continue
            if text == "/forget":
                await memory.adelete_thread(name)
                print(f"  Forgot everything saved for '{name}'.")
                continue

            try:
                fixed = await repair_dangling(app, config)  # Task 6: heal old orphans
                if fixed:
                    print(f"  (repaired {fixed} unanswered tool call(s) from an earlier run)")
                await run_turn(app, config, text)
            except Exception as err:  # noqa: BLE001 - keep the chat alive
                print(f"\n  [error: {type(err).__name__}: {err}] - try again.")


if __name__ == "__main__":
    asyncio.run(main())
