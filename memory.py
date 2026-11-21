"""memory.py - helpers around the SQLite checkpointer (Task 3 + Task 6 repair)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


async def list_threads(memory) -> list[str]:
    """Every thread_id that has a saved checkpoint."""
    await memory.setup()  # creates the tables on a fresh database
    async with memory.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    ) as cur:
        return [row[0] async for row in cur]

async def get_messages(app, config) -> list:
    state = await app.aget_state(config)
    return state.values.get("messages", []) if state and state.values else []


async def show_history(app, config) -> None:
    messages = await get_messages(app, config)
    if not messages:
        print("  (no saved messages)")
        return
    for m in messages:
        if isinstance(m, HumanMessage):
            print(f"  you    : {m.text}")
        elif isinstance(m, AIMessage):
            for tc in m.tool_calls:
                print(f"  [call] : {tc['name']}({tc['args']})")
            if m.text.strip():
                print(f"  agent  : {m.text}")
        elif isinstance(m, ToolMessage):
            print(f"  [tool] : {m.text}")


def _dangling_ids(messages) -> list[tuple[str, str]]:
    """(tool_call_id, name) of tool calls that never got a ToolMessage."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    return [
        (tc["id"], tc["name"])
        for m in messages
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if tc["id"] not in answered
    ]


async def repair_dangling(app, config) -> int:
    """Task 6: a tool call with no answer (server died / Ctrl-C mid-run) makes Groq
    reject the whole chat forever. Answer each orphan with an error ToolMessage so the
    model is told the tool failed. Returns how many were repaired."""
    orphans = _dangling_ids(await get_messages(app, config))
    if orphans:
        await app.aupdate_state(
            config,
            {
                "messages": [
                    ToolMessage(
                        content="Error: this tool call was interrupted and never returned a result.",
                        tool_call_id=cid,
                        name=name,
                        status="error",
                    )
                    for cid, name in orphans
                ]
            },
        )
    return len(orphans)
