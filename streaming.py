"""streaming.py - turns app.astream(...) into simple events for a UI (used by app.py)."""

from langchain_core.messages import HumanMessage


async def stream_events(app, config, text: str):
    """Yield ("token", str) | ("call", name, args) | ("result", str)."""
    async for chunk in app.astream(
        {"messages": [HumanMessage(text)]},
        config,
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata["langgraph_node"] == "model" and token.text:
                yield ("token", token.text)
        elif chunk["type"] == "updates":
            for node, update in chunk["data"].items():
                if not update or "messages" not in update:
                    continue
                for m in update["messages"]:
                    if node == "model":
                        for tc in getattr(m, "tool_calls", []):
                            yield ("call", tc["name"], tc["args"])
                    elif node == "tools":
                        yield ("result", m.text)
