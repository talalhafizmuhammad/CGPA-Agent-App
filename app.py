"""
app.py - Streamlit chat UI for the CGPA agent (bonus variant of main.py).

Same agent, same MCP server, same SQLite memory (chats.sqlite). Name in the sidebar =
thread_id. Run:  streamlit run app.py
Config: MCP_URL / FASTMCP_KEY / GROQ_API_KEY from .env, or from st.secrets when deployed.
"""

import asyncio
import os
import sqlite3
from pathlib import Path

import streamlit as st

# Streamlit Cloud has no .env: copy secrets into the environment BEFORE agent.py reads them.
try:
    for _k in ("GROQ_API_KEY", "MCP_URL", "FASTMCP_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass  # no secrets.toml locally - .env is used

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import MCP_URL, build_agent, load_tools
from memory import get_messages, repair_dangling
from streaming import stream_events

DB_PATH = str(Path(__file__).parent / "chats.sqlite")

st.set_page_config(page_title="PUCIT CGPA Guider", layout="wide")


def saved_threads() -> list[str]:
    if not Path(DB_PATH).exists():
        return []
    con = sqlite3.connect(DB_PATH)
    try:
        return [r[0] for r in con.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")]
    except sqlite3.Error:
        return []
    finally:
        con.close()


async def with_agent(fn):
    """Open the checkpointer + tools, run fn(app, memory), close. Streamlit reruns the
    script on every interaction, so each action gets its own short-lived connection."""
    tools = await load_tools()
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        return await fn(build_agent(tools, memory), memory)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("CGPA Guider")
    threads = saved_threads()
    pick = st.selectbox("Saved chats", ["(new)"] + threads)
    typed = st.text_input("Who is chatting? (this is your thread_id)", value="" if pick == "(new)" else pick)
    name = typed.strip()
    st.caption(f"MCP server: {MCP_URL}")
    if name and name in threads and st.button("Forget this chat", use_container_width=True):
        async def _forget(app, memory):
            await memory.adelete_thread(name)
        try:
            asyncio.run(with_agent(_forget))
            st.session_state.pop("reports", None)
            st.rerun()
        except Exception as err:
            st.error(f"Could not forget: {type(err).__name__}: {err}")

st.title("PUCIT GPA / CGPA Assistant")
if not name:
    st.info("Enter your name in the sidebar to start. Your chat is saved under that name.")
    st.stop()
config = {"configurable": {"thread_id": name}}

# ---------------------------------------------------------------- saved history
async def _history(app, memory):
    return await get_messages(app, config)

try:
    history = asyncio.run(with_agent(_history))
except Exception as err:
    st.error(f"Cannot reach the MCP server at {MCP_URL} ({type(err).__name__}). Start it or fix MCP_URL.")
    st.stop()

for m in history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.text)
    elif isinstance(m, AIMessage) and m.text.strip():
        with st.chat_message("assistant"):
            st.markdown(m.text)

# ---------------------------------------------------------------- new message
if prompt := st.chat_input("Ask about your GPA, CGPA, or a target CGPA..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    async def _turn(app, memory):
        fixed = await repair_dangling(app, config)  # heal orphaned tool calls (Task 6)
        with st.chat_message("assistant"):
            status = st.status("Working it out...", expanded=False)
            box = st.empty()
            text = ""
            async for ev in stream_events(app, config, prompt):
                if ev[0] == "token":
                    text += ev[1]
                    box.markdown(text + "▌")  # live typing
                elif ev[0] == "call":
                    status.write(f"Calling `{ev[1]}` {ev[2]}")
                    if ev[1] == "save_report":
                        st.session_state.setdefault("reports", []).append(
                            (str(ev[2].get("filename", "report")), str(ev[2].get("content", "")))
                        )
                elif ev[0] == "result":
                    status.write(f"Result: {ev[1]}")
                    text = ""  # the next model message is a fresh answer
            box.markdown(text)
            status.update(label="Done", state="complete")
        return fixed

    try:
        asyncio.run(with_agent(_turn))
    except Exception as err:  # server down, rate limit, ... - keep the app alive
        st.error(f"{type(err).__name__}: {err} - try again.")

for i, (fname, content) in enumerate(st.session_state.get("reports", [])):
    fname = Path(fname).name
    st.download_button(
        f"Download {fname if fname.endswith('.txt') else fname + '.txt'}",
        content, file_name=fname if fname.endswith(".txt") else fname + ".txt", key=f"dl{i}",
    )
