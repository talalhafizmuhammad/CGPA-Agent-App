"""Streamlit chat UI for the CGPA agent. Run: streamlit run app.py"""

import asyncio
import os
import sqlite3
from pathlib import Path

import streamlit as st

# On Streamlit Cloud the keys live in st.secrets, not .env
try:
    for _k in ("GROQ_API_KEY", "MCP_URL", "FASTMCP_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import MCP_URL, build_agent, load_tools
from memory import get_messages, repair_dangling
from streaming import stream_events

DB_PATH = str(Path(__file__).parent / "chats.sqlite")

TOOL_LABELS = {
    "marks_to_grade_points": "Converting marks to grade points",
    "calculate_semester_gpa": "Calculating semester GPA",
    "calculate_new_cgpa": "Projecting new CGPA",
    "required_gpa_for_target": "Working out the GPA you need",
    "get_semester_courses": "Looking up semester courses",
    "get_remaining_credit_hours": "Counting remaining credit hours",
    "save_report": "Saving your report",
}

STARTERS = [
    ("📘 Semester GPA", "I want to calculate my semester GPA from my marks."),
    ("📈 New CGPA", "What will my CGPA be after this semester?"),
    ("🎯 Target CGPA", "What GPA do I need to reach my target CGPA?"),
]

st.set_page_config(page_title="PUCIT CGPA Guider", page_icon="🎓", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 780px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%);}
.hero h1 {margin: 0; font-size: 1.75rem; color: #fff;}
.hero p {margin: .35rem 0 0; color: #d5e4f7; font-size: .98rem;}
.pill {display: inline-block; padding: .15rem .65rem; border-radius: 999px;
       font-size: .78rem; background: rgba(45,212,191,.15); color: #2dd4bf;}
[data-testid="stChatMessage"] {border-radius: 14px; padding: .8rem 1rem;
       background: #111a2e; border: 1px solid #1c2944;}
section[data-testid="stSidebar"] {border-right: 1px solid #1c2944;}
div.stButton > button {border-radius: 12px; width: 100%;}
</style>
""",
    unsafe_allow_html=True,
)


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
    # fresh connection per action; Streamlit reruns the whole script each time
    tools = await load_tools()
    st.session_state["tool_count"] = len(tools)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        return await fn(build_agent(tools, memory), memory)


st.markdown(
    '<div class="hero"><h1>🎓 PUCIT CGPA Guider</h1>'
    "<p>Semester GPA, projected CGPA and target planning. Every number comes from a tool, never a guess.</p></div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Your chat")
    threads = saved_threads()
    pick = st.selectbox("Saved chats", ["(new)"] + threads)
    name = st.text_input(
        "Your name", value="" if pick == "(new)" else pick, placeholder="e.g. talal",
        help="Used as your chat id. Use the same name later to get your history back.",
    ).strip()
    if name and name in threads:
        if st.button("🗑️ Forget this chat"):
            async def _forget(app, memory):
                await memory.adelete_thread(name)
            try:
                asyncio.run(with_agent(_forget))
                st.session_state.pop("reports", None)
                st.rerun()
            except Exception as err:
                st.error(f"Could not forget: {type(err).__name__}")
    st.divider()
    n = st.session_state.get("tool_count")
    st.markdown(
        f'<span class="pill">● server connected · {n} tools</span>' if n
        else '<span class="pill">● server not checked yet</span>',
        unsafe_allow_html=True,
    )
    st.caption(MCP_URL)

if not name:
    st.info("👈 Enter your name in the sidebar to start. Your chat is saved under that name.")
    st.stop()

config = {"configurable": {"thread_id": name}}

# ------------------------------------------------------------ history
async def _history(app, memory):
    return await get_messages(app, config)

try:
    history = asyncio.run(with_agent(_history))
except Exception as err:
    st.error(f"Can't reach the tool server ({type(err).__name__}). Try again in a moment.")
    st.stop()

shown = [m for m in history if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.text.strip())]
for m in shown:
    with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant",
                         avatar="🧑‍🎓" if isinstance(m, HumanMessage) else "🎓"):
        st.markdown(m.text)

if not shown:
    st.markdown(f"#### Hi {name}, what would you like to work on?")
    cols = st.columns(len(STARTERS))
    for col, (label, text) in zip(cols, STARTERS):
        if col.button(label, key=f"starter_{label}"):
            st.session_state["queued"] = text
            st.rerun()

# ------------------------------------------------------------ new message
typed = st.chat_input("Ask about your GPA, CGPA or a target CGPA…")
prompt = typed or st.session_state.pop("queued", None)

if prompt:
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    async def _turn(app, memory):
        await repair_dangling(app, config)
        with st.chat_message("assistant", avatar="🎓"):
            status = st.status("Thinking…", expanded=False)
            box = st.empty()
            text = ""
            async for ev in stream_events(app, config, prompt):
                if ev[0] == "token":
                    text += ev[1]
                    box.markdown(text + "▌")
                elif ev[0] == "call":
                    status.update(label=TOOL_LABELS.get(ev[1], ev[1]) + "…")
                    status.write(f"`{ev[1]}` {ev[2]}")
                    if ev[1] == "save_report":
                        st.session_state.setdefault("reports", []).append(
                            (str(ev[2].get("filename", "report")), str(ev[2].get("content", "")))
                        )
                elif ev[0] == "result":
                    status.write(f"→ {ev[1]}")
                    text = ""
            box.markdown(text)
            status.update(label="Done", state="complete")

    try:
        asyncio.run(with_agent(_turn))
    except Exception as err:
        st.error(f"Something went wrong ({type(err).__name__}). Please try again.")

for i, (fname, content) in enumerate(st.session_state.get("reports", [])):
    fname = Path(fname).name
    if not fname.endswith(".txt"):
        fname += ".txt"
    st.download_button(f"⬇️ Download {fname}", content, file_name=fname, key=f"dl{i}")