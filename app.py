"""Streamlit chat UI for the CGPA agent. Run: streamlit run app.py"""

import asyncio
import os
import sqlite3
import time
import uuid
from pathlib import Path

import streamlit as st

try:  # Streamlit Cloud keeps keys in st.secrets, not .env
    for _k in ("GROQ_API_KEY", "MCP_URL", "FASTMCP_KEY"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_agent, load_tools
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

SUGGESTIONS = [
    ("Semester GPA", "Work out my GPA from this semester's marks", "I want to calculate my semester GPA from my marks."),
    ("New CGPA", "See where my CGPA lands after this semester", "What will my CGPA be after this semester?"),
    ("Target CGPA", "Find the GPA I need to hit my target", "What GPA do I need to reach my target CGPA?"),
]

st.set_page_config(page_title="CGPA Guider", page_icon=":material/school:", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none;}
.block-container {max-width: 760px; padding-top: 2.5rem; padding-bottom: 6rem;}

/* sidebar */
section[data-testid="stSidebar"] {background:#1f1e1d; border-right:1px solid #33322f; min-width:290px;}
section[data-testid="stSidebar"] .block-container {padding-top:1.4rem;}
.brand {font-family: Georgia, 'Times New Roman', serif; font-size:1.35rem; letter-spacing:.2px; margin-bottom:.9rem;}
.brand span {color:#d97757;}
.sect {font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; color:#8b8a85; margin:1.1rem 0 .35rem .2rem;}
section[data-testid="stSidebar"] .stButton > button {
  width:100%; justify-content:flex-start; text-align:left; border:0; background:transparent;
  color:#cfccc4; padding:.5rem .7rem; border-radius:9px; font-weight:400; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis;}
section[data-testid="stSidebar"] .stButton > button:hover {background:#2c2b29; color:#fff;}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {background:#33312e; color:#fff;}
section[data-testid="stSidebar"] .stButton > button p {overflow:hidden; text-overflow:ellipsis;}
.newchat button {border:1px solid #3b3a36 !important; justify-content:center !important;}

/* chat */
[data-testid="stChatMessage"] {background:transparent; border:0; padding:.5rem 0; gap:.9rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background:#30302e; border-radius:14px; padding:.75rem 1rem; margin:.4rem 0;}
[data-testid="stChatMessage"] p {line-height:1.65; font-size:1.02rem;}
[data-testid="stChatInput"] {border-radius:16px; border:1px solid #3b3a36; background:#30302e;}
[data-testid="stChatInput"] textarea {font-size:1rem;}

/* welcome */
.hello {font-family: Georgia, 'Times New Roman', serif; font-size:2.35rem; line-height:1.2;
        text-align:center; margin:12vh 0 .4rem;}
.hello span {color:#d97757;}
.sub {text-align:center; color:#9b9a94; margin-bottom:2.2rem;}
[data-testid="stMain"] .stButton > button, .main .stButton > button {
  width:100%; min-height:96px; text-align:left; justify-content:flex-start; align-items:flex-start;
  background:#2b2a28; border:1px solid #3b3a36; border-radius:14px; padding:.9rem 1rem; white-space:normal;}
[data-testid="stMain"] .stButton > button:hover {border-color:#d97757; background:#302f2c;}
.status {color:#8b8a85; font-size:.8rem; margin:.3rem 0 0 .2rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- titles (sidebar list)
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS conversations "
        "(thread_id TEXT PRIMARY KEY, owner TEXT, title TEXT, updated REAL)"
    )
    return con


def list_conversations(owner):
    con = db()
    try:
        return con.execute(
            "SELECT thread_id, title FROM conversations WHERE owner=? ORDER BY updated DESC", (owner,)
        ).fetchall()
    finally:
        con.close()


def touch_conversation(thread_id, owner, first_message):
    con = db()
    try:
        title = " ".join(first_message.split())
        title = title if len(title) <= 38 else title[:36] + "…"
        con.execute(
            "INSERT INTO conversations VALUES (?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET updated=?",
            (thread_id, owner, title, time.time(), time.time()),
        )
        con.commit()
    finally:
        con.close()


def drop_conversation(thread_id):
    con = db()
    try:
        con.execute("DELETE FROM conversations WHERE thread_id=?", (thread_id,))
        con.commit()
    finally:
        con.close()


async def with_agent(fn):
    tools = await load_tools()
    st.session_state["tool_count"] = len(tools)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as memory:
        return await fn(build_agent(tools, memory), memory)


def new_thread(owner):
    return f"{owner}:{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- sign in (just a name)
name = (st.query_params.get("u") or "").strip()

if not name:
    st.markdown('<div class="hello">Welcome to <span>CGPA Guider</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Tell me your name and I will keep your chats for next time.</div>', unsafe_allow_html=True)
    entered = st.text_input("Your name", placeholder="e.g. Talal", label_visibility="collapsed")
    if st.button("Continue", type="primary") and entered.strip():
        st.query_params["u"] = entered.strip()
        st.rerun()
    st.stop()

if "thread" not in st.session_state or not st.session_state["thread"].startswith(f"{name}:"):
    st.session_state["thread"] = new_thread(name)
thread = st.session_state["thread"]
config = {"configurable": {"thread_id": thread}}

# ---------------------------------------------------------------- sidebar
convs = list_conversations(name)
with st.sidebar:
    st.markdown('<div class="brand">CGPA <span>Guider</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="newchat">', unsafe_allow_html=True)
    if st.button("＋  New chat", key="new", use_container_width=True):
        st.session_state["thread"] = new_thread(name)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if convs:
        st.markdown('<div class="sect">Recent</div>', unsafe_allow_html=True)
    for tid, title in convs:
        row = st.columns([8, 1.3])
        if row[0].button(title, key=f"open_{tid}", type="primary" if tid == thread else "secondary",
                         use_container_width=True):
            st.session_state["thread"] = tid
            st.rerun()
        if row[1].button("✕", key=f"del_{tid}", help="Delete this chat"):
            async def _forget(app, memory, _tid=tid):
                await memory.adelete_thread(_tid)
            try:
                asyncio.run(with_agent(_forget))
            except Exception:
                pass
            drop_conversation(tid)
            if tid == thread:
                st.session_state["thread"] = new_thread(name)
            st.rerun()

    st.markdown('<div class="sect">Account</div>', unsafe_allow_html=True)
    if st.button(f"{name}  ·  switch", key="switch", use_container_width=True):
        st.query_params.clear()
        st.session_state.pop("thread", None)
        st.rerun()
    n = st.session_state.get("tool_count")
    st.markdown(f'<div class="status">{"● " + str(n) + " tools online" if n else "● connecting…"}</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- history of this chat
async def _history(app, memory):
    return await get_messages(app, config)

try:
    history = asyncio.run(with_agent(_history))
except Exception:
    st.error("Can't reach the calculator service right now. Please try again in a moment.")
    st.stop()

shown = [m for m in history if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and m.text.strip())]
USER_AV, BOT_AV = ":material/person:", ":material/school:"
for m in shown:
    user = isinstance(m, HumanMessage)
    with st.chat_message("user" if user else "assistant", avatar=USER_AV if user else BOT_AV):
        st.markdown(m.text)

if not shown:
    st.markdown(f'<div class="hello">Hi {name}, what are we <span>working on?</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Every number comes from a tool, never a guess.</div>', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTIONS))
    for col, (head, body, text) in zip(cols, SUGGESTIONS):
        if col.button(f"**{head}**\n\n{body}", key=f"s_{head}"):
            st.session_state["queued"] = text
            st.rerun()

# ---------------------------------------------------------------- new message
typed = st.chat_input("Ask about your GPA, CGPA or a target…")
prompt = typed or st.session_state.pop("queued", None)

if prompt:
    touch_conversation(thread, name, prompt)
    with st.chat_message("user", avatar=USER_AV):
        st.markdown(prompt)

    async def _turn(app, memory):
        await repair_dangling(app, config)
        with st.chat_message("assistant", avatar=BOT_AV):
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
        st.rerun()  # refresh the sidebar so the new chat title shows up
    except st.errors.StreamlitAPIException:
        raise
    except Exception as err:
        if type(err).__name__ in ("RerunException", "StopException"):
            raise
        st.error(f"Something went wrong ({type(err).__name__}). Please try again.")

for i, (fname, content) in enumerate(st.session_state.get("reports", [])):
    fname = Path(fname).name
    if not fname.endswith(".txt"):
        fname += ".txt"
    st.download_button(f"Download {fname}", content, file_name=fname, key=f"dl{i}")