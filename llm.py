"""llm.py - Groq chat model. Key comes from agent/.env (never commit it)."""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

llm = init_chat_model("groq:openai/gpt-oss-20b")
