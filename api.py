from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from main import build_agent
from langgraph.prebuilt import ToolNode
from langchain_core.messages import ToolMessage
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import FileResponse
import traceback

# Configuración App


app = FastAPI(title="Chat Orchestrator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
AGENT = build_agent()

#  Modelos I/O


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message in natural language")
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque thread/session identifier; server will generate one if absent",
    )


class ToolEvent(BaseModel):
    name: str
    content: Any


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_events: List[ToolEvent] = []
    raw: Optional[Dict[str, Any]] = None


class NewSessionResponse(BaseModel):
    session_id: str


# Endpoints


@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions/new", response_model=NewSessionResponse, tags=["sessions"])
def new_session() -> NewSessionResponse:
    return NewSessionResponse(session_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest) -> ChatResponse:
    """
    Síncrono: procesa un mensaje del usuario y devuelve la última respuesta del agente.
    Usa `session_id` para mantener el hilo conversacional en LangGraph (checkpointing).
    """
    session_id = req.session_id or str(uuid.uuid4())

    try:
        state = AGENT.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {e}\n{tb}")

    messages = state.get("messages", [])
    if not messages:
        return ChatResponse(
            reply="(no response)",
            session_id=session_id,
            tool_events=[],
        )

    tool_events: List[ToolEvent] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            tool_events.append(ToolEvent(name=m.name or "tool", content=m.content))

    last = messages[-1]
    if isinstance(last, ToolMessage):
        reply_text = f"[tool {last.name}] {last.content}"
    else:
        content = getattr(last, "content", "")
        if isinstance(content, list):
            reply_text = " ".join(
                [c if isinstance(c, str) else str(c) for c in content]
            ).strip()
        elif isinstance(content, (dict,)):
            reply_text = str(content)
        else:
            reply_text = str(content)

    return ChatResponse(
        reply=reply_text or "(empty)",
        session_id=session_id,
        tool_events=tool_events,
        raw=None,
    )


BASE_DIR = Path(__file__).parent.resolve()
FILES_DIR = BASE_DIR / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)
(FILES_DIR / "csv").mkdir(parents=True, exist_ok=True)
(FILES_DIR / "excel").mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=str(FILES_DIR)), name="files")

INDEX_PATH = BASE_DIR / "public/index.html"


@app.get("/", include_in_schema=False)
def root():
    if not INDEX_PATH.exists():
        return {"error": f"index.html no encontrado en {INDEX_PATH}"}
    return FileResponse(str(INDEX_PATH))
