from typing import Dict, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from workflow import run_mindfulness_graph

class Message(BaseModel):
    role: str
    content: str


class SessionRequest(BaseModel):
    query: str
    history: List[Message] = []


class SessionResponse(BaseModel):
    message: str
    transcript: Optional[str] = None


app = FastAPI(title="Mindfulness AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/mindfulness/session", response_model=SessionResponse)
async def create_session(body: SessionRequest) -> SessionResponse:
    """
    Main entry point for the mindfulness agent.

    For now this uses the LangGraph-based agent for conversational text and
    the existing meditation tool for generating audio metadata.
    """
    history_msgs = [HumanMessage(content=m.content) for m in body.history if m.role == "user"]

    result = run_mindfulness_graph(
        query=body.query,
        history=history_msgs,
    )

    return SessionResponse(
        message=result["message"],
        transcript=result.get("transcript"),
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
