import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from cartesia_tts_client import CartesiaTTSClient
from meditation_agent import generate_audio_guided_meditation_session, get_llm, MindfulnessAgent
from meditation_graph import run_mindfulness_graph


class Message(BaseModel):
    role: str
    content: str


class SessionRequest(BaseModel):
    query: str
    history: List[Message] = []


class SessionResponse(BaseModel):
    message: str
    transcript: Optional[str] = None


class AudioRequest(BaseModel):
    transcript: str


app = FastAPI(title="Mindfulness AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize any heavyweight resources here if needed."""
    # This keeps a simple LangChain-based agent available as a fallback.
    tool_map = {"generate_audio_guided_meditation_session": generate_audio_guided_meditation_session}
    app.state.mindfulness_agent = MindfulnessAgent(
        llm=get_llm(),
        history=[],
        tool_mapping=tool_map,
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


@app.post("/v1/mindfulness/audio")
async def generate_audio(body: AudioRequest):
    """
    Generate audio using Cartesia for a given transcript.
    Returns a streaming WAV response.
    """
    tts = CartesiaTTSClient()

    def audio_generator():
        for chunk in tts.stream_bytes(body.transcript):
            yield chunk

    headers: Dict[str, Any] = {
        "Content-Disposition": 'inline; filename="mindfulness.wav"',
    }

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers=headers,
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}

