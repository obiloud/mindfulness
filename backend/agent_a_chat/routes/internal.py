"""
Internal module for Agent A Chat API.
Handles synthesis callbacks and internal webhooks.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

logger = logging.getLogger(__name__)

# === Schemas ===


class SynthesisResult(BaseModel):
    thread_id: str
    answer: str
    transcript: str
    chapters: Optional[list] = None
    status: str = "completed"

# === Internal Routes ===


@router.post("/v1/synthesis-callback")
async def handle_synthesis_complete(result: SynthesisResult, request: Request):
    """
    Receives the finished transcript and writes it to the BaseStore.
    This avoids the LangGraph checkpointer lock during active graph runs.
    """
    store = request.app.state.store

    thread_id = result.thread_id
    answer = result.answer
    transcript = result.transcript
    chapters = result.chapters
    status = result.status

    if status != "completed":
        logger.error(f"Synthesis failed for thread {thread_id}")
        return {"status": "error"}

    namespace = ("a2a", thread_id, "pending_updates")
    item_key = f"synth_{result.task_id}" if hasattr(
        result, 'task_id') else "latest_synthesis"

    try:
        await store.aput(
            namespace,
            item_key,
            {
                "answer": answer,
                "transcript": transcript,
                "chapters": chapters,
                "synth_status": "completed",
                "is_synthesis_ready": True
            }
        )

        logger.info(
            f"Artifact stored in BaseStore for thread {thread_id}. Bypassed checkpointer lock."
        )

    except Exception as e:
        logger.error(f"Failed to write to BaseStore: {e}")
        return {"status": "error", "message": str(e)}

    return {"status": "acknowledged"}
