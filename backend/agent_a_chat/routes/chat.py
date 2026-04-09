"""
Chat module for Agent A Chat API.
Handles conversation endpoints and message history.
"""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from langchain.messages import HumanMessage
from uuid import uuid4
from pydantic import BaseModel
from agent_a_chat.state import print_state
from agent_a_chat.routes.authentication import get_current_user_dict
from a2a.types import (
    SendMessageRequest,
    MessageSendConfiguration,
    MessageSendParams,
    PushNotificationConfig,
    Message,
    Role,
    Part,
    TextPart,
    DataPart
)

router = APIRouter()

logger = logging.getLogger(__name__)

# === Schemas ===


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    user_id: str
    answer: Optional[str] = None
    transcript: Optional[str] = None
    chapters: Optional[list] = None

# === Message Saving Helper ===


async def save_message(thread_id: str, user_id: str, role: str, content: str, db_pool):
    """Save message to database."""
    if not db_pool:
        logger.warning("Database pool not available. Cannot save message.")
        return
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute(
                    """
                    INSERT INTO messages (id, thread_id, user_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (str(uuid4()), thread_id, user_id, role, content)
                )
                logger.debug(f"Saved message: {role} -> {content[:50]}...")
            except Exception as e:
                logger.error(f"Failed to save message: {e}", exc_info=True)

# === Chat Routes ===


@router.get("/messages/history/{thread_id}")
async def get_message_history(thread_id: str, request: Request = None, current_user: dict = Depends(get_current_user_dict)):
    """
    Retrieve conversation history for a thread.
    """
    if not request or not request.app.state.db_pool:
        raise HTTPException(
            status_code=500, detail="Database pool not available")

    user_id = current_user.get("user_id")

    async with request.app.state.db_pool.connection() as conn:
        result = await conn.execute(
            """
            SELECT thread_id, user_id, role, content, created_at
            FROM messages
            WHERE thread_id = %s AND user_id = %s
            ORDER BY created_at ASC
            """,
            (thread_id, user_id)
        )
        messages = []
        for row in result.fetchall():
            messages.append({
                "thread_id": row["thread_id"],
                "user_id": row["user_id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"]
            })
    return {"thread_id": thread_id, "messages": messages}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request, bg_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user_dict)):
    """
    Handles user messages, advances the fast chat graph, and evaluates
    the patience loop to trigger the heavy-compute synthesis graph.
    """
    graph = request.app.state.chat_graph
    a2a_client = request.app.state.a2a_client
    store = request.app.state.store
    context = request.app.state.context

    thread_id = str(uuid4()) if body.thread_id is None else body.thread_id
    user_id = current_user.get("user_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    try:
        state = await graph.ainvoke({"messages": [HumanMessage(content=body.message)]}, config=config, context=context)
        logger.info(f"final_state: {print_state(state)}")

        synth_status = state.get("synth_status")

        if synth_status == "requested":
            summary = state.get("summary", "")
            namespace = ("preferences", user_id)

            data = {"thread_id": thread_id, "user_id": user_id}

            items = await store.asearch(namespace)
            if items:
                for item in items:
                    data[item.key] = item.value

            bg_tasks.add_task(
                a2a_client.send_message,
                SendMessageRequest(id=str(uuid4()), params=MessageSendParams(
                    configuration=MessageSendConfiguration(
                        push_notification_config=PushNotificationConfig(
                            url="http://chat-agent:8000/internal/v1/synthesis-callback")
                    ),
                    message=Message(
                        message_id=str(uuid4()),
                        role=Role('user'),
                        parts=[
                            Part(root=TextPart(text=summary)),
                            Part(root=DataPart(data=data))
                        ])
                ))
            )
            await graph.aupdate_state(config, {"synth_status": "in_progress"})

        last_message = state["messages"][-1]
        reply = last_message.content
        answer = last_message.additional_kwargs.get("answer")
        transcript = last_message.additional_kwargs.get("transcript")
        chapters = last_message.additional_kwargs.get("chapters")

        await save_message(thread_id, user_id, "user", body.message, request.app.state.db_pool)
        await save_message(thread_id, user_id, "ai", reply, request.app.state.db_pool)

        return ChatResponse(
            reply=reply,
            user_id=user_id,
            thread_id=thread_id,
            answer=answer,
            transcript=transcript,
            chapters=chapters
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
