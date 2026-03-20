# agent_a_chat/main.py
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# LangGraph & Checkpointing
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from agent_a_chat.graph import get_llm, create_chat_graph

# A2A SDK
from a2a.client import A2AClient

from shared.settings import get_settings
from agent_a_chat.state import GraphContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as httpx_client:

        app.state.a2a_client = A2AClient(
            url=s.synth_agent_url,
            httpx_client=httpx_client
        )

        # Initialize the connection pool using 'async with'
        # This automatically handles pool.open() and pool.close()
        async with AsyncConnectionPool(
            s.postgres_connection_string,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row}
        ) as pool:

            checkpointer = AsyncPostgresSaver(pool)
            store = AsyncPostgresStore(pool)

            await checkpointer.setup()
            await store.setup()

            dependencies = GraphContext(
                logger=logger,
                llm=get_llm(s.hf_token)
            )

            app.state.chat_graph = create_chat_graph(
                checkpointer=checkpointer, store=store)
            app.state.checkpointer = checkpointer
            app.state.store = store
            app.state.dependencies = dependencies

            logger.info(
                "Mindfulness API is ready. Database checkpointer and store initialized.")

            yield

    logger.info("Application shutting down. Database connection pool closed.")


# Initialize FastAPI with the lifespan
app = FastAPI(title="Pulse Lotus - Requester Agent", lifespan=lifespan)


# --- Schemas ---
class ChatRequest(BaseModel):
    user_id: str
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    synth_triggered: bool


# --- Routes ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, bg_tasks: BackgroundTasks):
    """
    Handles user messages, advances the fast chat graph, and evaluates 
    the patience loop to trigger the heavy-compute synthesis graph.
    """
    graph = app.state.chat_graph
    a2a_client = app.state.a2a_client

    # LangGraph config requires a thread_id to persist state across turns
    config = {"configurable": {"thread_id": request.thread_id}}

    try:
        # Advance the graph state with the new user message
        state = await graph.ainvoke({"messages": [("user", request.message)]}, config)

        # Check the patience loop threshold logic defined in your graph
        trigger_synth = state.get("trigger_synth", False)

        if trigger_synth:
            # Trigger Agent B asynchronously.
            # We pass the thread_id so Agent B can potentially read
            # the exact same checkpoint state from Postgres if needed.
            bg_tasks.add_task(
                a2a_client.request_task,
                agent_id="pulse-synth-v1",
                skill="generate_meditation",
                input_data={
                    "context": state.get("summary", ""),
                    "thread_id": request.thread_id,
                    "user_id": request.user_id
                }
            )

        # Extract the AI's latest reply (assuming standard LangGraph message list)
        last_message = state["messages"][-1].content

        return ChatResponse(
            reply=last_message,
            synth_triggered=trigger_synth
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent_a_chat"}
