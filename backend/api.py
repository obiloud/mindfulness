from dotenv import load_dotenv
from typing import Dict, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from workflow import build_mindfulness_graph, get_llm
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from contextlib import asynccontextmanager
from settings import get_settings
from state import GraphContext
from uuid import uuid4
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv(override=True)

s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize the connection pool using 'async with'
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
            llm=get_llm()
        )

        app.state.graph = build_mindfulness_graph(
            checkpointer=checkpointer, store=store)
        app.state.checkpointer = checkpointer
        app.state.store = store
        app.state.dependencies = dependencies

        logger.info(
            "Mindfulness API is ready. Database checkpointer and store initialized.")

        yield

    logger.info("Application shutting down. Database connection pool closed.")


class SessionRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    message: str
    transcript: Optional[str]


app = FastAPI(title="Mindfulness AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/mindfulness/session", response_model=SessionResponse)
async def create_session(body: SessionRequest, request: Request) -> SessionResponse:
    graph = request.app.state.graph
    dependecies = request.app.state.dependencies

    session_id = str(uuid4()) if body.session_id is None else body.session_id

    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = await graph.ainvoke({"messages": [HumanMessage(content=body.query)]}, config=config, context=dependecies)
        ai_message = final_state["messages"][-1]

        if final_state.get("safety_flag") == "unsafe":
            refusal = final_state.get("refusal_message") or ai_message.content
            return {
                "session_id": session_id,
                "message": refusal,
                "transcript": None,
            }

        transcript = final_state.get("transcript")

        return {
            "session_id": session_id,
            "message": ai_message.content,
            "transcript": transcript,
        }
    except Exception as e:
        logger.exception(f"Exception while invoking the graph: {e}")
        return {
            "session_id": session_id,
            "message": str(e),
            "transcript": None,
        }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
