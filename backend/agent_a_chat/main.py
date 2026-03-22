# agent_a_chat/main.py
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any
from pydantic import BaseModel
import logging
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from uuid import uuid4

# LangGraph & Checkpointing
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from agent_a_chat.graph import get_llm, create_chat_graph

# A2A SDK
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    Role,
    Part,
    TextPart,
    SendMessageSuccessResponse,
    Task,
    Message
)
from a2a.utils import get_data_parts
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

from shared.settings import get_settings
from agent_a_chat.state import GraphContext, print_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=60.0) as httpx_client:

        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=s.synth_agent_url,
        )

        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {s.synth_agent_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = await resolver.get_agent_card()

            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        f'\nPublic card supports authenticated extended card. Attempting to fetch from: {s.synth_agent_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
                        exc_info=True,
                    )
            elif _public_card:  # supports_authenticated_extended_card is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        app.state.a2a_client = A2AClient(
            httpx_client=httpx_client,
            agent_card=final_agent_card_to_use,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    user_id: str
    synth_ready: bool


class TaskStatusResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    transcript: Optional[str] = None

# --- Routes ---


@app.post("/v1/mindfulness/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request, bg_tasks: BackgroundTasks):
    """
    Handles user messages, advances the fast chat graph, and evaluates 
    the patience loop to trigger the heavy-compute synthesis graph.
    """
    graph = request.app.state.chat_graph
    a2a_client = app.state.a2a_client

    thread_id = str(
        uuid4()) if body.thread_id is None else body.thread_id

    user_id = str(
        uuid4()) if body.user_id is None else body.user_id

    # LangGraph config requires a thread_id to persist state across turns
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Advance the graph state with the new user message
        state = await graph.ainvoke({"messages": [HumanMessage(content=body.message)]}, config=config, context=request.app.state.dependencies)

        logger.info(f"final_state: {print_state(state)}")

        trigger_synth = state.get("trigger_synth", False)

        # Extract the AI's latest reply (assuming standard LangGraph message list)
        last_message = state["messages"][-1].content

        return ChatResponse(
            reply=last_message,
            user_id=user_id,
            thread_id=thread_id,
            synth_ready=trigger_synth,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/mindfulness/synthesis/{thread_id}", response_model=TaskStatusResponse)
async def trigger_synthesis(thread_id: str, request: Request):
    graph = request.app.state.chat_graph
    a2a_client = request.app.state.a2a_client
    config = {"configurable": {"thread_id": thread_id}}

    # Retrieve the latest state snapshot from LangGraph
    state_snapshot = await graph.aget_state(config)

    # Extract the summary (ensure your State schema defines this field)
    summary = state_snapshot.values.get("summary", "")

    if not summary:
        raise HTTPException(
            status_code=400,
            detail="No summary found for this thread. Agent A may not have finished processing."
        )

    try:
        response = await a2a_client.send_message(
            SendMessageRequest(id=str(uuid4()), params=MessageSendParams(
                message=Message(message_id=str(uuid4()), role=Role(
                    'user'), parts=[Part(root=TextPart(text=summary))])
            ))
        )

        if isinstance(response.root, SendMessageSuccessResponse):
            if isinstance(response.root.result, Task):
                data = get_data_parts(
                    response.root.result.artifacts[-1].parts)[-1]
            else:
                data = get_data_parts(response.root.result.parts)[-1]

            return {
                "answer": data.get("answer", ""),
                "transcript": data.get("transcript", ""),
                "status": "success"
            }
        else:
            raise HTTPException(
                status_code=500, detail=response.root.message)

    except Exception as e:
        logger.error(f"A2A Synthesis failed: {e}")
        raise HTTPException(
            status_code=500, detail="Synthesis agent failed to respond.")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent_a_chat"}
