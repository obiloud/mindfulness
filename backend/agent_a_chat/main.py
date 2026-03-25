# backend/agent_a_chat/main.py
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import logging
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from uuid import uuid4
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer

# LangGraph & Checkpointing
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore, PostgresIndexConfig
from langgraph.store.postgres.base import ANNIndexConfig
from fastembed import TextEmbedding
from .graph import get_llm, create_chat_graph

# A2A SDK
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    MessageSendConfiguration,
    PushNotificationConfig,
    Message, Role, Part, TextPart, DataPart
)
from a2a.utils import get_data_parts
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

from shared.settings import get_settings
from agent_a_chat.state import GraphContext, print_state

# === AUTHENTICATION ===
SECRET_KEY = "your-super-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# === JWT Helper Functions ===


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# === User Model (for DB) ===


class User(BaseModel):
    email: str
    password: str

# === Auth Schemas ===


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None

# === Auth Routes ===
# We'll add these to the app later


# === Database Setup ===
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
                    final_agent_card_to_use = _extended_card
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

        # Initialize the connection pool
        async with AsyncConnectionPool(
            s.postgres_connection_string,
            max_size=10,
            kwargs={"autocommit": True, "row_factory": dict_row}
        ) as pool:

            checkpointer = AsyncPostgresSaver(pool)
            cache_path = os.getenv("FASTEMBED_CACHE_PATH", "./local_cache")

            embedding_model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cache_dir=cache_path
            )

            def bge_embed_wrapper(input_data):
                # Ensure input is a list (FastEmbed requirement)
                if isinstance(input_data, str):
                    input_data = [input_data]

                # Generate embeddings (returns a generator of ndarrays)
                embeddings_generator = embedding_model.embed(input_data)

                # Convert ndarrays to lists (Postgres requirement)
                # and return the results
                return [e.tolist() for e in embeddings_generator]

            store = AsyncPostgresStore(
                pool,
                index=PostgresIndexConfig(
                    dims=384,
                    embed=bge_embed_wrapper,
                    fields=["content"],
                    ann_index_config=ANNIndexConfig(kind="hnsw"),
                    distance_type="cosine"
                )
            )

            await checkpointer.setup()
            await store.setup()

            context = GraphContext(
                logger=logger,
                llm=get_llm(s.hf_token),
                user_id=None
            )

            app.state.db_pool = pool
            app.state.chat_graph = create_chat_graph(
                checkpointer=checkpointer, store=store)
            app.state.checkpointer = checkpointer
            app.state.store = store
            app.state.context = context

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
    answer: Optional[str] = None
    transcript: Optional[str] = None


class SynthesisResult(BaseModel):
    thread_id: str
    answer: str
    transcript: str
    status: str = "completed"

# === Message Saving Helper ===


async def save_message(thread_id: str, user_id: str, role: str, content: str):
    if not app.state or not app.state.db_pool:
        logger.warning("Database pool not available. Cannot save message.")
        return
    async with app.state.db_pool.connection() as conn:
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

# === Auth Routes ===


@app.post("/auth/register")
async def register(user: User):
    """
    Register a new user.
    """
    try:
        # Check if user already exists
        async with app.state.db_pool.connection() as conn:
            result = await conn.execute(
                "SELECT id FROM users WHERE email = %s",
                (user.email,)
            )
            if await result.fetchone():
                raise HTTPException(
                    status_code=400, detail="Email already registered")

            # Hash password
            hashed_password = get_password_hash(user.password)

            # Insert user
            new_user_id = str(uuid4())

            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO users (id, email, password_hash)
                    VALUES (%s, %s, %s)
                    """,
                    (new_user_id, user.email, hashed_password)
                )

            # Create token
            token = create_access_token(data={"sub": new_user_id})
            return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=str(e))
    finally:
        pass


@app.post("/auth/login")
async def login(user: User):
    """
    Login user and return JWT token.
    """
    async with app.state.db_pool.connection() as conn:
        result = await conn.execute(
            "SELECT id, email, password_hash FROM users WHERE email = %s",
            (user.email,)
        )
        row = await result.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail="Invalid credentials")

        # Verify password
        if not verify_password(user.password, row["password_hash"]):
            raise HTTPException(status_code=400, detail="Invalid credentials")

        # Create token
        token = create_access_token(data={"sub": str(row["id"])})
        return {"access_token": token, "token_type": "bearer"}


@app.get("/users/me")
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Get current authenticated user.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# === Message History Endpoint ===


@app.get("/v1/mindfulness/messages/history/{thread_id}")
async def get_message_history(thread_id: str, user_id: str = None):
    """
    Retrieve conversation history for a thread.
    """
    async with app.state.db_pool.connection() as conn:
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

# === Routes (unchanged except for user_id handling) ===


@app.post("/v1/mindfulness/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest, request: Request, bg_tasks: BackgroundTasks, current_user: TokenData = Depends(get_current_user)):
    """
    Handles user messages, advances the fast chat graph, and evaluates
    the patience loop to trigger the heavy-compute synthesis graph.
    """
    graph = request.app.state.chat_graph
    a2a_client = app.state.a2a_client

    thread_id = str(uuid4()) if body.thread_id is None else body.thread_id
    user_id = current_user.get("user_id")

    # Validate user_id (in real app, verify user exists)
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    context = request.app.state.context
    context.user_id = user_id

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Advance the graph state with the new user message
        state = await graph.ainvoke({"messages": [HumanMessage(content=body.message)]}, config=config, context=context)

        logger.info(f"final_state: {print_state(state)}")

        synth_status = state.get("synth_status")

        if synth_status == "requested":
            summary = state.get("summary", "")
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
                            Part(root=DataPart(
                                data={"thread_id": thread_id, "user_id": user_id}))
                        ])
                ))
            )
            await graph.aupdate_state(config, {"synth_status": "in_progress"})

        # Extract the AI's latest reply
        last_message = state["messages"][-1]
        reply = last_message.content
        answer = last_message.additional_kwargs.get("answer")
        transcript = last_message.additional_kwargs.get("transcript")

        # Save user and AI messages
        await save_message(thread_id, user_id, "user", body.message)
        await save_message(thread_id, user_id, "ai", reply)

        return ChatResponse(
            reply=reply,
            user_id=user_id,
            thread_id=thread_id,
            answer=answer,
            transcript=transcript
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Internal Callback Endpoint ---
@app.post("/internal/v1/synthesis-callback")
async def handle_synthesis_complete(result: SynthesisResult, request: Request):
    """
    Receives the finished transcript from the synthesis agent and updates the session state.
    Also triggers a proactive message to the user via WebSocket.
    """
    chat_graph = request.app.state.chat_graph
    thread_id = result.thread_id
    answer = result.answer
    transcript = result.transcript
    status = result.status

    if status != "completed":
        logger.error(f"Synthesis failed for thread {thread_id}")
        return {"status": "error"}

    config = {"configurable": {"thread_id": thread_id}}

    await chat_graph.aupdate_state(
        config,
        {
            "answer": answer,
            "transcript": transcript,
            "synth_status": "completed",
            "is_synthesis_ready": True
        },
    )

    logger.info(
        f"Received transcript for thread {thread_id}: {transcript[:100]}...")

    return {"status": "acknowledged"}


# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent_a_chat"}
