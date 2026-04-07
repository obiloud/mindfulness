# backend/agent_a_chat/main.py
"""
Main entry point for Agent A Chat API.
Handles app initialization, lifespan, and health checks.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.settings import get_settings
from agent_a_chat.graph import create_chat_graph
from shared.model_factory import get_fast_ollama_llm, get_fast_hf_llm
from agent_a_chat.state import GraphContext
from agent_a_chat.routes.authentication import router as auth_router
from agent_a_chat.routes.chat import router as chat_router
from agent_a_chat.routes.internal import router as internal_router
from a2a.client import A2AClient, A2ACardResolver
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH
from a2a.types import AgentCard
from httpx import AsyncClient
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore, PostgresIndexConfig
from langgraph.store.postgres.base import ANNIndexConfig
from fastembed import TextEmbedding
import os

# === Lifespan Context Manager ===


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state and dependencies."""
    async with AsyncClient(timeout=90.0) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=get_settings().synth_agent_url,
        )

        final_agent_card_to_use: AgentCard | None = None

        try:
            logger = logging.getLogger(__name__)
            logger.info(
                f'Attempting to fetch public agent card from: {get_settings().synth_agent_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = await resolver.get_agent_card()

            logger.info('Successfully fetched public agent card:')
            logger.info(_public_card.model_dump_json(
                indent=2, exclude_none=True))
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        f'\nPublic card supports authenticated extended card. Attempting to fetch from: {get_settings().synth_agent_url}{EXTENDED_AGENT_CARD_PATH}'
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
                            indent=2, exclude_none=True)
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
            elif _public_card:
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
            get_settings().postgres_connection_string,
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
                if isinstance(input_data, str):
                    input_data = [input_data]
                embeddings_generator = embedding_model.embed(input_data)
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

            llm = None
            if get_settings().inference_provider == "ollama":
                llm = get_fast_ollama_llm()
            elif get_settings().inference_provider == "huggingface":
                llm = get_fast_hf_llm()

            context = GraphContext(llm=llm)

            app.state.db_pool = pool
            app.state.chat_graph = create_chat_graph(
                checkpointer=checkpointer, store=store)
            app.state.checkpointer = checkpointer
            app.state.store = store
            app.state.context = context

            logger.info(
                "Mindfulness API is ready. Database checkpointer and store initialized."
            )

            yield

    logger.info("Application shutting down. Database connection pool closed.")


# === Initialize FastAPI ===
app = FastAPI(title="Pulse Lotus - Requester Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Health Check ===
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent_a_chat"}


# === Include Routers ===
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(chat_router, prefix="/v1/mindfulness", tags=["Chat"])
app.include_router(internal_router, prefix="/internal", tags=["Internal"])
