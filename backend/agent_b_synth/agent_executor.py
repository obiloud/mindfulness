from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, TaskState, Part
from a2a.utils import new_task, new_agent_text_message, get_data_parts

from agent_b_synth.state import GraphContext
from agent_b_synth.graph import build_synth_graph

from shared.settings import get_settings
from shared.model_factory import get_heavy_hf_llm, get_heavy_ollama_llm
from shared.datamodels.preferences import VoiceBlueprint, MindfulnessProfile
import logging
import asyncio
import httpx

import os
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.store.postgres.aio import AsyncPostgresStore, PostgresIndexConfig
from langgraph.store.postgres.base import ANNIndexConfig
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PulseSynthExecutor(AgentExecutor):
    def __init__(self):
        super().__init__()

        llm = None
        if get_settings().inference_provider == "ollama":
            llm = get_heavy_ollama_llm()
        elif get_settings().inference_provider == "huggingface":
            llm = get_heavy_hf_llm()

        self.dependencies = GraphContext(
            llm=llm,
            logger=logger,
        )
        self.synth_graph = build_synth_graph()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.dependencies.logger.info(f"THE REQUEST CONTEXT: {context}")

        query = context.get_user_input()
        data = get_data_parts(context.message.parts)[-1]
        thread_id = data.get("thread_id", "")
        user_id = data.get("user_id", "")

        task = context.current_task or new_task(context.message)
        callback_url = None

        if context.configuration is not None:
            if context.configuration.push_notification_config is not None:
                callback_url = context.configuration.push_notification_config.url

        # Ensure the task is known to the queue
        if not context.current_task:
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:

            # Initialize the connection pool
            async with AsyncConnectionPool(
                get_settings().postgres_connection_string,
                max_size=10,
                kwargs={"autocommit": True, "row_factory": dict_row}
            ) as pool:

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

                namespace = ("preferences", user_id)
                voice_blueprint_item = await store.aget(namespace, "voice_blueprint")
                existing_blueprint = VoiceBlueprint(
                    **voice_blueprint_item.value) if voice_blueprint_item else VoiceBlueprint()

                logger.info(f"VOICE BLUEPRINT: {existing_blueprint}")

                minfulness_profile_item = await store.aget(namespace, "mindfulness_profile")
                existing_profile = MindfulnessProfile(
                    **minfulness_profile_item.value) if minfulness_profile_item else MindfulnessProfile()

                logger.info(f"MINDFULNESS PROFILE: {existing_profile}")

            config = {"configurable": {
                "voice_blueprint": existing_blueprint, "mindfulness_profile": existing_profile}}
            final_state = await self.synth_graph.ainvoke({"context": query}, context=self.dependencies, config=config)

            answer = final_state.get("answer", "No answer generated.")
            transcript = final_state.get(
                "transcript", "No transcript generated.")

            if final_state.get("is_answer_valid") and final_state.get("is_transcript_valid"):
                # Use add_artifact for the long transcript and complete the task
                await task_updater.add_artifact(
                    parts=[
                        Part(root=DataPart(data={
                            "answer": answer,
                            "transcript": transcript,
                        })),

                    ],
                    name='meditation_synthesis',
                )
                # A2A V3 uses .update_status for completion usually, or .complete()
                await task_updater.update_status(TaskState.completed)

                if callback_url:
                    self.dependencies.logger.info(
                        f"Firing callback to {callback_url} for thread {thread_id}")
                    # Use create_task to fire-and-forget, preventing the executor from hanging
                    asyncio.create_task(self._send_webhook(
                        callback_url, thread_id,  answer, transcript))
            else:
                await task_updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(
                        "Reflection failed to validate the output.")
                )

        except Exception as e:
            logger.error(f"Execution Error: {e}", exc_info=True)

            await task_updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Synthesis failed: {str(e)}")
            )

    async def _send_webhook(self, url: str, thread_id: str, answer: str, transcript: str):
        """Helper method to push the result back to the orchestrator."""
        payload = {
            "thread_id": thread_id,
            "answer": answer,
            "transcript": transcript,
            "status": "completed"
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                self.dependencies.logger.info(
                    f"Callback successfully delivered: HTTP {response.status_code}")
        except Exception as e:
            self.dependencies.logger.error(
                f"Failed to send callback to {url}: {e}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Required by the interface to handle client-side cancellation."""
        task_updater = TaskUpdater(
            event_queue, context.task_id, context.context_id)
        task_updater.update_status(state=TaskState.CANCELED)
