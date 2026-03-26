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

import logging
import asyncio
import httpx

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
            logger=logger
        )
        self.synth_graph = build_synth_graph()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.dependencies.logger.info(f"THE REQUEST CONTEXT: {context}")

        query = context.get_user_input()
        data = get_data_parts(context.message.parts)[-1]
        thread_id = data.get("thread_id", "")

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
            final_state = await self.synth_graph.ainvoke({"context": query}, context=self.dependencies)

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
