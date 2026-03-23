from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, TaskState, Part
from a2a.utils import new_task, new_agent_text_message

from agent_b_synth.state import GraphContext
from agent_b_synth.graph import get_heavy_llm, build_synth_graph

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PulseSynthExecutor(AgentExecutor):
    def __init__(self):
        super().__init__()
        self.dependencies = GraphContext(
            llm=get_heavy_llm(),
            logger=logger
        )
        self.synth_graph = build_synth_graph()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self.dependencies.logger.info(f"THE REQUEST CONTEXT: {context}")

        query = context.get_user_input()
        task = context.current_task or new_task(context.message)

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

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Required by the interface to handle client-side cancellation."""
        task_updater = TaskUpdater(
            event_queue, context.task_id, context.context_id)
        task_updater.update_status(state=TaskState.CANCELED)
