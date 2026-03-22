# agent_b_synth/graph.py
import os
import re
import logging
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# Import your heavy reflection nodes
from agent_b_synth.nodes.node_generate_answer import node_generate_answer
from agent_b_synth.nodes.node_generate_transcript import node_generate_transcript
from agent_b_synth.nodes.node_supervisor_agent import node_reflection

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TextPart, DataPart, TaskState, Part
from a2a.utils import new_task, new_agent_text_message

from agent_b_synth.state import SynthState, GraphContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_heavy_llm() -> ChatHuggingFace:
    """Create the heavy reflection model for Agent B."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"  # Or Qwen for synthesis
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=2048,  # Larger context for deep reflection
        temperature=0.4,     # Lower temp for more analytical reflection
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm)


def build_synth_graph():
    """
    Builds the asynchronous 3-turn reflection graph.
    Runs until the reflection supervisor validates both components.
    """
    def router(state: SynthState):
        # If both components passed reflection, terminate the loop
        if state.get("is_answer_valid") and state.get("is_transcript_valid"):
            return END

        # Dynamic Fan-Out based on what failed reflection
        tasks = []
        if not state.get("is_answer_valid"):
            tasks.append(Send("generate_answer", state))

        if not state.get("is_transcript_valid"):
            tasks.append(Send("generate_transcript", state))

        return tasks

    workflow = StateGraph(SynthState)

    workflow.add_node("generate_transcript", node_generate_transcript)
    workflow.add_node("generate_answer", node_generate_answer)
    workflow.add_node("reflection", node_reflection)

    # Entry point relies on the dynamic router immediately
    # Assuming initial state has valid flags as False
    workflow.set_conditional_entry_point(
        router,
        ["generate_answer", "generate_transcript", END]
    )

    workflow.add_edge(["generate_transcript", "generate_answer"], "reflection")

    workflow.add_conditional_edges(
        "reflection",
        router,
        ["generate_answer", "generate_transcript", END]
    )

    # Compiling without a checkpointer here if state is strictly ephemeral per A2A task
    # (If you want to resume failed synthesis, you can attach postgres here as well)
    return workflow.compile()


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
