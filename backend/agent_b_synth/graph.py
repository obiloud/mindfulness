# agent_b_synth/graph.py
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# Import your heavy reflection nodes
from .nodes.node_generate_answer import node_generate_answer
from .nodes.node_generate_transcript import node_generate_transcript
from .nodes.node_supervisor_agent import node_reflection
from .nodes.node_chapterize import node_chapterize

from .state import SynthState


def build_synth_graph():
    """
    Builds the asynchronous 3-turn reflection graph.
    Runs until the reflection supervisor validates both components.
    """
    def router(state: SynthState):
        # If both components passed reflection, terminate the loop
        if state.get("is_transcript_valid") and state.get("is_answer_valid"):
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
    workflow.add_node("chapterize", node_chapterize)

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

    # After reflection succeeds, add chapterization
    workflow.add_edge("reflection", "chapterize")

    # Compiling without a checkpointer here if state is strictly ephemeral per A2A task
    # (If you want to resume failed synthesis, you can attach postgres here as well)
    return workflow.compile()
