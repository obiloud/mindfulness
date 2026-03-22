# agent_b_synth/graph.py
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.types import Send

# Import your heavy reflection nodes
from agent_b_synth.nodes.node_generate_answer import node_generate_answer
from agent_b_synth.nodes.node_generate_transcript import node_generate_transcript
from agent_b_synth.nodes.node_supervisor_agent import node_reflection

from agent_b_synth.state import SynthState


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
