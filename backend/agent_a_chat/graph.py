# agent_a_chat/graph.py
from typing import Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END
from langchain.messages import AIMessage

from .nodes.node_user_input_agent import node_user_input
from .nodes.node_conversation_agent import node_conversation
from .nodes.node_proactive_engagement import node_proactive_engagement
from .nodes.node_hydrate import node_hydrate
from .nodes.node_manage_memory import node_manage_memory
from .nodes.node_summary import summarization_node
from .state import ChatState, GraphContext


def get_llm(hf_token: str) -> ChatHuggingFace:
    """Create the fast chat model for Agent A."""
    repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,  # Shorter context for fast replies
        temperature=0.7,
        huggingfacehub_api_token=hf_token,
    )
    return ChatHuggingFace(llm=llm)


def should_trigger_synth(state: ChatState) -> ChatState:
    """Evaluates the patience loop threshold."""
    status = state.get("synth_status", "idle")

    if status == "idle" and (state.get("info_score", 0) >= 0.8 or state.get("turn_count", 0) > 5):
        return {"synth_status": "requested"}

    return {}


def node_deliver_transcript(state: ChatState) -> ChatState:
    return {
        "messages": [
            AIMessage(
                content="Excellent. Let's begin.",
                additional_kwargs={
                    "answer": state["answer"], "transcript": state["transcript"]}
            )
        ],
        "awaiting_confirmation": False,
        "is_synthesis_ready": False  # Reset the flag
    }


def create_chat_graph(checkpointer: BaseCheckpointSaver = None, store: BaseStore = None):
    """
    Builds the fast Chat Graph.
    Evaluates context and flags when to trigger the Synthesis Agent.
    """

    workflow = StateGraph(ChatState, context_schema=GraphContext)

    # We pass llm to nodes via partials or adjust your node implementations to accept it
    workflow.add_node("user", node_user_input)
    workflow.add_node("hydrate", node_hydrate)
    workflow.add_node("conversation", node_conversation)
    workflow.add_node("manage_memory", node_manage_memory)
    workflow.add_node("evaluate_patience", should_trigger_synth)
    workflow.add_node("proactive_engagement", node_proactive_engagement)
    workflow.add_node("deliver_transcript", node_deliver_transcript)

    workflow.set_entry_point("user")

    def route_after_user(state: ChatState) -> Literal["safe_to_proceed", "deliver_transcript", "end"]:
        if state.get("safety_flag") == "unsafe":
            return "end"
        # If we were waiting for a confirmation, check if they said "Yes"
        if state.get("awaiting_confirmation"):
            user_input = state["messages"][-1].content.lower()
            if any(word in user_input for word in ["yes", "start", "sure", "let's do it"]):
                return "deliver_transcript"

        return "safe_to_proceed"

    workflow.add_conditional_edges("user", route_after_user, {
        "deliver_transcript": "deliver_transcript",
        "safe_to_proceed": "hydrate",
        "end": END
    })

    workflow.add_edge("hydrate", "conversation")
    workflow.add_edge("conversation", "manage_memory")

    def route_after_conversation(state: ChatState) -> Literal["proactive_engagement", "evaluate_patience"]:
        if state.get("is_synthesis_ready") and not state.get("awaiting_confirmation"):
            return "proactive_engagement"
        return "evaluate_patience"

    workflow.add_conditional_edges(
        "manage_memory",
        route_after_conversation,
        {
            "proactive_engagement": "proactive_engagement",
            "evaluate_patience": "evaluate_patience"
        })

    workflow.add_edge("deliver_transcript", "evaluate_patience")
    workflow.add_edge("proactive_engagement", "evaluate_patience")
    workflow.add_edge("evaluate_patience", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
