# agent_a_chat/graph.py
import os
from typing import Literal, TypedDict, List, Annotated
import operator
from langchain_core.messages import AnyMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END

# Import your fast conversation nodes
from nodes.node_user_input_agent import node_user_input
from nodes.node_conversation_agent import node_conversation


class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    status: str
    turn_count: int
    info_score: float
    summary: str
    trigger_synth: bool  # New flag for A2A handoff


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


def create_chat_graph(dependencies: dict):
    """
    Builds the fast Chat Graph.
    Evaluates context and flags when to trigger the Synthesis Agent.
    """
    hf_token = dependencies.get("hf_token")
    llm = get_llm(hf_token)

    def safe_to_proceed(state: ChatState) -> Literal["yes", "no"]:
        if state.get("status") == "done":
            return "no"
        return "yes"

    def should_trigger_synth(state: ChatState) -> ChatState:
        """Evaluates the patience loop threshold."""
        if state.get("info_score", 0) >= 0.8 or state.get("turn_count", 0) > 5:
            return {"trigger_synth": True}
        return {"trigger_synth": False}

    workflow = StateGraph(ChatState)

    # We pass llm to nodes via partials or adjust your node implementations to accept it
    workflow.add_node("user", node_user_input)
    workflow.add_node("conversation", node_conversation)
    workflow.add_node("evaluate_patience", should_trigger_synth)

    workflow.set_entry_point("user")

    workflow.add_conditional_edges("user", safe_to_proceed, {
        "yes": "conversation",
        "no": END
    })

    # After conversation, we always evaluate if we hit the threshold
    workflow.add_edge("conversation", "evaluate_patience")
    workflow.add_edge("evaluate_patience", END)

    return workflow
