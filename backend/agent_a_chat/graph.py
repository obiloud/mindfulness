# agent_a_chat/graph.py
from typing import Literal
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END

from agent_a_chat.nodes.node_user_input_agent import node_user_input
from agent_a_chat.nodes.node_conversation_agent import node_conversation
from agent_a_chat.state import ChatState


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


def create_chat_graph(checkpointer: BaseCheckpointSaver = None, store: BaseStore = None):
    """
    Builds the fast Chat Graph.
    Evaluates context and flags when to trigger the Synthesis Agent.
    """

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

    return workflow.compile(checkpointer=checkpointer, store=store)
