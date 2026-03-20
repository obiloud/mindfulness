import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.types import Send

import logging
import json

from state import ConversationState
from agents.node_user_input_agent import node_user_input
from agents.node_conversation_agent import node_conversation
from agents.node_generate_answer import node_generate_answer
from agents.node_generate_transcript import node_generate_transcript
from agents.node_supervisor_agent import node_reflection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def print_state(state: ConversationState) -> str:
    if len(state["messages"]):
        print_data = {
            **state, "messages": messages_to_dict(state["messages"])}
    else:
        print_data = state

    return json.dumps(print_data, indent=2)


def get_llm() -> ChatHuggingFace:
    """Create the base chat model used by the workflow."""
    hf_token = os.getenv("HF_TOKEN")

    # repo_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    repo_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.7,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.1,
        huggingfacehub_api_token=hf_token,
        provider="auto"
    )

    return ChatHuggingFace(llm=llm)


def build_mindfulness_graph(checkpointer: BaseCheckpointSaver = None, store: BaseStore = None):
    """
    Build a LangGraph workflow that can:
    - hold a short conversation about the user's context
    - optionally call tools (e.g. guided meditation audio session)
    - reflect on its answer once and improve it.
    """
    def safe_to_proceed(state: ConversationState) -> Literal["yes", "no"]:
        if state.get("status") == "done":
            return "no"

        return "yes"

    def should_answer(state: ConversationState) -> Literal["proceed_to_draft", "ask_again"]:
        logger.info(
            f"Should answer: info_score {state['info_score']} turns {state['turn_count']}")
        if state.get("info_score", 0) >= 0.8 or state["turn_count"] > 5:
            return "proceed_to_draft"
        return "ask_again"

    def node_patience(state: ConversationState) -> ConversationState:
        return state

    def router(state: ConversationState):
        # If both components passed reflection, terminate the loop
        if state.get("is_answer_valid") and state.get("is_transcript_valid"):
            return END

        # Dynamic Fan-Out
        tasks = []

        if not state.get("is_answer_valid"):
            tasks.append(Send("generate_answer", state))

        if not state.get("is_transcript_valid"):
            tasks.append(Send("generate_transcript", state))

        return tasks

    workflow = StateGraph(ConversationState)
    workflow.add_node("user", node_user_input)
    workflow.add_node("conversation", node_conversation)
    workflow.add_node("patience", node_patience)
    workflow.add_node("generate_transcript", node_generate_transcript)
    workflow.add_node("generate_answer", node_generate_answer)
    workflow.add_node("reflection", node_reflection)

    workflow.set_entry_point("user")
    workflow.add_conditional_edges("user", safe_to_proceed, {
        "yes": "conversation",
        "no": END
    })
    workflow.add_conditional_edges("conversation", should_answer, {
        "ask_again": END,
        "proceed_to_draft": "patience"
    })
    workflow.add_conditional_edges(
        "patience", router, ["generate_answer", "generate_transcript", END])
    workflow.add_edge(["generate_transcript", "generate_answer"], "reflection")
    workflow.add_conditional_edges(
        "reflection", router, ["generate_answer", "generate_transcript", END])

    return workflow.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    workflow = build_mindfulness_graph()

    with open("workflow_mermaid.png", "wb") as f:
        f.write(workflow.get_graph().draw_mermaid_png())
