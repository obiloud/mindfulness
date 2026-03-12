import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

import logging
import json

from state import ConversationState, GraphContext
from agents.user_input_agent import node_user_input
from agents.conversation_agent import node_conversation
from agents.meditation_guide_agent import node_assistant
from agents.supervisor_agent import node_reflection
from settings import get_settings

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
    """Create the base chat model used by the graph."""
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
    Build a LangGraph graph that can:
    - hold a short conversation about the user's context
    - optionally call tools (e.g. guided meditation audio session)
    - reflect on its answer once and improve it.
    """
    def safe_to_proceed(state: ConversationState) -> Literal["conversation", "end"]:
        if state.get("status") == "done":
            return "end"

        return "conversation"

    def should_answer(state: ConversationState) -> Literal["proceed_to_draft", "ask_again"]:
        if state.get("info_score", 0) >= 0.8 or state["turn_count"] > 5:
            return "proceed_to_draft"
        return "ask_again"

    def should_refine(state: ConversationState) -> Literal["refine", "end"]:
        if state.get("status") == "done":
            return "end"

        return "refine"

    graph = StateGraph(ConversationState)
    graph.add_node("user", node_user_input)
    graph.add_node("conversation", node_conversation)
    graph.add_node("assistant", node_assistant)
    graph.add_node("reflection", node_reflection)

    graph.set_entry_point("user")
    graph.add_conditional_edges("user", safe_to_proceed, {
        "conversation": "conversation",
        "end": END
    })
    graph.add_conditional_edges("conversation", should_answer, {
        "ask_again": END,
        "proceed_to_draft": "assistant"
    })
    graph.add_edge("assistant", "reflection")
    graph.add_conditional_edges(
        "reflection",
        should_refine,
        {
            "refine": "assistant",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer, store=store)


def run_mindfulness_graph(query: str):
    """
    Convenience function for external callers (FastAPI, CLI, etc.).
    Returns the last assistant message content and any generated metadata.
    """
    app = build_mindfulness_graph()

    dependencies = GraphContext(
        logger=logger,
        llm=get_llm()
    )

    initial_state: ConversationState = {
        "query": query,
        "messages": [],
        "transcript": None,
        "safety_flag": None,
        "refusal_message": None,
        "status": "initial",
        "info_score": 0,
        "turn_count": 0,
        "reflection_count": 0,
    }

    final_state = app.invoke(initial_state, context=dependencies, config={
                             "configurable": {"thread_id": "1"}})
    messages = final_state["messages"]
    last_ai = next((m for m in reversed(messages)
                   if getattr(m, "type", None) == "ai"), None)
    content = last_ai.content if last_ai is not None else ""

    # Safety refusal: never return a transcript.
    if final_state.get("safety_flag") == "unsafe":
        refusal = final_state.get("refusal_message") or content
        return {
            "message": refusal,
            "transcript": None,
        }

    transcript = final_state.get("transcript")

    logger.info(
        f"Graph completed. Final status: {print_state(final_state)}")

    return {
        "message": content,
        "transcript": transcript,
    }


if __name__ == "__main__":
    graph = build_mindfulness_graph()

    with open("workflow_mermaid.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
