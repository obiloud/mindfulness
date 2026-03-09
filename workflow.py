import os
from typing import List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, messages_to_dict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END

import logging
import json

from state import ConversationState, GraphContext
from agents.user_input_agent import node_user_input
from agents.conversation_agent import node_clarification
from agents.meditation_guide_agent import node_assistant
from agents.supervisor_agent import node_reflection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meditation_graph")

load_dotenv(override=True)


def print_state(state: ConversationState, full: bool = False) -> str:
    history = []
    if full:
        history = messages_to_dict(state["history"])

    if len(state["messages"]):
        print_data = {**state, "history": history,
                      "messages": messages_to_dict([state["messages"][-1]])}
    else:
        print_data = {**state, "history": history}

    return json.dumps(print_data, indent=2)


def _get_llm() -> ChatHuggingFace:
    """Create the base chat model used by the graph."""
    hf_token = os.getenv("HF_TOKEN")
    # repo_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    # llm = HuggingFaceEndpoint(
    #     repo_id=repo_id,
    #     task="text-generation",
    #     max_new_tokens=1024,
    #     temperature=0.5,
    #     top_k=80,
    #     top_p=0.8,
    #     repetition_penalty=1.1,
    #     huggingfacehub_api_token=hf_token,
    #     provider="auto",
    # )

    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

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


def build_mindfulness_graph():
    """
    Build a LangGraph graph that can:
    - hold a short conversation about the user's context
    - optionally call tools (e.g. guided meditation audio session)
    - reflect on its answer once and improve it.
    """

    def needs_clarification(state: ConversationState) -> bool:
        """
        Heuristic to decide if we should ask for more information.

        For now we look at very short or extremely vague queries and limit
        the number of clarification turns to 3.
        """
        count = state.get("clarification_count", 0)
        if count >= 3:
            return False

        query = (state.get("query") or "").strip().lower()
        if len(query) < 10:
            return True

        vague_tokens = [
            "stress",
            "stressed",
            "anxious",
            "anxiety",
            "bad",
            "not good",
            "overwhelmed",
        ]
        # Ask for clarification when the user gives only a very generic label
        # without describing context.
        return any(token == query or (token in query and len(query.split()) <= 6) for token in vague_tokens)

    def should_proceed(state: ConversationState) -> Literal["clarify", "answer", "end"]:
        if state.get("status") == "done":
            return "end"

        if needs_clarification(state):
            return "clarify"

        return "answer"

    def should_refine(state: ConversationState) -> Literal["refine", "end"]:
        if state.get("status") == "done":
            return "end"

        return "refine"

    graph = StateGraph(ConversationState)
    graph.add_node("user", node_user_input)
    graph.add_node("clarification", node_clarification)
    graph.add_node("assistant", node_assistant)
    graph.add_node("reflection", node_reflection)

    graph.set_entry_point("user")
    graph.add_conditional_edges(
        "user",
        should_proceed,
        {
            "clarify": "clarification",
            "answer": "assistant",
            "end": END
        },
    )
    graph.add_edge("assistant", "reflection")
    graph.add_conditional_edges(
        "reflection",
        should_refine,
        {
            "refine": "assistant",
            "end": END,
        },
    )

    return graph.compile()


def run_mindfulness_graph(query: str, history: Optional[List[AnyMessage]] = None):
    """
    Convenience function for external callers (FastAPI, CLI, etc.).
    Returns the last assistant message content and any generated metadata.
    """
    app = build_mindfulness_graph()

    dependencies = GraphContext(
        logger=logger,
        llm=_get_llm()
    )

    initial_state: ConversationState = {
        "query": query,
        "history": history or [],
        "messages": [],
        "transcript": None,
        "safety_flag": None,
        "refusal_message": None,
        "status": "initial",
        "clarification_count": 0,
        "reflection_count": 0,
    }

    final_state = app.invoke(initial_state, context=dependencies)
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
        f"Graph completed. Final status: {print_state(final_state, full=True)}")

    return {
        "message": content,
        "transcript": transcript,
    }


if __name__ == "__main__":
    graph = build_mindfulness_graph()

    with open("workflow_mermaid.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
