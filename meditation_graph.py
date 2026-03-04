import os
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END

from story_generator_pipeline import meditation_guide_generator_chain
from voice_generator_pipeline import voice_character_chain


load_dotenv(override=True)


class ConversationState(TypedDict):
    """Shared state for the mindfulness LangGraph agent."""

    query: str
    history: List[AnyMessage]
    messages: List[AnyMessage]
    transcript: Optional[str]
    voice_character: Optional[str]
    reflection: Optional[str]
    status: Literal["initial", "answering", "reflecting", "done"]


def _get_llm() -> ChatHuggingFace:
    """Create the base chat model used by the graph."""
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.5,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.1,
        huggingfacehub_api_token=hf_token,
        provider="auto",
    )

    return ChatHuggingFace(llm=llm)


def build_mindfulness_graph(tools: List[BaseTool]):
    """
    Build a LangGraph graph that can:
    - hold a short conversation about the user's context
    - optionally call tools (e.g. guided meditation audio session)
    - reflect on its answer once and improve it.
    """
    chat_model_with_tools = _get_llm().bind_tools(tools)

    def node_user_input(state: ConversationState) -> ConversationState:
        """Seed the conversation with the latest user query."""
        messages = list(state.get("history", []))
        messages.append(HumanMessage(content=state["query"]))
        return {
            **state,
            "messages": messages,
            "status": "answering",
        }

    def node_assistant(state: ConversationState) -> ConversationState:
        """Main assistant step that can decide to call tools."""
        messages = state["messages"]
        ai_msg = chat_model_with_tools.invoke(messages)
        messages = messages + [ai_msg]
        return {
            **state,
            "messages": messages,
        }

    def node_reflection(state: ConversationState) -> ConversationState:
        """
        Simple reflection step: have the model critique and, if needed,
        refine its previous answer into a clearer, more soothing response.
        """
        messages = state["messages"]
        last_ai = messages[-1]

        reflection_prompt = (
            "You are reflecting on your previous response as a mindfulness coach.\n"
            "1. Briefly critique your last answer for clarity, empathy, and usefulness.\n"
            "2. Then provide an improved, final response, keeping only what helps the user.\n"
            "Respond directly with the improved response, not the critique."
        )

        reflect_msg = HumanMessage(content=reflection_prompt)
        reflected = _get_llm().invoke(messages + [reflect_msg])

        messages = messages + [reflected]

        return {
            **state,
            "messages": messages,
            "reflection": "done",
            "status": "done",
        }

    def should_reflect(state: ConversationState) -> str:
        """
        Decide whether to go through a reflection pass.
        For now we reflect once, after the first assistant answer.
        """
        if state.get("reflection") == "done":
            return "end"
        return "reflect"

    graph = StateGraph(ConversationState)
    graph.add_node("user", node_user_input)
    graph.add_node("assistant", node_assistant)
    graph.add_node("reflection", node_reflection)

    graph.set_entry_point("user")
    graph.add_edge("user", "assistant")
    graph.add_conditional_edges(
        "assistant",
        should_reflect,
        {
            "reflect": "reflection",
            "end": END,
        },
    )
    graph.add_edge("reflection", END)

    return graph.compile()


def run_mindfulness_graph(query: str, history: Optional[List[AnyMessage]] = None, tools: Optional[List[BaseTool]] = None):
    """
    Convenience function for external callers (FastAPI, CLI, etc.).
    Returns the last assistant message content and any generated metadata.
    """
    tools = tools or []
    app = build_mindfulness_graph(tools)

    initial_state: ConversationState = {
        "query": query,
        "history": history or [],
        "messages": [],
        "transcript": None,
        "voice_character": None,
        "reflection": None,
        "status": "initial",
    }

    final_state = app.invoke(initial_state)
    messages = final_state["messages"]
    last_ai = next((m for m in reversed(messages) if m.type == "ai"), None)
    content = last_ai.content if last_ai is not None else ""

    return {
        "message": content,
        "transcript": final_state.get("transcript"),
        "voice_character": final_state.get("voice_character"),
    }

