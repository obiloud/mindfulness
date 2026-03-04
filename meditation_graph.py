import os
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END

import json
import re


load_dotenv(override=True)


class ConversationState(TypedDict):
    """Shared state for the mindfulness LangGraph agent."""

    query: str
    history: List[AnyMessage]
    messages: List[AnyMessage]
    transcript: Optional[str]
    # Safety / refusal
    safety_flag: Optional[str]
    refusal_message: Optional[str]
    # Control flow
    status: Literal["initial", "answering", "reflecting", "clarifying", "done"]
    clarification_count: int
    reflection_count: int


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

    def detect_unsafe(text: str) -> bool:
        """
        Lightweight profanity / hate / insult detection.

        This is intentionally conservative and can be tightened or replaced
        with a dedicated moderation model if needed.
        """
        if not text:
            return False

        lower = text.lower()

        profanity_patterns = [
            r"\b(fuck|f\*\*k|shit|bitch|asshole|bastard|cunt)\b",
        ]
        hate_patterns = [
            r"\b(?:kill|eliminate|exterminate|eradicate)\s+(?:them|those\s+people|all\s+\w+)\b",
        ]
        insult_patterns = [
            r"\b(you\s+are\s+stupid|you\s+are\s+an\s+idiot|worthless\s+piece\s+of)\b",
        ]

        patterns = profanity_patterns + hate_patterns + insult_patterns
        return any(re.search(p, lower) for p in patterns)

    def make_refusal_message() -> str:
        return (
            "I'm here to support your wellbeing, but I can't respond to "
            "hateful, abusive, or excessively profane language. "
            "Please rephrase your request respectfully so we can continue."
        )

    def node_user_input(state: ConversationState) -> ConversationState:
        """Seed the conversation with the latest user query, with safety check."""
        query = state["query"]

        if detect_unsafe(query):
            refusal = make_refusal_message()
            messages: List[AnyMessage] = [
                AIMessage(content=refusal),
            ]
            return {
                **state,
                "messages": messages,
                "status": "done",
                "safety_flag": "unsafe",
                "refusal_message": refusal,
            }

        messages = list(state.get("history", []))
        messages.append(HumanMessage(content=query))
        return {
            **state,
            "messages": messages,
            "status": "answering",
        }

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

    def node_clarification(state: ConversationState) -> ConversationState:
        """
        Ask a short, targeted follow-up question to gather more context.

        The frontend is expected to include the resulting messages in the
        next request's history so the agent can continue the conversation.
        """
        messages = state["messages"]

        clarification_prompt = (
            "Before I guide you with a meditation, I need a bit more context.\n"
            "Ask the user one short, compassionate follow-up question that helps "
            "you understand what they are experiencing or what they hope to get "
            "from this session. Respond only with that single question."
        )

        human = HumanMessage(content=clarification_prompt)
        ai_msg = _get_llm().invoke(messages + [human])
        messages = messages + [ai_msg]

        return {
            **state,
            "messages": messages,
            "clarification_count": state.get("clarification_count", 0) + 1,
            "status": "clarifying",
        }

    def node_assistant(state: ConversationState) -> ConversationState:
        """Main assistant step that can decide to call tools."""
        messages = state["messages"]
        ai_msg = chat_model_with_tools.invoke(messages)
        messages = messages + [ai_msg]

        # If the model decided to call tools, we defer transcript handling
        # until after the tools have been executed by the runtime using
        # the bound tools. Here we only work with already produced messages.

        # If no explicit transcript has been set yet and there is an AI answer
        # and no tools were called, we can later use the final AI content
        # as a fallback transcript.
        return {
            **state,
            "messages": messages,
        }

    def node_reflection(state: ConversationState) -> ConversationState:
        """
        Iterative reflection: the model critiques and, if needed, refines
        its previous answer into a clearer, more soothing response.
        """
        messages = state["messages"]
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

        if last_ai is None:
            return {
                **state,
                "status": "done",
            }

        reflection_prompt = (
            "You are reflecting on your previous response as a mindfulness coach.\n"
            "1. Briefly critique your last answer for clarity, empathy, and usefulness.\n"
            "2. Then provide an improved response, keeping only what helps the user.\n"
            "If the previous answer is already clear, empathetic, and helpful, "
            "respond with the single word SATISFIED.\n"
            "Otherwise, respond only with the improved response."
        )

        reflect_msg = HumanMessage(content=reflection_prompt)
        reflected = _get_llm().invoke(messages + [reflect_msg])

        content = reflected.content.strip() if isinstance(reflected.content, str) else str(reflected.content)

        max_reflections = 3
        current_count = state.get("reflection_count", 0)

        if content.upper() == "SATISFIED" or current_count >= max_reflections:
            # No further reflection; keep the existing last_ai message as final.
            return {
                **state,
                "reflection_count": current_count,
                "status": "done",
            }

        messages = messages + [AIMessage(content=content)]

        return {
            **state,
            "messages": messages,
            "reflection_count": current_count + 1,
        }

    def should_reflect(state: ConversationState) -> str:
        """
        Decide whether to go through a reflection pass.

        We reflect until the model is satisfied (signaled inside node_reflection)
        or until a maximum number of passes is reached.
        """
        if state.get("status") == "done":
            return "end"
        return "reflect"

    graph = StateGraph(ConversationState)
    graph.add_node("user", node_user_input)
    graph.add_node("clarification", node_clarification)
    graph.add_node("assistant", node_assistant)
    graph.add_node("reflection", node_reflection)

    graph.set_entry_point("user")
    graph.add_conditional_edges(
        "user",
        lambda s: "clarify" if needs_clarification(s) else "answer",
        {
            "clarify": "clarification",
            "answer": "assistant",
        },
    )
    graph.add_edge("clarification", END)
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
        "safety_flag": None,
        "refusal_message": None,
        "status": "initial",
        "clarification_count": 0,
        "reflection_count": 0,
    }

    final_state = app.invoke(initial_state)
    messages = final_state["messages"]
    last_ai = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
    content = last_ai.content if last_ai is not None else ""

    # Safety refusal: never return a transcript.
    if final_state.get("safety_flag") == "unsafe":
        refusal = final_state.get("refusal_message") or content
        return {
            "message": refusal,
            "transcript": None,
        }

    transcript = final_state.get("transcript")

    # If a tool was called and returned a JSON payload, prefer its transcript.
    if transcript is None:
        tool_msg = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)),
            None,
        )
        if tool_msg is not None:
            try:
                data = json.loads(tool_msg.content)
                if isinstance(data, dict) and "transcript" in data:
                    transcript = data.get("transcript")
            except Exception:
                # If parsing fails, silently fall back to other strategies.
                pass

    # Fallback: if no explicit transcript was set but we have an AI message
    # and the conversation reached a completed state, use the final AI content
    # as the transcript.
    if (
        final_state.get("status") == "done"
        and transcript is None
        and isinstance(content, str)
        and content.strip()
    ):
        transcript = content

    return {
        "message": content,
        "transcript": transcript,
    }

