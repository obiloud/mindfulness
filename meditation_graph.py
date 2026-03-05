import os
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, END
from story_generator_pipeline import meditation_guide_generator_chain
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

import json
import re

load_dotenv(override=True)

class AssistantResponse(BaseModel):
    answer: str = Field(description="The empathetic, conversational response to the user.")
    transcript: str = Field(description="The step-by-step guided meditation script.")

class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if the answer and transcript meet all safety and quality guidelines."
    )
    feedback: str = Field(
        description="Detailed feedback if the content needs revision. Be specific about what is missing or wrong."
    )

class ConversationState(TypedDict):
    """Shared state for the mindfulness LangGraph agent."""

    query: str
    history: List[AnyMessage]
    messages: List[AnyMessage]
    answer: Optional[str]
    transcript: Optional[str]
    # Safety / refusal
    safety_flag: Optional[str]
    refusal_message: Optional[str]
    # Control flow
    status: Literal["initial", "answering", "reflecting", "clarifying", "done"]
    clarification_count: int
    reflection_count: int
    reflection_notes: Optional[str]


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
        top_p=0.8,
        repetition_penalty=1.1,
        huggingfacehub_api_token=hf_token,
        provider="auto",
    )

    return ChatHuggingFace(llm=llm)


@tool
def generate_transcript(context: str):
    """Generates a guided meditation session transcript.

        Args:
            context (str): User's context/condition

        Returns:
            str: JSON string containing the transcript of the session.
    """
    result = meditation_guide_generator_chain.invoke({"query": context})
    transcript = result.get("text", "") if isinstance(result, dict) else str(result)

    print(transcript)

    return transcript

tools = [generate_transcript]
tool_node = ToolNode(tools)

def build_mindfulness_graph():
    """
    Build a LangGraph graph that can:
    - hold a short conversation about the user's context
    - optionally call tools (e.g. guided meditation audio session)
    - reflect on its answer once and improve it.
    """
    llm = _get_llm()

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

        system_prompt = """You are the Clarification Node for a Mindfulness Coach. Your sole task is to gather necessary context before a meditation is generated.

When a user mentions a condition (e.g., anxiety, stress), ask targeted follow-up questions:

    What specifically triggers or exacerbates this feeling?

    How long have you been experiencing this, and what is the current severity?

    What have you tried in the past to manage this?

Constraint: Do not provide exercises yet. Only ask the questions needed to move to the next phase."""

        clarification_prompt = """Before I guide you with a meditation, I need a bit more context.
Ask the user one short, compassionate follow-up question that helps you understand what they are experiencing or what they hope to get from this session.
Respond only with that single question."""

        system = SystemMessage(content=system_prompt)
        human = HumanMessage(content=clarification_prompt)
        ai_msg = llm.invoke([system] + messages + [human])
        messages = messages + [ai_msg]

        return {
            **state,
            "messages": messages,
            "clarification_count": state.get("clarification_count", 0) + 1,
            "status": "clarifying",
        }

    def node_assistant(state: ConversationState) -> ConversationState:
        """Main assistant step that generates both answer and transcript."""

        messages = state["messages"]

        # Decide whether this is the initial pass or a refinement loop.
        reflection_notes = state.get("reflection_notes")
        is_refinement = bool(reflection_notes)

        # 1. Ensure we have a meditation transcript, generating it directly in code.
        transcript = state.get("transcript")
        if not transcript or not transcript.strip():
            # Build a simple textual context from the conversation.
            human_messages = [m for m in messages if isinstance(m, HumanMessage)]
            if human_messages:
                context_text = "\n\n".join(m.content for m in human_messages)
            else:
                context_text = state.get("query", "")

            try:
                raw_transcript = meditation_guide_generator_chain.invoke({"query": context_text})
                transcript = raw_transcript.get("text", "") if isinstance(raw_transcript, dict) else str(raw_transcript)
            except Exception:
                # In case transcript generation fails, fall back to an empty string.
                transcript = ""

        # 2. Generate only the conversational answer via the chat model.
        system_prompt = """You are the Mindfulness Assistant. Your role is to resolve emotional struggles through supportive, empathetic conversation and to introduce a guided meditation (which has already been generated separately).

Task: Based on the user's clarified situation, generate a personalized answer:

    Answer: Be empathetic and supportive. Acknowledge the user's feelings, normalise their experience, and offer gentle guidance. You may briefly mention that a guided meditation has been prepared for them, but do not include the full script.

Constraints:
- Do NOT output the meditation transcript or script itself.
- Respond in a warm, concise, and human tone.
- Avoid medical or clinical diagnoses."""

        if reflection_notes:
            system_prompt += f"\n\nIMPORTANT: Please address this feedback when refining your answer: {reflection_notes}"

        system = SystemMessage(content=system_prompt)

        # Provide the model with the user context and the transcript as background only.
        answer_context = f"""User query: {state.get('query', '')}

Meditation transcript (for your reference only, do NOT repeat it verbatim):
{transcript}
"""
        answer_human = HumanMessage(
            content=answer_context
        )

        ai_msg = llm.invoke([system] + messages + [answer_human])
        answer_text = ai_msg.content if isinstance(ai_msg, AIMessage) else str(ai_msg)

        return {
            **state,
            "answer": answer_text,
            "transcript": transcript,
            "messages": messages + [ai_msg],
            "status": "answering" if not is_refinement else state.get("status", "answering"),
        }

    def node_reflection(state: ConversationState) -> ConversationState:
        """
        Iterative reflection: the model critiques and, if needed, refines
        its previous answer into a clearer, more soothing response.
        """

        if not state.get("transcript") or len(state["transcript"].strip()) < 10:
            return {
                **state,
                "reflection_notes": "The assistant failed to generate a meditation transcript. Please provide a full script.",
            }

        messages = state["messages"]
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

        if last_ai is None:
            return {
                **state,
                "status": "done",
            }

        # Ask the model for a simple, parseable reflection signal.
        reflection_prompt = f"""You are a Senior Mindfulness Supervisor. Review the following Assistant response:

ANSWER TO USER: {state['answer']}
MEDITATION TRANSCRIPT: {state['transcript']}

CRITERIA:
1. SAFETY: Does it avoid medical diagnoses or clinical advice?
2. LANGUAGE: Is the tone simple, accessible, and free of jargon?
3. STRUCTURE: Does the transcript include an Intro, Practice, and Affirmations?
4. PERSONALIZATION: Is it tailored to the user's specific triggers without making assumptions?

Respond in EXACTLY ONE LINE using one of these formats:
- \"SATISFACTORY\"  (if the answer and transcript fully meet all criteria)
- \"REVISION_NEEDED: <short explanation of what must be improved>\"
"""

        reflect_msg = SystemMessage(content=reflection_prompt)
        reflect_ai = llm.invoke([reflect_msg] + messages)
        reflect_text = reflect_ai.content.strip() if isinstance(reflect_ai, AIMessage) else str(reflect_ai).strip()

        max_reflections = 3
        current_count = state.get("reflection_count", 0)

        lower = reflect_text.lower()
        if lower.startswith("satisfactory") or current_count >= max_reflections:
            return {
                **state,
                "reflection_count": current_count,
                "status": "done",
            }

        feedback = reflect_text
        if ":" in reflect_text:
            feedback = reflect_text.split(":", 1)[1].strip() or reflect_text

        return {
            **state,
            "messages": messages,
            "reflection_count": current_count + 1,
            "reflection_notes": feedback,
        }

    def router_function(state: ConversationState) -> Literal["reflect", "end"]:
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
        router_function,
        {
            "reflect": "reflection",
            "end": END,
        },
    )
    graph.add_edge("reflection", "assistant")

    return graph.compile()


def run_mindfulness_graph(query: str, history: Optional[List[AnyMessage]] = None):
    """
    Convenience function for external callers (FastAPI, CLI, etc.).
    Returns the last assistant message content and any generated metadata.
    """
    app = build_mindfulness_graph()

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

    return {
        "message": content,
        "transcript": transcript,
    }

if __name__ == "__main__":
    graph = build_mindfulness_graph()

    with open("graph_output(1).png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())