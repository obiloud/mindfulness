from dataclasses import dataclass
from typing import Annotated, TypedDict, Literal, Optional, List
from langchain_core.messages import AnyMessage, messages_to_dict
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace
import logging
import json


@dataclass
class GraphContext:
    logger: logging.Logger
    llm: ChatHuggingFace


class ConversationState(TypedDict):
    """Shared state for the mindfulness LangGraph agent."""

    query: str
    long_term_memory: Optional[str]
    history: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]
    answer: Optional[str]
    transcript: Optional[str]
    # Safety / refusal
    safety_flag: Optional[str]
    refusal_message: Optional[str]
    # Control flow
    status: Literal["initial", "conversation", "answering",
                    "reflecting", "done"]
    info_score: float
    turn_count: int
    reflection_count: int
    reflection_notes: Optional[str]


def print_state(state: ConversationState, full: bool = False) -> str:
    print_data = {
        **state,
        "history": messages_to_dict(state["history"]),
        "messages": messages_to_dict(state["messages"])
    }
    return json.dumps(print_data, indent=2)
