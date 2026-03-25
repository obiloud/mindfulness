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


class ChatState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    long_term_memory: Optional[str]
    turn_count: int
    info_score: float
    summary: str
    answer: Optional[str]
    transcript: Optional[str]
    synth_status: Literal["idle", "requested", "in_progress", "completed"]
    is_synthesis_ready: bool
    awaiting_confirmation: bool


def print_state(state: ChatState) -> str:
    print_data = {
        **state,
        # "messages": messages_to_dict(state["messages"])
        "messages": []
    }
    return json.dumps(print_data, indent=2)
