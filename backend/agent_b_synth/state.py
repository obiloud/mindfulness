from typing import TypedDict
from dataclasses import dataclass
from langchain_huggingface import ChatHuggingFace
import logging
import json


@dataclass
class GraphContext:
    logger: logging.Logger
    llm: ChatHuggingFace


class SynthState(TypedDict):
    context: str  # The summary passed from Agent A
    draft_transcript: str
    draft_answer: str
    is_transcript_valid: bool
    is_answer_valid: bool
    final_transcript: str


def print_state(state: SynthState) -> str:
    print_data = {
        **state,
        # "messages": messages_to_dict(state["messages"])
        "messages": []
    }
    return json.dumps(print_data, indent=2)
