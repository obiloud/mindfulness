from typing import TypedDict
from dataclasses import dataclass
from langchain.chat_models import BaseChatModel
import logging
import json


@dataclass
class GraphContext:
    logger: logging.Logger
    llm: BaseChatModel


class SynthState(TypedDict):
    context: str
    transcript: str
    answer: str
    is_transcript_valid: bool
    is_answer_valid: bool


def print_state(state: SynthState) -> str:
    print_data = {
        **state,
        # "messages": messages_to_dict(state["messages"])
        "messages": []
    }
    return json.dumps(print_data, indent=2)
