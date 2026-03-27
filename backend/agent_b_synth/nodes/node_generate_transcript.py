import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from ..state import SynthState, GraphContext, print_state
from ..prompts.meditation import generate_meditation_prompt

logger = logging.getLogger(__name__)


def fix_ssml(text: str) -> str:
    # Fixes <break time="1s"> to <break time="1s" />
    return re.sub(r'<break time="([^"]+)"\s*>', r'<break time="\1" />', text)


def clean_escaped_newlines(text: str) -> str:
    """
    Removes all instances of escaped newlines (\\n) from the string 
    while preserving actual newlines (\n).

    Args:
        text (str): The input string to clean

    Returns:
        str: The cleaned string with escaped newlines removed
    """
    # Replace \\n with empty string (removes escaped newlines)
    # This preserves actual \n characters
    return re.sub(r'\\n', '', text)


async def node_generate_transcript(state: SynthState, runtime: Runtime[GraphContext], config: RunnableConfig) -> dict:
    llm = runtime.context.llm
    logger.info(f"TRANSCRIPT: {print_state(state)}")

    # Check if we are refining based on feedback
    feedback = state.get("transcript_feedback", "")
    messages = state.get("messages", [])

    context_text = ""

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if human_messages:
        context_text = "\n\n".join(m.content for m in human_messages)

    content = f"Context: {context_text}\nFeedback: {feedback}" if feedback else context_text

    voice_blueprint = config["configurable"].get("voice_blueprint")
    mindfulness_profile = config["configurable"].get("mindfulness_profile")

    system = SystemMessage(content=generate_meditation_prompt(
        profile=mindfulness_profile, blueprint=voice_blueprint))
    human = HumanMessage(content=content)

    # Async invocation
    response = await llm.ainvoke([system, human])

    # Clean the transcript by removing escaped newlines
    cleaned_transcript = clean_escaped_newlines(response.content)

    return {
        "transcript": fix_ssml(cleaned_transcript),
        "is_transcript_valid": False  # Reset for re-validation
    }
