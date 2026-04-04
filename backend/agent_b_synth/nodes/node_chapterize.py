import logging
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from ..state import SynthState, GraphContext, print_state
from ..prompts.chapterize import CHAPTERIZER_PROMPT

logger = logging.getLogger(__name__)


async def node_chapterize(state: SynthState, runtime: Runtime[GraphContext], config: RunnableConfig) -> dict:
    """
    Splits the full meditation transcript into semantically coherent chapters.
    Uses LLM to identify natural transition points while preserving instructional flow.
    """
    llm = runtime.context.llm
    logger.info(f"CHAPTERIZING: {print_state(state)}")

    full_transcript = state.get("transcript", "")

    if not full_transcript or len(full_transcript.strip()) < 100:
        logger.warning("Transcript too short for chapterization")
        return {
            "chapters": [full_transcript],
        }

    # Prepare the prompt with the full transcript
    messages = [
        SystemMessage(content=CHAPTERIZER_PROMPT),
        HumanMessage(content=full_transcript)
    ]

    # Invoke LLM with JSON schema output
    response = await llm.ainvoke(messages)

    # Parse JSON output
    try:
        chapters_data = json.loads(response.content)
        chapters = chapters_data.get("chapters", [])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse chapterization response: {e}")
        # Fallback: split by double newlines if JSON parsing fails
        raw_chapters = full_transcript.split("\n\n")
        chapters = [ch.strip() for ch in raw_chapters if ch.strip()]
        logger.warning(
            f"Fallback to naive split, got {len(chapters)} chapters")

    # Ensure we have at least one chapter
    if not chapters:
        chapters = [full_transcript]

    logger.info(f"Chapterized into {len(chapters)} chapters")

    return {
        "chapters": chapters,
        "is_chapterized": True
    }
