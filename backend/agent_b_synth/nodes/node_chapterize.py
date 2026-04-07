import logging
import json
import re
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from ..state import SynthState, GraphContext, print_state
from ..prompts.chapterize import CHAPTERIZER_PROMPT


class ChapterizationOutput(BaseModel):
    """Pydantic model for chapterization output."""
    chapters: list[str]


logger = logging.getLogger(__name__)


def normalize_chapters(chapters: list[str]) -> list[str]:
    """
    Post-process chapters to enforce prompt rules:
    1. Every chapter must start with <emotion value="serene" />
    2. Break tags must be at the END of a chapter, never standalone
    3. Merge standalone break tags with preceding chapter
    4. Ensure each chapter ends with a break tag (append if missing)
    """
    normalized = []
    current_chapter = ""

    for chapter in chapters:
        chapter = chapter.strip()

        # Handle standalone break tags - merge with previous chapter
        if chapter.startswith("<break") or chapter == "<break":
            if current_chapter:
                # Append break to current chapter
                current_chapter += " " + chapter
            # else: ignore orphaned break at start
        else:
            # Save previous chapter if exists
            if current_chapter:
                # Ensure chapter ends with break tag
                if not re.search(r"<break.*time=", current_chapter):
                    current_chapter += " <break time=\"1.5s\" />"
                normalized.append(current_chapter)

            # Start new chapter with emotion tag
            if not chapter.startswith("<emotion"):
                chapter = "<emotion value=\"serene\" />" + chapter

            current_chapter = chapter

    # Don't forget the last chapter
    if current_chapter:
        # Ensure last chapter ends with break tag
        if not re.search(r"<break.*time=", current_chapter):
            current_chapter += " <break time=\"1.5s\" />"
        normalized.append(current_chapter)

    return normalized


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

    # Create structured output schema
    structured_llm = llm.with_structured_output(
        ChapterizationOutput, include_raw=False, method="json_schema")

    # Invoke LLM with structured output
    response: ChapterizationOutput = await structured_llm.ainvoke(messages)

    # Parse response - handle both dict (HuggingFace) and Pydantic model (Ollama)
    if isinstance(response, dict):
        chapters_data = response
    else:
        # Pydantic model instance - convert to dict
        chapters_data = response.model_dump()

    chapters = chapters_data.get("chapters", [])

    # Post-process to enforce prompt rules
    chapters = normalize_chapters(chapters)

    # Ensure we have at least one chapter
    if not chapters:
        chapters = [full_transcript]

    logger.info(f"Chapterized into {len(chapters)} chapters")

    return {
        "chapters": chapters,
        "is_chapterized": True
    }
