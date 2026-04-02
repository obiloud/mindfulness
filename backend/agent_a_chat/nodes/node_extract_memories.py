"""
Memory Extraction Node for Memorable Facts System

This node extracts biographical/anecdotal information from conversation history
to create memorable facts that can be stored and retrieved for natural conversation.
"""

import logging
import re
from datetime import datetime
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from shared.datamodels.memories import MemorableFact, MemoryCategory, MemoryValence
from langgraph.runtime import Runtime
from ..state import GraphContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryExtractionState(TypedDict):
    """State for memory extraction node."""
    messages: List[BaseMessage]
    summary: str
    user_id: str


async def extract_memorable_facts(state: MemoryExtractionState, runtime: Runtime[GraphContext]) -> dict:
    """
    Extract memorable facts from conversation history.

    This function analyzes the conversation and identifies:
    - Biographical information
    - Anecdotes and life events
    - Important people in user's life
    - Recurring goals and interests
    - Significant obstacles or challenges

    Args:
        state: Current state containing messages and summary

    Returns:
        Dictionary containing extracted memorable facts
    """
    llm = runtime.context.llm

    messages = state["messages"]
    summary = state.get("summary", "")
    user_id = state["user_id"]

    logger.info(
        f"Extracting memorable facts for user {user_id}, summary: {summary[:100]}...")

    # Create extraction prompt
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a memory extraction assistant for an empathetic AI companion.
        Your task is to identify memorable facts from conversation history that would help
        create a natural, friend-like experience.
        
        Focus on extracting:
        1. Biographical information (hobbies, interests, background)
        2. Anecdotes and life events (recent experiences, achievements)
        3. Important people (family, friends, colleagues mentioned)
        4. Recurring goals and interests (what they care about)
        5. Significant obstacles or challenges (struggles, fears)
        
        IMPORTANT: Do NOT extract:
        - Temporary preferences (voice settings, TTS configuration)
        - One-off statements without significance
        - Generic statements that anyone could make
        - Information already stored in preferences
        
        For each fact, provide:
        - category: One of [person, location, event, goal, obstacle, preference, insight, emotion, anecdote, interest]
        - content: The memorable fact itself
        - category: One of [biographical, anecdote, person, goal, obstacle, interest]
        - valence: One of [positive, neutral, negative]
        - timestamp: ISO 8601 timestamp (use current time if unknown)
        - confidence: Score from 0.0 to 1.0 indicating how memorable this is
        
        Format your output as a JSON array of objects with these fields.
        """),
        ("human", """Conversation Summary:
        {summary}
        
        Messages:
        {messages}
        
        Extract memorable facts from this conversation."""),
    ])

    # Create LLM chain for extraction
    llm_chain = extraction_prompt | llm | StrOutputParser()

    # Run extraction
    try:
        raw_output = await llm_chain.ainvoke({
            "summary": summary,
            "messages": messages
        })

        logger.info(f"Raw extraction output: {raw_output[:200]}...")

        # Parse and validate the output
        facts = parse_extraction_output(raw_output)

        if facts:
            logger.info(f"Extracted {len(facts)} memorable facts")
        else:
            logger.info("No memorable facts extracted from this conversation")

        return {"memorable_facts": facts}

    except Exception as e:
        logger.error(f"Error extracting memorable facts: {e}")
        return {"memorable_facts": []}


def parse_extraction_output(raw_output: str) -> List[MemorableFact]:
    """
    Parse the raw LLM output into MemorableFact objects.

    Args:
        raw_output: Raw string output from LLM

    Returns:
        List of MemorableFact objects
    """
    import json

    try:
        # Clean and parse as JSON
        raw_output = raw_output.strip()
        
        # Remove markdown code blocks if present
        if raw_output.startswith("```"):
            raw_output = raw_output.split("`", 1)[1]
        if raw_output.endswith("```"):
            raw_output = raw_output.rsplit("`", 1)[0]
        
        raw_output = raw_output.strip()
        
        # Handle malformed JSON by attempting to extract valid JSON array
        # The LLM sometimes adds extra text after the JSON array
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        
        if json_match:
            raw_output = json_match.group(0)
        
        facts_data = json.loads(raw_output)

        if not isinstance(facts_data, list):
            logger.warning(f"Expected list of facts, got: {type(facts_data)}")
            return []

        facts = []
        for fact_data in facts_data:
            try:
                # Create MemorableFact from parsed data
                # Convert timestamp string to datetime if provided
                timestamp = fact_data.get("timestamp", "")
                if timestamp:
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.utcnow()
                else:
                    timestamp = datetime.utcnow()

                fact = MemorableFact(
                    content=fact_data.get("content", ""),
                    category=MemoryCategory(
                        fact_data.get("category", "anecdote")),
                    valence=MemoryValence(fact_data.get("valence", "neutral")),
                    timestamp=timestamp,
                    confidence=fact_data.get("confidence", 0.5)
                )
                facts.append(fact)
            except Exception as e:
                logger.warning(f"Failed to parse fact: {e}")
                continue

        return facts

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction output as JSON: {e}")
        logger.error(f"Raw output: {raw_output[:500]}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing extraction output: {e}")
        return []


async def store_memorable_fact(fact: MemorableFact, user_id: str, runtime: Runtime[GraphContext]) -> bool:
    """
    Store a memorable fact in the vector store.

    Args:
        fact: The memorable fact to store
        user_id: The user ID for the fact

    Returns:
        True if storage was successful
    """

    store = runtime.store

    # Create content for embedding
    content = f"{fact.content} | {fact.category} | {fact.valence}"

    # Convert datetime to ISO string for JSON serialization
    timestamp_str = fact.timestamp.isoformat() if fact.timestamp else datetime.utcnow().isoformat()

    # Create metadata with type filter
    metadata = {
        "type": "memorable_fact",
        "category": fact.category,
        "valence": fact.valence,
        "timestamp": timestamp_str,
        "confidence": fact.confidence,
        "user_id": user_id
    }

    try:
        # Store the fact
        await store.aput(
            ("memories", user_id),
            f"fact_{user_id}_{timestamp_str}",
            {"content": content, "metadata": metadata}
        )

        logger.info(f"Stored memorable fact: {fact.content[:100]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to store memorable fact: {e}")
        return False


async def store_memorable_facts(facts: List[MemorableFact], user_id: str, runtime: Runtime[GraphContext]) -> int:
    """
    Store multiple memorable facts in the vector store.

    Args:
        facts: List of memorable facts to store
        user_id: The user ID for the facts

    Returns:
        Number of facts successfully stored
    """
    stored_count = 0

    for fact in facts:
        if await store_memorable_fact(fact, user_id, runtime):
            stored_count += 1

    return stored_count
