"""
Data models for Memorable Facts memory system.
These capture biographical/anecdotal information for natural conversation.
"""
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import logging
import inspect
from langchain_core.language_models import BaseChatModel
from langchain_core.stores import BaseStore

logger = logging.getLogger(__name__)


class MemoryCategory(str, Enum):
    """Categories for memorable facts to enable better retrieval."""
    PERSON = "person"           # Family, friends, pets
    LOCATION = "location"       # Cities, places, travel
    EVENT = "event"             # Life events, milestones
    GOAL = "goal"               # Current or past goals
    OBSTACLE = "obstacle"       # Challenges, struggles
    PREFERENCE = "preference"   # Likes/dislikes (personal, not voice)
    INSIGHT = "insight"         # Realizations, beliefs
    EMOTION = "emotion"         # Emotional states, feelings
    INTEREST = "interest"       # Topics, hobbies, passions
    ANECDOTE = "anecdote"       # Personal stories and experiences


class MemoryValence(str, Enum):
    """Emotional valence of the memory."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class MemorableFact(BaseModel):
    """
    A memorable fact about the user that can be retrieved semantically.

    These are distinct from voice/profile preferences - they are personal
    anecdotes, life details, and recurring topics that make the agent feel
    like a friend who remembers.
    """
    content: str = Field(
        ...,
        description="The memorable fact itself (e.g., 'User's dog is named Luna', "
                    "'User is feeling anxious about the move to Belgrade')"
    )
    category: MemoryCategory = Field(
        default=MemoryCategory.EVENT,
        description="Category of the memory for better filtering."
    )
    valence: MemoryValence = Field(
        default=MemoryValence.NEUTRAL,
        description="Emotional tone of the memory."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this fact was learned (prioritizes recent updates)."
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="LLM's confidence in this fact (0.0-1.0)."
    )

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Content must be at least 5 characters long")
        return v.strip()

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        if v is None:
            return datetime.utcnow()
        return v


class MemorySearchResult(BaseModel):
    """Result from semantic memory search."""
    content: str
    category: MemoryCategory
    valence: MemoryValence
    timestamp: datetime
    confidence: float
    score: float = Field(
        ...,
        description="Cosine similarity score (higher = more relevant)."
    )

    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        if v < 0.35:  # Friendship threshold
            return None  # Filter out low-confidence matches
        return v


class Memories(BaseModel):
    """Container for a list of memorable facts."""
    memorable_facts: List[MemorableFact] = Field(
        description="List of memorable facts about the user."
    )


async def extract_memorable_facts(llm: BaseChatModel, memory: str, message: str) -> List[MemorableFact]:
    """
    Extract memorable facts from conversation history.

    This function analyzes the conversation and identifies:
    - Biographical information
    - Anecdotes and life events
    - Important people in user's life
    - Recurring goals and interests
    - Significant obstacles or challenges

    Args:
        llm: The LLM instance to use for extraction
        memory: Already stored memory
        message: The user message to analyze

    Returns:
        List of MemorableFact objects extracted from the message
    """

    allowed_categories = [c for c in MemoryCategory]
    allowed_valence = [v for v in MemoryValence]

    # Create extraction prompt
    memory_extraction_prompt = inspect.cleandoc("""
        You are a memory extraction assistant for an empathetic AI companion.
        Your task is to identify memorable facts from user input that would help
        create a natural, friend-like experience.
        
        STORED MEMORIES: 
        ```
        {memory}
        ```

        USER INPUT: "{message}"
        
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
        
        CRITICAL: For each fact, provide:
        - category: One of {allowed_categories}
        - content: The memorable fact itself
        - valence: One of {allowed_valence}
        - timestamp: ISO 8601 timestamp (use current time if unknown)
        - confidence: Score from 0.0 to 1.0 indicating how memorable this is
        
        ### OUTPUT FORMAT:
        You must return a valid JSON object ONLY. Do not include any preamble.
        {{
            "memorable_facts": list[...]
        }}
        """).strip()

    memory_extraction_message = memory_extraction_prompt.format(
        memories=memory,
        message=message,
        allowed_categories=allowed_categories,
        allowed_valence=allowed_valence,
    )

    # Create LLM chain for extraction using structured output
    structured_llm = llm.with_structured_output(Memories, method="json_schema")

    # Run extraction
    try:
        # Extract facts using structured output
        # The LLM will return a list of MemorableFact objects
        memories = await structured_llm.ainvoke(memory_extraction_message)

        if isinstance(memories, dict):
            memories = Memories(**memories)

        logger.info(f"MEMORIES {memories.model_dump_json()}")

        facts = memories.memorable_facts

        logger.info(f"Facts extraction output: {facts[:200]}...")

        if facts:
            logger.info(f"Extracted {len(facts)} memorable facts")
        else:
            logger.info("No memorable facts extracted from this conversation")

        return facts

    except Exception as e:
        logger.error(f"Error extracting memorable facts: {e}")
        return []


async def store_memorable_fact(fact: MemorableFact, user_id: str, store: BaseStore) -> bool:
    """
    Store a memorable fact in the vector store.

    Args:
        fact: The memorable fact to store
        user_id: The user ID for the fact
        store: The vector store to use for storage

    Returns:
        True if storage was successful
    """

    # Create content for embedding
    content = f"{fact.content} | {fact.category} | {fact.valence}"

    # Convert datetime to ISO string for JSON serialization
    timestamp_str = fact.timestamp.isoformat(
    ) if fact.timestamp else datetime.utcnow().isoformat()

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


async def store_memorable_facts(facts: List[MemorableFact], user_id: str, store: BaseStore) -> int:
    """
    Store multiple memorable facts in the vector store.

    Args:
        facts: List of memorable facts to store
        user_id: The user ID for the facts
        store: The vector store to use for storage

    Returns:
        Number of facts successfully stored
    """
    stored_count = 0

    for fact in facts:
        if await store_memorable_fact(fact, user_id, store):
            stored_count += 1

    return stored_count
