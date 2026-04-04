"""
Data models for Memorable Facts memory system.
These capture biographical/anecdotal information for natural conversation.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


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
