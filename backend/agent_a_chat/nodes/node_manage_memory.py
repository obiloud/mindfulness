from typing import Any, Dict, Optional
from typing import Any, Dict
from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig


async def node_manage_memory(state: ChatState, runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Logic for saving facts cross-session.
    """
    llm = runtime.context.llm

    # 1. Get user ID from config to ensure we save to the right drawer
    user_id = config["configurable"].get("user_id")
    namespace = ("memories", user_id)

    # 2. Look at the last message.
    # Usually, you'd use a small LLM call here to "extract_preferences"
    last_msg = state["messages"][-1].content

    # Pseudo-logic: If LLM identifies a preference (e.g., "I hate cilantro")
    extracted_preference = extract_facts_with_llm(llm, last_msg)

    if extracted_preference:
        # 3. Store the fact permanently
        # store.put(namespace, key, value)
        await runtime.store.put(namespace, "food_preferences", {"dislikes": "cilantro"})

    return {"messages": []}  # No change to message history needed


def extract_facts_with_llm(llm, message: str) -> Optional[Dict[str, Any]]:
    """
    Extract personal preferences from a user message using an LLM.

    Args:
        llm: The language model instance to use for analysis
        message: The user message to analyze

    Returns:
        A dictionary of extracted preferences if found, otherwise None
    """
    # Prompt template to guide the LLM to extract preferences
    prompt = (
        "Analyze the following message and extract personal preferences. "
        "Return only a JSON object with the preferences in the format:\n"
        "{\n"
        "  \"food\": [\"favorite\", \"dislikes\"],\n"
        "  \"colors\": [\"favorite\", \"dislikes\"],\n"
        "  \"music\": [\"favorite\", \"dislikes\"],\n"
        "  \"time_of_day\": [\"favorite\", \"dislikes\"],\n"
        "  \"other\": [\"favorite\", \"dislikes\"]\n"
        "}\n\n"
        "If no preferences are mentioned, return an empty object.\n"
        "Do not add any explanations or additional text.\n\n"
        "Message: {message}\n"
    ).format(message=message)

    # Call the LLM to get a response
    try:
        # In a real implementation, this would call the actual LLM API
        # For now, we'll simulate the LLM response with a simple heuristic
        # In production, this would be a real LLM call via the llm interface

        # Simulate LLM response - in real implementation, this would be:
        # response = llm.invoke([HumanMessage(content=prompt)])

        # Heuristic-based extraction for demonstration
        preferences = {}

        # Extract food preferences
        if "food" in message.lower() or "like" in message.lower() or "dislike" in message.lower():
            food_keywords = ["favorite", "like", "love",
                             "hate", "dislike", "don't like", "can't stand"]
            food_matches = []
            for kw in food_keywords:
                if kw in message.lower():
                    food_matches.append(kw)
            if food_matches:
                preferences["food"] = food_matches

        # Extract color preferences
        if "color" in message.lower() or "color" in message.lower():
            color_keywords = ["favorite", "like", "love",
                              "hate", "dislike", "dislike", "can't stand"]
            color_matches = []
            for kw in color_keywords:
                if kw in message.lower():
                    color_matches.append(kw)
            if color_matches:
                preferences["colors"] = color_matches

        # Extract music preferences
        if "music" in message.lower() or "sound" in message.lower():
            music_keywords = ["favorite", "like", "love",
                              "hate", "dislike", "dislike", "can't stand"]
            music_matches = []
            for kw in music_keywords:
                if kw in message.lower():
                    music_matches.append(kw)
            if music_matches:
                preferences["music"] = music_matches

        # Extract time of day preferences
        if "day" in message.lower() or "night" in message.lower():
            time_keywords = ["favorite", "like", "love",
                             "hate", "dislike", "dislike", "can't stand"]
            time_matches = []
            for kw in time_keywords:
                if kw in message.lower():
                    time_matches.append(kw)
            if time_matches:
                preferences["time_of_day"] = time_matches

        # Extract other preferences
        if any(k in message.lower() for k in ["prefer", "like", "love", "hate", "dislike"]):
            other_keywords = ["prefer", "like", "love",
                              "hate", "dislike", "can't stand"]
            other_matches = []
            for kw in other_keywords:
                if kw in message.lower():
                    other_matches.append(kw)
            if other_matches:
                preferences["other"] = other_matches

        # Return only if we have actual preferences
        return preferences if preferences else None

    except Exception as e:
        # Log error (in production)
        # print(f"Error extracting preferences: {e}")
        return None
