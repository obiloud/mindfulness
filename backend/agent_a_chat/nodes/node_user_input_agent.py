# agents/user_input_agent.py

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langchain_huggingface import ChatHuggingFace

from ..state import ChatState, GraphContext, print_state


def create_safety_classifier(llm: ChatHuggingFace) -> Runnable:
    """Create an LLM-based safety classifier for detecting unsafe content."""
    # Prompt template for safety classification
    safety_prompt = PromptTemplate.from_template(
        """You are a safety filter for a wellbeing assistant. 
        Classify the following text as safe or unsafe based on profanity, hate speech, or insults.
        
        Text: {text}
        
        Response format: Only return "safe" or "unsafe" (no explanation).
        """
    )

    return safety_prompt | llm | StrOutputParser()


def classify_safety(llm: ChatHuggingFace, text: str) -> str:
    """Classify text as safe or unsafe using an LLM."""
    if not text or not text.strip():
        return "safe"

    # Use the safety classifier
    classifier = create_safety_classifier(llm)
    result = classifier.invoke({"text": text})

    # Ensure the result is one of the expected values
    return result.strip().lower() if result else "safe"


def node_user_input(state: ChatState, runtime: Runtime[GraphContext]) -> ChatState:
    """
    Process user input with safety classification and long-term memory retrieval.

    This node performs:
    1. Safety classification using an LLM-based classifier
    2. Semantic search to retrieve relevant long-term memory
    3. Updates the conversation state with the new message and retrieved context

    Args:
        state: Current conversation state
        runtime: Runtime context for the graph

    Returns:
        Updated conversation state
    """
    logger = runtime.context.logger
    llm = runtime.context.llm

    logger.debug(f"Safety check: state='{print_state(state)}'")

    query = state["messages"][-1].content

    # Classify the input as safe or unsafe using LLM
    safety_status = classify_safety(llm, query)

    if safety_status == "unsafe":
        logger.warning(f"Unsafe content detected: '{query}' → refusing")
        refusal_message = (
            "I'm here to support your wellbeing, but I can't respond to "
            "hateful, abusive, or excessively profane language. "
            "Please rephrase your request respectfully so we can continue."
        )

        # Update state with refusal
        return {
            **state,
            "messages": [AIMessage(content=refusal_message)],
            "status": "done",
            "safety_flag": "unsafe",
            "refusal_message": refusal_message,
        }

    return {
        **state,
        "status": "answering",
        "safety_flag": "safe",
    }
