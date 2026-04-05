import inspect
import logging
from pydantic import BaseModel, Field
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages
)
from langgraph.runtime import Runtime
from ..state import ChatState, GraphContext, print_state
from ..prompts.conversation import CONVERSATION_PROMPT

logger = logging.getLogger(__name__)


class EvaluationOutput(BaseModel):
    summary: str = Field(description="The conversation summary")
    info_score: float = Field(description="The information score")


def get_trimmed_messages(state):
    """
    A helper to keep the LLM focused on the 'now'.
    Returns the trimmed messages that are suitable for processing.
    """
    # Define your strategy
    try:
        # Trim messages to keep only the most relevant ones
        # We use a strategy that keeps the most recent messages
        # and ensures we start with a user query
        trimmed_messages = trim_messages(
            messages=state["messages"],
            max_tokens=2000,           # Don't exceed context window
            strategy="last",          # Keep the most recent messages
            # Use length as token counter (simple approach)
            token_counter=len,
            include_system=True,      # ALWAYS keep the system prompt
            start_on="human"          # Ensure the history starts with a user query
        )

        # Ensure we have a valid list of messages
        if not isinstance(trimmed_messages, list):
            trimmed_messages = []

        # If no messages, return empty list
        if len(trimmed_messages) == 0:
            return []

        return trimmed_messages

    except Exception as e:
        # If trimming fails, return original messages as fallback
        print(f"Error during message trimming: {e}")
        return state["messages"]


async def node_conversation(state: ChatState, runtime: Runtime[GraphContext]) -> ChatState:
    """
    Ask a short, targeted follow-up question to gather more context.

    The frontend is expected to include the resulting messages in the
    next request's history so the agent can continue the conversation.
    """
    llm = runtime.context.llm

    logger.debug(f"Conversation continued: state='{print_state(state)}'")

    # Trim the messages to keep only the most relevant ones
    trimmed_messages = get_trimmed_messages(state)

    logger.debug(f"TRIMMED MESSAGES: {trimmed_messages}")

    # Create a prompt that combines the entire conversation history
    # This prompt asks the LLM to evaluate if the conversation is mature enough
    maturity_prompt = inspect.cleandoc(
        """
    You are a mindfulness conversation evaluator. Your task is to determine if the conversation
    between the user and the agent has reached a mature state where we can proceed to generating
    a response (answering) or wait for the user to respond.
    
    Summarize and analyze the following conversation history. Consider the following criteria:
    
    1. Topic coherence: Are the topics related and meaningful? Does the conversation flow naturally
        from one topic to another without jumping between unrelated subjects?
    2. Conversation length: Is the conversation too long? Consider that sometimes it takes multiple
        messages to express an idea, and sometimes a single message can express multiple ideas.
        As a fallback, if the total number of messages exceeds 10, the conversation is considered
        too long and should proceed to answering.
    3. User context: Does the conversation provide sufficient context about the user's current
        mood and state of mind? Does the user express clear needs, goals, or emotional states?
    4. Verify that the latest agent's message is not asking any follow up questions. 
    
    The conversation is allowed to transition between topics naturally, as long as they are
    related and meaningful. However, if the conversation has drifted too far from the core
    topic or has become excessively long, it should proceed to answering.
    
    Evaluate the conversation based on these criteria. Provide only the score from 0 to 1.
    
    Conversation history:
    {conversation_summary}
    {conversation_history}
                                    
    ### OUTPUT FORMAT:
    You MUST return a valid JSON object ONLY. Do not include any preamble.
    ```
    {{
        "summary": str,
        "info_score": float
    }}
    ```
    """).strip()

    clarification_prompt = inspect.cleandoc(
        """
    Be agreeable with the user. 
    If appropriate, make a short statement to reinforce or confirm the user's beliefs or claims without repeating the user's exact words.
    Ask the user a short, compassionate follow-up question to gather information about their mood, state of mind, insecurities, or doubts.

    # IMPORTANT: 
    Do NOT repeat the user's exact words.
    Do NOT repeat the questions you already asked.
    If the user has clearly described what he expects from this session, simply acknowledge it and Do NOT ask any further questions.

    Conversation history:
    {conversation_history}
    """).strip()

    system_prompt = CONVERSATION_PROMPT.format(
        memories=state.get("long_term_memory", "No history available."),
        summary=state.get("summary", "New conversation.")
    )

    logger.info(f"CONVERSATION SETTINGS: {system_prompt}")

    turn_count = state.get("turn_count", 0)

    structured_llm = llm.with_structured_output(
        EvaluationOutput, method="json_schema")

    # Use the trimmed messages in the maturity evaluation
    maturity_message = maturity_prompt.format(
        conversation_summary=state.get("summary", ""),
        conversation_history="\n".join(
            [msg.content for msg in trimmed_messages])
    )

    eval_output = await structured_llm.ainvoke(maturity_message)

    if hasattr(eval_output, "model_dump_json"):
        logger.info(f"EVALUATION: {eval_output.model_dump_json()}")
    else:
        logger.info(f"EVALUATION: {eval_output}")

    if hasattr(eval_output, "model_dump"):
        eval_output = eval_output.model_dump()
    else:
        # Fallback for dict output from HuggingFace models
        pass

    if not isinstance(eval_output, dict):
        eval_output = {
            "summary": "No summary generated.",
            "info_score": 0.0,
        }

    # Use the summary from evaluation for clarification prompt
    clarification_message = clarification_prompt.format(
        conversation_history=eval_output.get("summary"))

    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=clarification_message)
    ai_msg = llm.invoke([system] + trimmed_messages + [human])

    messages_to_keep = 6
    all_messages = state["messages"]

    prune_updates = []
    if len(all_messages) > messages_to_keep:
        # Identify the IDs of the messages that exceed our window
        # and create RemoveMessage commands for them
        to_remove = all_messages[:-messages_to_keep]
        prune_updates = [RemoveMessage(id=m.id) for m in to_remove if m.id]

    return {
        **state,
        "messages": [ai_msg] + prune_updates,
        "info_score": float(eval_output.get("info_score")),
        "turn_count": turn_count + 1,
        "status": "conversation",
        "summary": eval_output.get("summary"),
    }
