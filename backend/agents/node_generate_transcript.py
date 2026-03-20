from langchain_core.messages import HumanMessage, SystemMessage
from state import ConversationState, GraphContext, print_state
from langgraph.runtime import Runtime
from prompts.meditation import TRANSCRIPT_PROMPT


async def node_generate_transcript(state: ConversationState, runtime: Runtime[GraphContext]) -> dict:
    llm = runtime.context.llm
    runtime.context.logger.info(f"TRANSCRIPT: {print_state(state)}")

    # Check if we are refining based on feedback
    feedback = state.get("transcript_feedback", "")
    messages = state.get("messages", [])

    context_text = ""

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if human_messages:
        context_text = "\n\n".join(m.content for m in human_messages)

    content = f"Context: {context_text}\nFeedback: {feedback}" if feedback else context_text

    system = SystemMessage(content=TRANSCRIPT_PROMPT)
    human = HumanMessage(content=content)

    # Async invocation
    response = await llm.ainvoke([system, human])

    return {
        "transcript": response.content,
        "is_transcript_valid": False  # Reset for re-validation
    }
