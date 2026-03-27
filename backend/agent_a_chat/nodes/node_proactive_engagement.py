import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from ..state import ChatState, GraphContext, print_state
from ..prompts.conversation import CONVERSATION_PROMPT

logger = logging.getLogger(__name__)


def node_proactive_engagement(state: ChatState, runtime: Runtime[GraphContext]) -> ChatState:
    """
    Ask a short, targeted follow-up question to gather more context.

    The frontend is expected to include the resulting messages in the
    next request's history so the agent can continue the conversation.
    """
    llm = runtime.context.llm

    logger.info(f"Proactive continued: state='{print_state(state)}'")

    messages = state["messages"]

    system_prompt = CONVERSATION_PROMPT.format(
        memories=state.get("long_term_memory", "No history available."),
        summary=state.get("summary", "New conversation.")
    )
    prompt = "The transcript is ready. Invite the user to start the session naturally."

    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=prompt)
    ai_msg = llm.invoke([system] + messages + [human])

    return {
        "messages": [ai_msg],
        "awaiting_confirmation": True
    }
