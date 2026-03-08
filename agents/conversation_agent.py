from state import ConversationState, GraphContext, print_state
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from prompts.conversation import CONVERSATION_PROMPT
import inspect

def node_clarification(state: ConversationState, runtime: Runtime[GraphContext]) -> ConversationState:
    """
    Ask a short, targeted follow-up question to gather more context.

    The frontend is expected to include the resulting messages in the
    next request's history so the agent can continue the conversation.
    """
    logger = runtime.context.logger
    llm = runtime.context.llm

    logger.info(f"Clarification triggered: state='{print_state(state)}'")

    messages = state["messages"]

    # system_prompt = inspect.cleandoc("""
    #     You are the Clarification Node for a Mindfulness Coach. Your sole task is to gather necessary context before a meditation is generated.

    #     When a user mentions a condition (e.g., anxiety, stress), ask targeted follow-up questions:

    #         What specifically triggers or exacerbates this feeling?

    #         How long have you been experiencing this, and what is the current severity?

    #         What have you tried in the past to manage this?

    #     Constraint: Do not provide exercises yet. Only ask the questions needed to move to the next phase.
    # """)

    clarification_prompt = inspect.cleandoc("""
        Before I guide you with a meditation, I need a bit more context.
        Ask the user one short, compassionate follow-up question that helps you understand what they are experiencing or what they hope to get from this session.
        Respond only with that single question.
    """)

    system_prompt = CONVERSATION_PROMPT.format(
        memories=state.get("long_term_memory", "No history available."),
        summary=state.get("summary", "New conversation.")
    )


    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=clarification_prompt)
    ai_msg = llm.invoke([system] + messages + [human])
    messages = messages + [ai_msg]

    return {
        **state,
        "messages": messages,
        "clarification_count": state.get("clarification_count", 0) + 1,
        "status": "clarifying",
    }