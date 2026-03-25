from ..state import ChatState, GraphContext
from langchain.messages import RemoveMessage
from langgraph.runtime import Runtime


def summarization_node(state: ChatState, runtime: Runtime[GraphContext]) -> ChatState:
    """
    Triggers when the message count is too high.
    """
    llm = runtime.context.llm
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # Check if we've crossed the 'annoyance threshold' (e.g., 10 messages)
    if len(messages) > 10:
        # Ask the LLM to merge the old summary + new messages
        # We keep the last 2 messages for immediate context, summarize the rest.
        to_summarize = messages[:-2]
        new_summary = llm.invoke(
            f"Summarize this: {existing_summary} + {to_summarize}")

        # Return the new summary and a command to DELETE the old messages
        # In LangGraph, returning a RemoveMessage() or slicing the list
        # clears the state for those specific IDs.
        return {
            "summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in to_summarize]
        }

    return {}
