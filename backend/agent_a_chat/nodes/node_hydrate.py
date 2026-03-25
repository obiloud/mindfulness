from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime


async def node_hydrate(state: ChatState,  runtime: Runtime[GraphContext]) -> ChatState:
    """
    Fetches stored facts and formats them for the LLM.
    """
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)

    # Search the store for all memories in this user's namespace
    # You can search by semantic similarity or just get all keys
    items = await runtime.store.asearch(namespace, query=state["messages"][-1].content, limit=5)

    # Format the items into a readable block
    if items:
        memory_strings = [f"- {item.key}: {item.value}" for item in items]
        formatted_memories = "\n".join(memory_strings)
    else:
        formatted_memories = "No previous preferences known."

    # Save this to the state so the Agent Node can see it
    return {"long_term_memory": formatted_memories}
