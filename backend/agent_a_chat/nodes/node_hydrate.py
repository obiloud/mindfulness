from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
import logging

logger = logging.getLogger(__name__)


async def node_hydrate(state: ChatState,  runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Fetches stored facts and formats them for the LLM.
    """
    user_id = config["configurable"].get("user_id")
    thread_id = config["configurable"].get("thread_id")
    namespace = ("memories", user_id)

    # Search the store for all memories in this user's namespace
    # You can search by semantic similarity or just get all keys
    items = await runtime.store.asearch(namespace, query=state.get("summary"), limit=5)

    logger.info(f"HYDRATE ITEMS: {items}")

    # Format the items into a readable block
    if items:
        memory_strings = [f"- {item.key}: {item.value}" for item in items]
        formatted_memories = "\n".join(memory_strings)
    else:
        formatted_memories = "No previous preferences known."

    if thread_id:
        namespace = ("a2a", thread_id, "pending_updates")

        items = await runtime.store.asearch(namespace)

        if items:
            updates = {}
            for item in items:
                updates.update(item.value)
                await runtime.store.adelete(namespace, item.key)

            logger.info(f"HYDRATE A2A UPDATES: {updates}")
            return {
                **updates,
                "long_term_memory": formatted_memories
            }

    # Save this to the state so the Agent Node can see it
    return {"long_term_memory": formatted_memories}
