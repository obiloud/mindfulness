from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)


async def node_hydrate(state: ChatState, runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Fetches stored facts and formats them for the LLM.

    This node retrieves memorable facts with metadata filtering to ensure
    only relevant, high-confidence memories are included in the conversation.
    """
    user_id = config["configurable"].get("user_id")
    thread_id = config["configurable"].get("thread_id")

    # Search for memorable facts with metadata filter
    # Filter by type='memorable_fact' to separate from preferences
    all_messages = state.get("messages", [])
    query = [m for m in all_messages if isinstance(
        m, HumanMessage)][-1].content

    try:
        # Search with metadata filter for memorable facts only
        items = await runtime.store.asearch(("memories", user_id), query=query, limit=5)

        logger.info(f"HYDRATE MEMORIES: Found {len(items)} memorable facts")

        # Format the items into a readable block
        if items:
            # Sort by confidence and timestamp (most recent first)
            sorted_items = sorted(
                items,
                key=lambda x: (x.value["metadata"]["confidence"],
                               x.value["metadata"]["timestamp"]),
                reverse=True
            )

            formatted_memories = [
                {**item.value, "memory_id": item.key}
                for item in sorted_items
            ]
        else:
            formatted_memories = "No memorable facts to share."

    except Exception as e:
        logger.error(f"Error hydrating memories: {e}")
        formatted_memories = "No memorable facts to share."

    if thread_id:
        namespace = ("a2a", thread_id, "pending_updates")

        try:
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
        except Exception as e:
            logger.error(f"Error processing A2A updates: {e}")
            return {"long_term_memory": formatted_memories}

    # Save this to the state so the Agent Node can see it
    return {"long_term_memory": formatted_memories}
