from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from shared.datamodels.preferences import (
    MindfulnessProfile,
    VoiceBlueprint,
    update_voice_blueprint,
    update_mindfulness_profile,
)
from .node_extract_memories import extract_memorable_facts, store_memorable_facts
import logging

logger = logging.getLogger(__name__)


async def node_manage_memory(state: ChatState, runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Logic for saving facts cross-session.

    This node handles both:
    1. Voice/profile preferences (existing functionality)
    2. Memorable facts extraction and storage (new functionality)
    """
    llm = runtime.context.llm

    user_id = config["configurable"].get("user_id")
    namespace = ("preferences", user_id)

    summary = state.get("summary", "")

    voice_blueprint_item = await runtime.store.aget(namespace, "voice_blueprint")
    existing_blueprint = voice_blueprint_item.value if voice_blueprint_item else None

    minfulness_profile_item = await runtime.store.aget(namespace, "mindfulness_profile")
    existing_profile = minfulness_profile_item.value if minfulness_profile_item else None

    voice_blueprint = await update_voice_blueprint(llm, summary, existing_blueprint)

    mindfulness_profile = await update_mindfulness_profile(llm, summary, existing_profile)

    if isinstance(voice_blueprint, VoiceBlueprint):
        logger.info(f"Voice blueprint: {voice_blueprint}")
        if hasattr(voice_blueprint, "model_dump"):
            await runtime.store.aput(namespace, "voice_blueprint", voice_blueprint.model_dump())
        else:
            await runtime.store.aput(namespace, "voice_blueprint", voice_blueprint)

    if isinstance(mindfulness_profile, MindfulnessProfile):
        logger.info(f"Mindfulness profile: {mindfulness_profile}")
        if hasattr(mindfulness_profile, "model_dump"):
            await runtime.store.aput(namespace, "mindfulness_profile", mindfulness_profile.model_dump())
        else:
            await runtime.store.aput(namespace, "mindfulness_profile", mindfulness_profile)

    # Extract and store memorable facts
    try:
        # Extract memorable facts from conversation
        extraction_result = await extract_memorable_facts({
            "messages": state.get("messages", []),
            "summary": summary,
            "user_id": user_id
        }, runtime)

        facts = extraction_result.get("memorable_facts", [])

        if facts:
            logger.info(f"Extracted {len(facts)} memorable facts")

            # Store the facts with metadata filter
            await store_memorable_facts(facts, user_id, runtime)

            logger.info(
                f"Stored {len(facts)} memorable facts for user {user_id}")
        else:
            logger.info("No memorable facts extracted from this conversation")

    except Exception as e:
        logger.error(f"Error extracting/storing memorable facts: {e}")

    return {}
