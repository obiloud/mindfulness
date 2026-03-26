from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig
from shared.datamodels.preferences import (
    MindfulnessProfile,
    VoiceBlueprint,
    update_voice_blueprint,
    update_mindfulness_profile,
)


async def node_manage_memory(state: ChatState, runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Logic for saving facts cross-session.
    """
    llm = runtime.context.llm
    logger = runtime.context.logger

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
        await runtime.store.aput(namespace, "voice_blueprint", voice_blueprint.model_dump())

    if isinstance(mindfulness_profile, MindfulnessProfile):
        logger.info(f"Mindfulness profile: {mindfulness_profile}")
        await runtime.store.aput(namespace, "mindfulness_profile", mindfulness_profile.model_dump())

    return {}  # No change to message history needed
