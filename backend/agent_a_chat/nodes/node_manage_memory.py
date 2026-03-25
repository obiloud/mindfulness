from typing import List, Optional
from enum import Enum
from typing import Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from ..state import ChatState, GraphContext
from langgraph.runtime import Runtime
from langgraph.config import RunnableConfig


async def node_manage_memory(state: ChatState, runtime: Runtime[GraphContext], config: RunnableConfig) -> ChatState:
    """
    Logic for saving facts cross-session.
    """
    llm = runtime.context.llm

    user_id = config["configurable"].get("user_id")
    namespace = ("preferences", user_id)

    last_msg = state["messages"][-1].content

    item = await runtime.store.aget(namespace, "voice_blueprint")
    existing_blueprint = item.value if item else None

    voice_blueprint = await update_voice_blueprint(llm, last_msg, existing_blueprint)

    if voice_blueprint:
        await runtime.store.aput(namespace, "voice_blueprint", voice_blueprint)

    return {}  # No change to message history needed


# ==========================================
# 1. Core Data Models
# ==========================================


class VoiceGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    NO_PREFERENCE = "no_preference"


class VoiceAge(str, Enum):
    YOUNG = "young"     # Energetic, bright
    MATURE = "mature"   # Grounded, authoritative
    ELDER = "elder"     # Wise, slow, textured


class VoiceTexture(str, Enum):
    SOFT = "soft"
    DEEP = "deep"
    BREATHY = "breathy"
    CRISP = "crisp"
    WARM = "warm"


class VoiceBlueprint(BaseModel):
    """Semantic description of the user's ideal guide voice."""
    gender: VoiceGender = Field(
        default=VoiceGender.NO_PREFERENCE,
        description="The preferred gender of the voice."
    )
    age: VoiceAge = Field(
        default=VoiceAge.MATURE,
        description="The perceived age or maturity of the voice."
    )
    textures: List[VoiceTexture] = Field(
        default_factory=list,
        description="Textural qualities like 'soft', 'warm', or 'deep' mentioned by the user."
    )

    @field_validator('textures', mode='before')
    @classmethod
    def filter_invalid_textures(cls, v):
        if not isinstance(v, list):
            return v
        # Only keep items that actually exist in our Enum
        allowed = {t.value for t in VoiceTexture}
        return [item for item in v if item in allowed]

# ==========================================
# 2. Extraction Wrapper
# ==========================================


class VoiceBlueprintExtraction(BaseModel):
    """Wrapper to safely determine if extraction was actually successful."""
    has_voice_preference: bool = Field(
        description="Set to True ONLY if the user explicitly mentions preferences related to voice gender, age, or tone/texture."
    )
    blueprint: Optional[VoiceBlueprint] = Field(
        default=None,
        description="The extracted voice blueprint. Leave null if has_voice_preference is False."
    )

# ==========================================
# 3. Extraction Function
# ==========================================


async def update_voice_blueprint(
    llm: BaseChatModel,
    message: str,
    existing_blueprint: Optional[dict] = None
) -> VoiceBlueprint:
    """
    Refines the voice blueprint by comparing the user's new message 
    with the existing stored preferences.
    """

    # Format the existing state for the LLM
    current_state_str = "None"
    if existing_blueprint:
        if isinstance(existing_blueprint, dict):
            clean_data = {k: v for k,
                          v in existing_blueprint.items() if v is not None}

            if "textures" in clean_data and isinstance(clean_data["textures"], list):
                allowed = {t.value for t in VoiceTexture}
                clean_data["textures"] = [
                    t for t in clean_data["textures"] if t in allowed]
            existing_blueprint = VoiceBlueprint(**clean_data)

        current_state_str = existing_blueprint.model_dump_json(indent=2)
        current_state_str = current_state_str.replace(
            "{", "{{").replace("}", "}}")

    allowed_textures = [t.value for t in VoiceTexture]

    system_prompt = (
        "You are a state-management assistant for an AI voice system.\n"
        f"CURRENT VOICE PREFERENCES:\n{current_state_str}\n\n"
        "USER INPUT: \"{message}\"\n\n"
        "TASK:\n"
        "1. Analyze the USER INPUT for any changes to voice gender, age, or texture.\n"
        "2. If the user specifies a new attribute (e.g., 'more breathy'), UPDATE the current preferences.\n"
        "3. If the user contradicts a current preference, REPLACE it.\n"
        "4. Keep all other existing preferences UNCHANGED if they are not mentioned.\n"
        "5. Return the full, updated VoiceBlueprint object."

        f"IMPORTANT: For 'textures', you MUST ONLY use these exact values: {allowed_textures}. "
        "Do not use synonyms like 'soothing' or 'calming'. Map them to the closest allowed value."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{message}")
    ])

    # We use the full VoiceBlueprint here, not the extraction wrapper,
    # because we want the LLM to always return a complete state.
    structured_llm = llm.with_structured_output(
        VoiceBlueprint, method="json_schema")

    # The LLM now returns the "New Truth"
    updated_blueprint = await (prompt | structured_llm).ainvoke({"message": message})
    return updated_blueprint
