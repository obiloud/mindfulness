import inspect
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.language_models.chat_models import BaseChatModel

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
    current_state_str = "{}"
    if existing_blueprint:
        if isinstance(existing_blueprint, dict):
            clean_data = {k: v for k,
                          v in existing_blueprint.items() if v is not None}
            existing_blueprint = VoiceBlueprint(**clean_data)

        try:
            current_state_str = existing_blueprint.model_dump_json(indent=2)
        except Exception as e:
            print(f"EXISTING BLUEPRINT {existing_blueprint}\n{e}")

    allowed_gender = [g.value for g in VoiceGender]
    allowed_age = [a.value for a in VoiceAge]
    allowed_textures = [t.value for t in VoiceTexture]

    voice_blueprint_prompt = inspect.cleandoc("""
        You are a state-management assistant for an AI voice system.
        CURRENT VOICE PREFERENCES:
        ```
        {current_state_str}
        ```
                                              
        USER INPUT: "{message}"

        TASK:
        1. Analyze the USER INPUT for any changes to voice gender, age, or texture.
        2. If the user specifies a new attribute (e.g., 'more breathy'), UPDATE the current preferences.
        3. If the user contradicts a current preference, REPLACE it.
        4. Keep all other existing preferences UNCHANGED if they are not mentioned.
        5. Return the full, updated VoiceBlueprint object.

        IMPORTANT: 
        For 'geneder', you MUST ONLY use one of these exact values: {allowed_gender}.
        For 'age', you MUST ONLY useone of these exact values: {allowed_age}.
        For 'textures', you MUST ONLY use some of these exact values: {allowed_textures}.
        "Do not use synonyms like 'soothing' or 'calming'. Map them to the closest allowed value.

        ### OUTPUT FORMAT:
        You must return a valid JSON object ONLY. Do not include any preamble.
        {{
            "gender": str,
            "age": int,
            "textures": list[str]
        }}
        """).strip()

    voice_blueprint_message = voice_blueprint_prompt.format(
        message=message,
        current_state_str=current_state_str,
        allowed_gender=allowed_gender,
        allowed_age=allowed_age,
        allowed_textures=allowed_textures
    )

    # We use the full VoiceBlueprint here, not the extraction wrapper,
    # because we want the LLM to always return a complete state.
    structured_llm = llm.with_structured_output(
        VoiceBlueprint, method="json_schema")

    # The LLM now returns the "New Truth"
    try:
        updated_blueprint = await structured_llm.ainvoke(voice_blueprint_message)
    except Exception as e:
        print(f"Failed to update voice blueprint: {e}")
        updated_blueprint = existing_blueprint

    return updated_blueprint

# FOR INSTRUCTING A CARTESIA MODEL
# Internal Voice Registry (Example)
VOICE_LIBRARY = [
    {
        "id": "63833296-3393-41c3-8884-2594612e5241",
        "name": "Sarah",
        "gender": "female", "age": "mature", "textures": ["soft", "warm"]
    },
    {
        "id": "79a125e3-fdcd-48b2-9214-7474727dcc9e",
        "name": "James",
        "gender": "male", "age": "elder", "textures": ["deep", "gravelly"]
    }
]


def resolve_voice_id(blueprint: VoiceBlueprint) -> str:
    """
    Selects the best voice_id based on the extracted blueprint.
    Fallback to a high-quality 'General' voice if no match is found.
    """
    for voice in VOICE_LIBRARY:
        gender_match = (blueprint.gender == VoiceGender.NO_PREFERENCE or
                        blueprint.gender.value == voice["gender"])
        age_match = blueprint.age.value == voice["age"]

        # Check if at least one texture matches
        texture_match = any(t.value in voice["textures"]
                            for t in blueprint.textures)

        if gender_match and age_match and (texture_match or not blueprint.textures):
            return voice["id"]

    return "DEFAULT_MEDITATION_VOICE_ID"  # Your safest, most neutral voice


# --- Enums for strict extraction ---

class ScriptMetaphor(str, Enum):
    NATURE = "nature"        # Rivers, clouds, mountains
    SPACE = "space"          # Stars, void, expansion
    BIOLOGICAL = "biological"  # Neurons, nervous system, muscles
    URBAN = "urban"          # City lights, structural, rhythm


class InstructionStyle(str, Enum):
    DIRECT = "direct"             # "Breathe in. Hold. Release."
    INVITATIONAL = "invitational"  # "I invite you to notice...", "If it feels right..."


class TechnicalDepth(str, Enum):
    LOW = "low"     # Purely experiential and guiding
    HIGH = "high"   # Includes neuroscience or psychological theory


class MindfulnessProfile(BaseModel):
    """Stored in AsyncPostgresStore for persistent personalization."""
    metaphor_preference: ScriptMetaphor = Field(default=ScriptMetaphor.NATURE)
    instruction_style: InstructionStyle = Field(
        default=InstructionStyle.INVITATIONAL)
    technical_depth: TechnicalDepth = Field(default=TechnicalDepth.LOW)
    favorite_anchors: List[str] = Field(
        default_factory=lambda: ["breath"],
        description="Physical focus points like 'breath', 'soles of feet', or 'ambient sound'."
    )


async def update_mindfulness_profile(
    llm: BaseChatModel,
    message: str,
    existing_profile: Optional[dict] = None
) -> MindfulnessProfile:
    """
    Extracts updates from the message and merges them with the existing profile.
    If no existing profile is provided, it populates a default one.
    """

    # Check if existing_profile is valid
    if existing_profile is not None:
        if isinstance(existing_profile, dict):
            clean_data = {k: v for k, v in existing_profile.items()
                          if v is not None}
            existing_profile = MindfulnessProfile(**clean_data)

    # Format the existing state for the LLM
    current_state_json = "{}"
    if existing_profile:
        current_state_json = existing_profile.model_dump_json(indent=2)

    allowed_metaphor = [m.value for m in ScriptMetaphor]
    allowed_style = [s.value for s in InstructionStyle]
    allowed_technical_depth = [t.value for t in TechnicalDepth]

    mindfulness_profile_prompt = inspect.cleandoc("""
        You are a profile management agent for Pulse Lotus, a mindfulness platform.
        Your goal is to maintain the 'LongTermMindfulnessProfile' based on user conversation.

        CURRENT PROFILE STATE:
        {current_state_json}
                                                  
        USER INPUT: "{message}"

        EXTRACTION RULES:
        1. **Metaphors**: Identify if they prefer nature, space, or biological imagery.
        2. **Style**: 'Direct' is authoritative/short. 'Invitational' is gentle/suggestive.
        3. **Technical Depth**: 'High' means they enjoy hearing about neuroscience or 'why' it works.
        4. **Anchors**: These are focus objects. If they mention a new one (e.g., 'I like focusing on my hands'), ADD it to the list. If they say 'I don't like focusing on my breath', REMOVE it.
        5. **Persistence**: If a field isn't mentioned in the message, KEEP the value from the CURRENT PROFILE STATE.
        
        IMPORTANT: 
        For 'metaphor_preference', you MUST ONLY use one of these exact values: {allowed_metaphor}.
        For 'instruction_style', you MUST ONLY useone of these exact values: {allowed_style}.
        For 'technical_depth', you MUST ONLY useone of these exact values: {allowed_technical_depth}.
        For 'favorite_anchors', use physical focus points like 'breath', 'soles of feet', or 'ambient sound'.
                                                  
        ### OUTPUT FORMAT:
        You must return a valid JSON object ONLY. Do not include any preamble.
        {{
            "metaphor_preference": str,
            "instruction_style": str,
            "technical_depth": str,
            "favorite_anchors": list[str]
        }}
    """).strip()

    mindfulness_profile_message = mindfulness_profile_prompt.format(
        current_state_json=current_state_json,
        message=message,
        allowed_metaphor=allowed_metaphor,
        allowed_style=allowed_style,
        allowed_technical_depth=allowed_technical_depth,
    )

    # Bind to the extraction wrapper
    structured_llm = llm.with_structured_output(
        MindfulnessProfile, method="json_schema")

    try:
        updated_profile = await structured_llm.ainvoke(mindfulness_profile_message)
    except Exception as e:
        print(f"Failed to update profile: {e}")
        updated_profile = existing_profile

    return updated_profile
