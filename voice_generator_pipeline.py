import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=128,
    temperature=0.5,
    top_k=80,
    top_p=0.9,
    repetition_penalty=1.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    provider="auto"
)

class VoiceDesign(BaseModel):
    age: str = Field(description="Describing the perceived age of the voice helps define its maturity, vocal texture, and energy. Use specific terms to guide the AI toward the right vocal quality.",
                     examples=["Adolscent male", "adolescent female", "Young adult", "in their 20s", "early 30s", "Middle-aged man", "woman in her 40s", "Elderly man", "older woman", "man in his 80s"])
    gender: str = Field(description="Gender often typically influences pitch, vocal weight, and tonal presence — but you can push beyond simple categories by describing the sound instead of the identity.",
                        examples=["A lower-pitched, husky female voice", "A masculine male voice, deep and resonant", "A neutral gender — soft and mid-pitched"])
    accent: str = Field(description="Accent plays a critical role in defining a voice’s regional, cultural, and emotional identity.", 
                        examples=["A middle-aged man with a thick French accent", "A young woman with a slight Southern drawl", "A cheerful woman speaking with a crisp British accent", "A younger male with a soft Irish lilt"])
    timbre: str = Field(description="Refers to the physical quality of the voice, shaped by pitch, resonance, and vocal texture. It’s distinct from emotional delivery or attitude.", 
                        examples=["Deep", "low-pitched", 
                                  "Smooth", "rich", 
                                  "Nasally", "shrill", 
                                  "Airy", "breathy", 
                                  "Booming", "resonant", 
                                  "Light", "thin", 
                                  "Warm", "mellow", 
                                  "Tinny", "metalic"])
    pacing: str = Field(description="Pacing refers to the speed and rhythm at which a voice speaks.",
                        examples=["Speaking quickly", "at a fast pace", 
                                  "At a normal pace", "speaking normally", 
                                  "Speaking slowly", "with a slow rhythm", 
                                  "Deliberate and measured pacing",
                                  "Drawn out, as if savoring each word",
                                  "With a hurried cadence, like they’re in a rush",
                                  "Relaxed and conversational pacing",
                                  "Rhythmic and musical in pace",
                                  "Erratic pacing, with abrupt pauses and bursts",
                                  "Even pacing, with consistent timing between words",
                                  "Staccato delivery"])
    
parser = PydanticOutputParser(pydantic_object=VoiceDesign)

model = ChatHuggingFace(llm=llm)

voice_character_system_template = voice_character_template = """Here is a detailed and extensive prompt for the model:

**You are an experienced radio director**

**Your role is to select a suitable voice character for a meditation guide narrator based on the user's query.**

**Instructions:**

1. **Analyze the user's query:** Carefully read and understand the user's input, which may describe their emotional state or specific needs.
2. **Select an appropriate voice character:** Based on the user's query, choose a voice character that would be most effective in guiding them through a meditation session. This might include considering factors such as:
	* Emotion: Should the narrator convey calmness, energy, or empathy?
	* Timbre: What unique personality trait can the narrator convey (warm, smooth, whispery, airy, rich, soulful, brassy, smokey etc.)?
	* Tone: Should the tone be soothing, uplifting, or challenging?
	* Intensity: Should the intensity of the voice match the user's emotional state (e.g., anxious, relaxed)?
	* Pitch: What pitch range is most suitable for the user's needs (high, medium, low)?
	* Pace: How fast or slow should the narrator speak to guide the listener into a meditative state?
3. **Choose a voice that complements the query:** Select a voice character that would resonate with the user and help them achieve their desired outcome.
4. **Avoid jarring or conflicting voices:** Refrain from selecting a voice that might be distracting or counterproductive to the user's needs.
5. **Remember the role of a character:** The voice is ultimately a meditation guide, generally needs to calm, comfort, and soothe a listener.

{format_instructions}
"""

voice_character_system_message = SystemMessagePromptTemplate.from_template(voice_character_system_template, partial_variables={"format_instructions": parser.get_format_instructions()})

voice_character_template = """Select a suitable voice character for a meditation guide narrator based on the following query: {query}"""

voice_character_human_message = HumanMessagePromptTemplate.from_template(voice_character_template)

voice_character_prompt = ChatPromptTemplate.from_messages([voice_character_system_message, voice_character_human_message])

def to_string(voice_design):
    return f"Realistic sounding voice. {voice_design.age}. {voice_design.gender}. {voice_design.accent}. {voice_design.timbre}. {voice_design.pacing}."

voice_character_chain = (
    voice_character_prompt 
    | model
    | parser
    | to_string
)


if __name__ == "__main__":

    # * Mind relaxation (e.g., reducing stress, improving concentration)
    # * Body relaxation (e.g., releasing physical tension, promoting flexibility)
    # * Calming relaxation (e.g., soothing anxiety, promoting calmness)
    # * Anxiety relief meditation (e.g., managing worry, building resilience)
    # * Sleep induction meditation (e.g., preparing for sleep, promoting deep relaxation)
    # * Energy and motivation meditation (e.g., boosting creativity, increasing productivity)


    # user_query = "My muscles are tensed, and I want to loosen up with a guided session spoken by the male voice"
    user_query ="I am having trouble falling asleep"
    # user_query = "I will be interviewed for a job in Bristol next week, and I am anxious about it. Help me calm down and focus."

    # TEST INAPROPRIATE
    # user_query = "I hate gingers I wish everyone else to die 8===D"

    character = voice_character_chain.invoke({"query": user_query})

    print(f"Generated character:\n{character}")

