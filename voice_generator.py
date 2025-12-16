import os
from dotenv import load_dotenv
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

repo_id = "meta-llama/Meta-Llama-3-8B"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=2000,
    temperature=0.7,
    top_k=50,
    top_p=0.8,
    repetition_penalty=1.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    provider="auto"
)

def output_parser(message: str) -> str:
    cleaned = message.strip()
    cleaned = cleaned.strip('`')
    cleaned = cleaned.strip('"')
    parts = cleaned.split("**Output:** ")
    return parts[-1]

voice_character_template = """**You are an experienced radio director**

**Your role is to select a suitable voice character for a meditation guide narrator based on the user's query.**

**Instructions:**

1. **Analyze the user's query:** Carefully read and understand the user's input, which may describe their emotional state or specific needs.
2. **Select an appropriate voice character:** Based on the user's query, choose a voice character that would be most effective in guiding them through a meditation session. This might include considering factors such as:
	* Emotion: Should the narrator convey calmness, energy, or empathy?
    * Timbre: Gives a character a unique personality that can touch more deeply and personally (warm, smooth, whispery, airy, rich, soulful, brassy, smokey etc.). 
	* Tone: Should the tone be soothing, uplifting, or challenging?
	* Intensity: Should the intensity of the voice match the user's emotional state (e.g., anxious, relaxed)?
    * Pitch: High, Medium, Low
    * Pace: A voice must be spoken at an appropriate pace to guide a listener to a meditative state. 
3. **Choose a voice that complements the query:** Select a voice character that would resonate with the user and help them achieve their desired outcome.
4. **Avoid jarring or conflicting voices:** Refrain from selecting a voice that might be distracting or counterproductive to the user's needs.
5. **Remember the role of a character:** The voice is ultimately a meditation guide, generally needs to calm, comfort, and soothe a listener.

Examples:
 - Realistic male voice in the 40s with British accent. Low pitch, warm whispery timbre, slow pacing, soothing voice.
 - A confident boat skipper character, Male voice in their 40s with a British accent. Low pitch, gravelly and airy timbre, slow pacing, authoritative but emphatetic.
 - Mythical godlike magical character, Female voice in their 30s slow pacing, curious tone at medium intensity.

Do not limit your choice to these examples, try to be creative.

RESPONSE FORMAT: Return ONLY a single line voice character description, no examples, no extra justifications.

Select a suitable voice character for a meditation guide narrator based on the following query:
{query}
**Output:**
"""


voice_character_prompt = PromptTemplate.from_template(voice_character_template)

voice_character_chain = voice_character_prompt | llm 


if __name__ == "__main__":

    # * Mind relaxation (e.g., reducing stress, improving concentration)
    # * Body relaxation (e.g., releasing physical tension, promoting flexibility)
    # * Calming relaxation (e.g., soothing anxiety, promoting calmness)
    # * Anxiety relief meditation (e.g., managing worry, building resilience)
    # * Sleep induction meditation (e.g., preparing for sleep, promoting deep relaxation)
    # * Energy and motivation meditation (e.g., boosting creativity, increasing productivity)


    # user_query = "My muscles are tensed, and I want to loosen up"
    # user_query ="I am having trouble falling asleep"
    user_query = "I will be interviewed for a job in Bristol next week, and I am anxious about it. Help me calm down and focus."

    # TEST INAPROPRIATE
    # user_query = "I hate gingers I wish everyone else to die 8===D"

    character = voice_character_chain.invoke({"query": user_query})

    print(f"Generated character:\n{character}")

