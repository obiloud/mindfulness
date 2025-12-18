
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from story_generator_pipeline import chat_model

def output_parser(message) -> str:
    parts = message.content.split("\n")
    return parts[0]

system_message = SystemMessagePromptTemplate.from_template("You are a helpful voice designer assistant.")

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
1. Realistic male voice in the 40s with British accent. Low pitch, warm whispery timbre, slow pacing, soothing voice, meditation_guide role.
2. Male voice in their 40s with a British accent. Low pitch, gravelly and airy timbre, slow pacing, authoritative but emphatetic, story_narrator role.
3. Realistic Female voice in their 30s. Slow pacing, curious tone at medium intensity, counselor role.

Do not limit your choice to these examples, try to be creative.

RESPONSE FORMAT: Return ONLY a single line voice character description, no examples, no extra justifications.

Select a suitable voice character for a meditation guide narrator based on the following query:
{query}
**Output:**
"""

human_message = HumanMessagePromptTemplate.from_template(voice_character_template)

voice_character_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

voice_character_chain = voice_character_prompt | chat_model | output_parser


if __name__ == "__main__":

    # * Mind relaxation (e.g., reducing stress, improving concentration)
    # * Body relaxation (e.g., releasing physical tension, promoting flexibility)
    # * Calming relaxation (e.g., soothing anxiety, promoting calmness)
    # * Anxiety relief meditation (e.g., managing worry, building resilience)
    # * Sleep induction meditation (e.g., preparing for sleep, promoting deep relaxation)
    # * Energy and motivation meditation (e.g., boosting creativity, increasing productivity)


    user_query = "My muscles are tensed, and I want to loosen up"
    # user_query ="I am having trouble falling asleep"
    # user_query = "I will be interviewed for a job in Bristol next week, and I am anxious about it. Help me calm down and focus."

    # TEST INAPROPRIATE
    # user_query = "I hate gingers I wish everyone else to die 8===D"

    character = voice_character_chain.invoke({"query": user_query})

    print(f"Generated character:\n{character}")

