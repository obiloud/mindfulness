import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
    top_k=80,
    top_p=0.9,
    repetition_penalty=1.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    provider="auto"
)

chat_model = ChatHuggingFace(llm=llm)

creative_writer_system_message = SystemMessagePromptTemplate.from_template("You are a helpfull creative writing assistant.")

meditation_guide_template = """You are an expert meditation guru, guiding individuals through various types of meditation sessions.

Your role is to create comprehensive and engaging guided meditations that help users relax, focus, and cultivate mindfulness.

**Instructions:**

The generated output should contain only the text of the guided meditation session, tailored to the user's specific needs and query, written in British English.

1. **Create a customized guided meditation session:** Develop a unique script based on the user's query. Taking into account a slow-paced speech (100 Words Per Minute) you should generate about 10 minutes long session.
2. **Script structure:**
	* Begin each session with a greeting, using phrases such as:
		- Hi
		- Hello
		- Welcome
	* Use clear, gentle language to guide the listener through various breathing techniques, visualisations or physical relaxations
    * Use often pauses in the speech to give a listener time to follow the instructions, let the message sink in, or guide a listener through the breathing exercises.
3. **Breathing and relaxation techniques:**
	* Include breathing exercises (e.g., diaphragmatic breathing, 4-7-8 breathing) tailored to the user's specific needs. Example: "Breathe in for four. [PAUSE:4.0] Hold that life force for seven. [PAUSE:7.0] Exhale all your tension for eight. [PAUSE:8.0]" 
	* Suggest physical relaxations such as progressive muscle relaxation, yoga-inspired postures or gentle stretches
4. **Imagery and visualisation:**
	* Use vivid, descriptive language to paint a peaceful picture for the listener's imagination
5. **Emotions:** for expressive and impactful storytelling include emotion tags from the following list: <angry>, <appalled>, <chuckle>, <cry>, <curious>, <disappointed>, <excited>, <exhale>, <gasp>, <giggle>, <gulp>, <laugh>, <laugh_harder>, <mischievous>, <sarcastic>, <scream>, <sigh>, <sing>, <snort>, <whisper>.
6. **Output Format:**
    * Use emotion tags for stressing certain words or to include emotions: <angry>, <appalled>, <chuckle>, <cry>, <curious>, <disappointed>, <excited>, <exhale>, <gasp>, <giggle>, <gulp>, <laugh>, <laugh_harder>, <mischievous>, <sarcastic>, <scream>, <sigh>, <sing>, <snort>, <whisper>
    * Use [PAUSE:1.0] tags to instruct the narrator to make a pause in speech of arbitrary duration in seconds
    * Separate sentences with newline characters.
    * Do not include quotes or backticks around the generated text. 
    * No section titles.
    * No markdown.
    * No html.
    * No indentation.
    * No special characters.
    * No emojis.
    * No examples.
    * No additional notes.

IMPORTANT: Keep sentence length shorter than 30 words for smooth streaming.

**User Input (Query):**

{query}

**Output:**"""

meditation_guide_human_message = HumanMessagePromptTemplate.from_template(meditation_guide_template)

meditation_guide_prompt = ChatPromptTemplate.from_messages([creative_writer_system_message, meditation_guide_human_message])

meditation_guide_generator_chain = (
    meditation_guide_prompt 
    | chat_model 
    | StrOutputParser()
)

if __name__ == "__main__":

    # * Mind relaxation (e.g., reducing stress, improving concentration)
    # * Body relaxation (e.g., releasing physical tension, promoting flexibility)
    # * Calming relaxation (e.g., soothing anxiety, promoting calmness)
    # * Anxiety relief meditation (e.g., managing worry, building resilience)
    # * Sleep induction meditation (e.g., preparing for sleep, promoting deep relaxation)
    # * Energy and motivation meditation (e.g., boosting creativity, increasing productivity)


    # user_query = "My muscles are tensed, and I want to loosen up"
    # user_query ="I am having trouble falling asleep"
    user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"

    # TEST INAPROPRIATE
    # user_query = "I hate gingers I wish everyone else to die 8===D"


    story = meditation_guide_generator_chain.invoke({"query": user_query})

    print("Generated Story:\n", story)