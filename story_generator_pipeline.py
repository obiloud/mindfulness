import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.6,
    top_k=80,
    top_p=0.9,
    repetition_penalty=1.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    provider="auto"
)

chat_model = ChatHuggingFace(llm=llm)

system_message = SystemMessagePromptTemplate.from_template("You are a helpfull creative writing assistant.")

def output_parser(message) -> str:
    cleaned = message.content.strip()
    cleaned = cleaned.strip('`')
    cleaned = cleaned.strip('"')
    parts = cleaned.split("**Output:**")
    return parts[-1]

template = """You are an expert meditation guru, guiding individuals through various types of meditation sessions.

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
	* Include breathing exercises (e.g., diaphragmatic breathing, 4-7-8 breathing) tailored to the user's specific needs
	* Suggest physical relaxations such as progressive muscle relaxation, yoga-inspired postures or gentle stretches
4. **Imagery and visualisation:**
	* Use vivid, descriptive language to paint a peaceful picture for the listener's imagination
5. **Emotions:** for expressive and impactful storytelling include emotion tags from the following list: <angry>, <appalled>, <chuckle>, <cry>, <curious>, <disappointed>, <excited>, <exhale>, <gasp>, <giggle>, <gulp>, <laugh>, <laugh_harder>, <mischievous>, <sarcastic>, <scream>, <sigh>, <sing>, <snort>, <whisper>.      

IMPORTANT: Keep sentence length shorter than 30 words for smooth streaming.
IMPORTANT: Use emotion tags for stressing certain words or to include emotions: <angry>, <appalled>, <chuckle>, <cry>, <curious>, <disappointed>, <excited>, <exhale>, <gasp>, <giggle>, <gulp>, <laugh>, <laugh_harder>, <mischievous>, <sarcastic>, <scream>, <sigh>, <sing>, <snort>, <whisper>
IMPORTANT: Use [PAUSE:1.0] tags to instruct the narrator to make a pause in speech of arbitrary duration in seconds
IMPORTANT: Separate sentences with newline characters.
IMPORTANT: Do not include quotes or backticks around the generated text. 
IMPORTANT: No section titles.
IMPORTANT: No markdown.
IMPORTANT: No html.
IMPORTANT: No indentation.
IMPORTANT: No special characters.
IMPORTANT: No emojis.
IMPORTANT: No examples.
IMPORTANT: No additional notes.

**User Input (Query):**

{query}

**Output:**"""

human_message = HumanMessagePromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# story_generator_chain = prompt | llm | output_parser
story_generator_chain= prompt | chat_model | output_parser


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


    story = story_generator_chain.invoke({"query": user_query})

    print("Generated Story:\n", story)