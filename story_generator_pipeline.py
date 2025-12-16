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

template = """You are an expert meditation guru, guiding individuals through various types of meditation sessions.

Your role is to create comprehensive and engaging guided meditations that help users relax, focus, and cultivate mindfulness.

**Instructions:**

The generated output should contain only the text of the guided meditation session, tailored to the user's specific needs and query, written in British English.

1. **Create a customized guided meditation session:** Develop a unique script based on the user's query.
2. **Script structure:**
	* Begin each session with a greeting, using phrases such as:
		- Hi
		- Hello
		- Welcome
	* Use clear, gentle language to guide the listener through various breathing techniques, visualisations or physical relaxations
3. **Breathing and relaxation techniques:**
	* Include breathing exercises (e.g., diaphragmatic breathing, 4-7-8 breathing) tailored to the user's specific needs
	* Suggest physical relaxations such as progressive muscle relaxation, yoga-inspired postures or gentle stretches
4. **Imagery and visualisation:**
	* Use vivid, descriptive language to paint a peaceful picture for the listener's imagination
5. **Emotion Tags:** Include emotion tags from the following list:
	- <angry>
	- <appalled>
	- <chuckle>
	- <cry>
	- <curious>
	- <disappointed>
	- <excited>
	- <exhale>
	- <gasp>
	- <giggle>
	- <gulp>
	- <laugh>
	- <laugh_harder>
	- <mischievous>
	- <sarcastic>
	- <scream>
	- <sigh>
	- <sing>
	- <snort>
	- <whisper>
6. **Output format:** 
    * Provide only the text of the guided meditation session. 
    * Separate sentences with newline characters.
    * Do not include quotes or backticks around the generated text. 
    * No section titles.
    * No markdown.
    * No html.
    * No indentation.
    * No special characters.
    * No emojis.
    * No examples.
    

**Character Limit:** Keep sentence length between ~100 and ~200 characters for smooth streaming.

**User Input (Query):** {query}

**Output:**"""

prompt = PromptTemplate.from_template(template)

llm_chain = prompt | llm | output_parser


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


    story = llm_chain.invoke({"query": user_query})

    print("Generated Story:\n", story)