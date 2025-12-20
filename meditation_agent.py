import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.messages import ToolMessage
from story_generator_pipeline import meditation_guide_generator_chain
from voice_generator_pipeline import voice_character_chain
from tts_client import AudioStreamer
import gradio as gr
import json
import threading

load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

def get_llm(): 
    repo_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.5,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        provider="auto"
    )

    return llm

# Tools

@tool
def generate_audio_guided_meditation_session(context: str) -> str:
    """ Generates the guided meditation session tailored to user's specific context and returns the transcript

        Args:
            condition (str): User's condition

        Returns:
            session transcript
    """

    def play_audio(voice_character, transcript):
        streamer = AudioStreamer(voice_character)
        streamer.start(transcript)


    pipeline = RunnableParallel(description=voice_character_chain, text=meditation_guide_generator_chain)
    result = pipeline.invoke({"query": context})

    print(json.dumps(result))

    voice_character = result.get("description")
    transcript = result.get("text", "")

    p = threading.Thread(target=play_audio, args=(voice_character, transcript, ))

    p.start()
    

    return str(transcript)


class MindfulnessAgent:
    def __init__(self, llm, history = [], tools=[], tool_mapping={}):

        self.history = history
        self.tool_mapping = tool_mapping

        chat_model = ChatHuggingFace(llm=llm)

        self.chat_model_with_tools = chat_model.bind_tools(tools)

        system_template = """**You are an expert mindfulness coach, helping individuals resolve their emotional or mental struggles through guided meditations and supportive conversations.**

**Your role is to provide guidance, ask follow-up questions, and offer targeted recommendations using the provided tools and schema.**

**Instructions:**

1. **Initial Assessment:** When prompted with a condition (e.g., anxiety, stress, insomnia), ask follow-up questions to clarify and assess the individual's situation, such as:
	* Can you describe what triggers or exacerbates your [condition]?
	* How long have you been experiencing symptoms, and how severe are they?
	* Have you tried any previous methods for managing [condition], and if so, with what success?
2. **Diagnosis and Guidance:** Based on the individual's responses, use the provided tools and schema to generate a personalized guided meditation script tailored to their specific needs.
3. **Tool Utilization:**
	* Use the "Mindfulness Exercises" tool to select relevant exercises (e.g., body scan, loving-kindness meditation) that address the individual's condition.
	* Incorporate the "Emotional Awareness" tool to help the individual recognize and acknowledge their emotions related to the condition.
	* Employ the "Breathwork" tool to guide the individual in using specific breathing techniques for relaxation and calmness.
4. **Guided Meditation Generation:** Use the schema to create a clear, concise, and engaging guided meditation script that:
	* Begins with a gentle introduction and explanation of the chosen exercises
	* Gradually builds into more immersive and targeted mindfulness practices
	* Concludes with calming affirmations or visualizations for lasting relaxation
5. **Response and Follow-up:** Provide a thoughtful response to the individual, summarizing their condition, outlining the guided meditation script, and offering additional recommendations (e.g., journaling, yoga) for further support.

**What not to do:**

* Never provide diagnoses or medical advice; instead, focus on offering guidance and support
* Refrain from using overly complex or technical language that might confuse the individual
* Avoid making assumptions about the individual's situation or circumstances

Remember to follow these instructions carefully and use the provided tools and schema to create a personalized guided meditation experience for the individual.

{tools_schema}
"""

        system_message = SystemMessagePromptTemplate.from_template(system_template)

        chat_history = MessagesPlaceholder(variable_name="history")

        user_message = HumanMessagePromptTemplate.from_template("Context for a guided audio-session: {context}")

        self.chat_prompt = ChatPromptTemplate.from_messages([system_message, chat_history, user_message])

    def execute_tool(self, tool_call):
        """Execute single tool call and return ToolMessage"""
        try:
            result = self.tool_mapping[tool_call["name"]].invoke(tool_call["args"])
            content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            content = f"Error: {str(e)}"
        
        return ToolMessage(
            content=content,
            tool_call_id=tool_call["id"]
        )

    def process_tool_calls(self, messages):
        """Recursive tool call processor"""
        last_message = messages[-1]
        
        # Execute all tool calls in parallel
        tool_messages = [
            self.execute_tool(tc) 
            for tc in getattr(last_message, 'tool_calls', [])
        ]
        
        # Add tool responses to message history
        updated_messages = messages + tool_messages
        
        # Get next LLM response
        next_ai_response = self.chat_model_with_tools.invoke(updated_messages)
        
        return updated_messages + [next_ai_response]
    
    def should_continue(self, messages):
        """Check if you need another iteration"""
        last_message = messages[-1]
        return bool(getattr(last_message, 'tool_calls', None))
    
    def _recursive_chain(self, messages):
        """Recursively process tool calls until completion"""
        if self.should_continue(messages):
            new_messages = self.process_tool_calls(messages)
            return self._recursive_chain(new_messages)
        return messages

    
    def run(self, query):
        for tool in self.tool_mapping.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema.model_json_schema() if tool.args_schema else {}
            }
        
        recursive_chain = RunnableLambda(self._recursive_chain)
        universal_chain = (
            RunnableLambda(lambda x: self.chat_prompt.invoke({
                "tools_schema": schema,
                "history": self.history,
                "context": x["query"]}
            ).to_messages())
            | RunnableLambda(lambda messages: messages + [self.chat_model_with_tools.invoke(messages)])
            | recursive_chain
        )

        try:
            response = universal_chain.invoke({"query": query})
            return response[-1].content
        except Exception as e:
            return f"Error: {e}"


def mindfulness_agent_fn(query, history):
    agent = MindfulnessAgent(llm=get_llm(), history = history, tools=[generate_audio_guided_meditation_session], tool_mapping={"generate_audio_guided_meditation_session": generate_audio_guided_meditation_session})

    response = agent.run(query)

    return response


demo = gr.ChatInterface(
    fn=mindfulness_agent_fn,
    title="Mindfulness App",
    description="Meditation sessions tailored just for you"
)

demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
