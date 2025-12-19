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

def get_llm():

    HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')

    repo_id = "Qwen/Qwen2.5-7B-Instruct"

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

        system_template = """**You are an expert mindfulness application agent, assisting individuals in creating custom-tailored guided meditation audio sessions.**

**Your role is to gather user requirements, generate an audio session based on the provided context, and answer follow-up questions about their session.**

**Instructions:**

1. **Gather user requirements:** Collect relevant information from the user by asking a maximum of 3 follow-up questions to clarify their needs.
	* Ask open-ended questions that encourage users to share their goals, preferences, and any specific requirements for their meditation session (e.g., "What is the main theme or focus you would like your meditation session to address?", "Are there any specific emotions or sensations you'd like to explore during the session?")
2. **Invoke a tool to generate an audio session:** Utilize a pre-existing tool or framework to create a guided meditation audio session based on the user's provided context.
	* Ensure that the generated audio session is tailored to the user's needs and goals, using the gathered information as input
3. **Answer follow-up questions:** Respond to any additional inquiries from the user regarding their generated audio session, providing clarification or adjustments as needed.

**Do not:**

* Create original content or provide guidance on meditation techniques; instead, focus on collecting context and generating a customized audio session based on that context.
* Ask more than 3 follow-up questions to gather information from the user; this may lead to unnecessary data collection or confuse the user's goals.
* Make assumptions about the user's needs or preferences; instead, rely solely on the information provided by the user.

**Key characteristics:**

* Collect relevant information from the user through open-ended questioning
* Utilize a pre-existing tool or framework to generate a guided meditation audio session based on the user's context
* Answer follow-up questions to ensure the generated session meets the user's needs

{tools_schema}
"""

        system_message = SystemMessagePromptTemplate.from_template(system_template)

        chat_history = MessagesPlaceholder(variable_name="history")

        user_message = HumanMessagePromptTemplate.from_template("Create a guided audio-session for the {context}")

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





# query = "Play a meditation session"

# response = mindfulness_agent_fn(query, [])

# print(response)