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
from gradio import ChatMessage
import json

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

audio_stream_generator = None

@tool
def generate_audio_guided_meditation_session(context: str) -> str:
    """ Generates the guided meditation session tailored to user's specific context and prepares the audio stream.

        Args:
            context (str): User's context/condition

        Returns:
            str: The transcript of the session
    """
    global audio_stream_generator

    pipeline = RunnableParallel(description=voice_character_chain, text=meditation_guide_generator_chain)
    result = pipeline.invoke({"query": context})

    voice_character = result.get("description")
    transcript = result.get("text", "")

    print(f"\n{json.dumps(result, indent=4)}\n")

    streamer = AudioStreamer()
    audio_stream_generator = streamer.make_generator(transcript)

    return f"The audio will start in a few seconds, please be patient. Here is the transcript: {transcript}"


class MindfulnessAgent:
    def __init__(self, llm, history=[], tool_mapping={}):

        self.history = history
        self.tool_mapping = tool_mapping

        chat_model = ChatHuggingFace(llm=llm)

        self.chat_model_with_tools = chat_model.bind_tools(self.tool_mapping.values())

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


def interaction_fn(user_input, history_state):
    global audio_stream_generator
    audio_stream_generator = None # Reset previous stream
    
    # 1. Update History
    history_state = history_state or []
    
    # 2. Initialize Agent
    tool_map = {"generate_audio_guided_meditation_session": generate_audio_guided_meditation_session}
    
    agent = MindfulnessAgent(
        llm=get_llm(), 
        history=history_state, 
        tool_mapping=tool_map
    )
    
    # 3. Run Agent Logic
    response_text = agent.run(user_input)
    
    # 4. Update History with AI response
    history_state.append({"role": "user", "content": user_input})
    
    # 5. Format Chat for Display (User, AI)
    # chat_display = []
    # for i in range(0, len(history_state), 2):
    #     u = history_state[i]
    #     a = history_state[i+1] if i+1 < len(history_state) else ChatMessage(role="assistant", content="")
    #     chat_display.append([u, a])

    # 6. Stream Audio if available
    # We yield the updated chat immediately, then yield audio chunks as they arrive
    if audio_stream_generator:
        # First yield: Text is done, Audio starts
        history_state.append({"role": "assistant", "content": "Your session will start shortly."})
        yield "", history_state, None
        
        # Loop through audio chunks
        for chunk in audio_stream_generator:
            # Yield: Text (static), History (static), Audio (new chunk)
            yield "", history_state, chunk
    
    # No audio generated, just return text
    history_state = history_state[:-1]
    history_state.append({"role": "assistant", "content": response_text})
    yield "", history_state, None


# --- 5. Gradio Layout (Blocks) ---

with gr.Blocks(title="Mindfulness AI") as demo:
    gr.Markdown("# 🧘 Mindfulness AI Agent")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            msg = gr.Textbox(label="How are you feeling?", placeholder="I'm feeling anxious about work...")
            submit_btn = gr.Button("Send")
        
        with gr.Column(scale=1):
            # The Audio component is set to 'streaming=True' and 'autoplay=True'
            # It expects (sample_rate, numpy array) tuples
            audio_out = gr.Audio(
                label="Guided Session", 
                streaming=True,
                autoplay=True,
                format="wav"
            )

    # State to hold conversation history
    state = gr.State([])

    # Event Listener
    submit_btn.click(
        fn=interaction_fn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, audio_out]
    )
    
    # Allow "Enter" key to submit
    msg.submit(
        fn=interaction_fn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, audio_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, theme=gr.themes.Soft())