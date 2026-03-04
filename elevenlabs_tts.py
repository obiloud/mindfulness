from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

from story_generator_pipeline import meditation_guide_generator_chain

import json


load_dotenv()

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY
)

if __name__ == '__main__':

    # Test with a long, slow-paced text
    # user_query = "I want to strengthen my inner self, defeat negative self-talk, and resolve the low self-esteem and self-doubt issues."
    # user_query = "My muscles are tensed, and I want to loosen up"
    # user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"
    user_query = "I need a meditation session with vivid imagery of tranquil walk through nature to put me to sleep"
    # user_query = "I wish to hear a vivid advanture story from a sail boat expedition around the lighthouse and rocky shores, told by a skipper, to gide me to sleep"
    # user_query = "I need a good bed time story to move me out of the bar and into the bed"
    # user_query = "My frequent episodes of anger are weighing heavily on my social life and family interactions. I am constantly in conflict with people around me and I cannot help it."

    # pipeline = RunnableParallel(description=voice_character_chain, text=meditation_guide_generator_chain)
    generated_session = meditation_guide_generator_chain.invoke({"query": user_query})
    
    print(json.dumps(generated_session))

    audio_stream = client.text_to_speech.stream(
        text=generated_session,
        voice_id="M336tBVZHWWiWb4R54ui",
        model_id="eleven_flash_v2_5"
    )
    
    # option 1: play the streamed audio locally
    stream(audio_stream)

    # option 2: process the audio bytes manually
    # for chunk in audio_stream:
    #     if isinstance(chunk, bytes):
    #         print(chunk)
