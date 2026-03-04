import asyncio
import httpx
import sounddevice as sd
import numpy as np
import logging
from utils import recursive_word_chunker
from scipy.io.wavfile import write

# --- CONFIGURATION ---
RATE = 24000
CHANNELS = 1
# Pre-buffer 2.5 seconds for meditation quality
START_THRESHOLD_SAMPLES = int(RATE * 3)

SERVER_URL = "https://maya1-tts-434000853810.europe-west1.run.app/v1/tts/generate"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncAudioStreamer:
    def __init__(self, tts_description: str):
        self.tts_description = tts_description
        self.buffer = np.array([], dtype='float32')
        self.recorded = []
        self.lock = asyncio.Lock()
        self.streaming_complete = False
        self.stop_event = asyncio.Event()

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Hardware-timed pull. Note: This runs in a C-thread, 
        so we use a standard threading.Lock or simply rely on 
        atomic numpy operations if possible. Here we use a simpler approach.
        """
        # We use a small hack for the callback because sd runs in its own thread
        # We'll pull from a thread-safe list or keep the buffer logic lean
        available = len(self.buffer)
        
        if available == 0:
            outdata.fill(0)
            if self.streaming_complete:
                raise sd.CallbackStop
            return

        num_to_read = min(available, frames)
        # Pull data
        read_data = self.buffer[:num_to_read]
        self.buffer = self.buffer[num_to_read:]
        
        outdata[:num_to_read] = read_data.reshape(-1, CHANNELS)
        if num_to_read < frames:
            outdata[num_to_read:].fill(0)

    async def stream_audio(self, text):
        self.buffer = np.array([], dtype='float32')
        self.streaming_complete = False
        self.stop_event.clear()

        # Initialize the stream
        # blocksize=2048 helps asyncio manage the background tasks without jitter
        stream = sd.OutputStream(
            samplerate=RATE, channels=CHANNELS, dtype='float32', 
            callback=self._audio_callback, blocksize=2048
        )

        payload = {
            "description": self.tts_description, 
            "text": text, 
            "max_tokens": 2048,
            "stream": True,
            # "temperature": 0.3,
            # "top_p": 0.9
            # "max_words_per_chunk": 30
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            try:
                async with client.stream("POST", SERVER_URL, json=payload) as response:
                    response.raise_for_status()
                    
                    # Track if playback has started
                    playback_started = False
                    leftover = b''

                    async for chunk in response.aiter_bytes():
                        if self.stop_event.is_set():
                            break
                        
                        # Byte alignment
                        data = leftover + chunk
                        if len(data) % 2 != 0:
                            leftover = data[-1:]
                            data = data[:-1]
                        else:
                            leftover = b''

                        if len(data) > 0:
                            new_samples = np.frombuffer(data, dtype='int16').astype('float32') / 32768.0
                            # Append to buffer
                            self.buffer = np.append(self.buffer, new_samples)
                            self.recorded.append(new_samples)

                        # Logic to start the 'stream' context once the threshold is met
                        # if not playback_started and len(self.buffer) >= START_THRESHOLD_SAMPLES:
                        if not playback_started:
                            stream.start()
                            playback_started = True
                            logger.info("Jitter buffer filled. Playback started.")

            except Exception as e:
                logger.error(f"Async streaming error: {e}")
            finally:
                self.streaming_complete = True
                # If the stream was so short it never hit the threshold
                if not playback_started:
                    stream.start()
                
                # Wait for playback to finish
                while stream.active:
                    await asyncio.sleep(0.1)
                
                stream.stop()
                stream.close()

                write("qux.wav", RATE, np.concatenate(self.recorded))

    def stop(self):
        self.stop_event.set()

# --- HOW TO RUN ---
async def main():
    from story_generator_pipeline import meditation_guide_generator_chain
    from voice_generator_pipeline import voice_character_chain
    from langchain_core.runnables import RunnableParallel
    import re
    import json

    # user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"
    user_query = "I need a meditation session with vivid imagery of tranquil walk through nature to put me to sleep"
    # user_query = "I wish to hear a vivid advanture story from a sail boat expedition around the lighthouse and rocky shores, told by a skipper, to gide me to sleep"
    # pipeline = RunnableParallel(description=voice_character_chain, text=meditation_guide_generator_chain)
    # result = pipeline.invoke({"query": user_query})
    text = meditation_guide_generator_chain.invoke({"query": user_query})

    # pattern = r'\[PAUSE:\s?(\d+(?:\.\d+)?)\]'
    # result['text'] = re.sub(pattern=pattern, repl="", string=result['text'])

    # text = re.sub(pattern=pattern, repl="", string=text)

    # text =  """Hello, welcome to this guided meditation session. I'm here to help you cultivate calmness and focus ahead of your job interview tomorrow."""

#     # text = text.replace("\n\n", " ")
    print(text)
    # text = "Hello world <excited> this is amazing!"

    streamer = AsyncAudioStreamer("Realistic male voice in the 50s with Irish accent. Low pitch, gravely timbre, slow pacing.")
    await streamer.stream_audio(text)
    

if __name__ == "__main__":
    asyncio.run(main())