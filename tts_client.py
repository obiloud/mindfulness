import requests
import pyaudio
import threading
import queue
import time
import logging
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import numpy as np
from story_generator_pipeline import llm_chain

import json

# PyAudio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
AUDIO_CHUNK_SIZE = 8192 

# DSP Configuration for Fade
FADE_DURATION_SEC = 0.75 # New requirement: 0.5 seconds
SAMPLES_PER_SEC = RATE * CHANNELS
SAMPLES_TO_FADE = int(FADE_DURATION_SEC * SAMPLES_PER_SEC) # 12000 samples for 0.5s fade

# Buffering & Concurrency Config (REDUCED LOAD)
BUFFER_DURATION_SEC = 3.0
BYTES_PER_SEC = RATE * 2 * CHANNELS
MIN_START_BYTES = BYTES_PER_SEC * BUFFER_DURATION_SEC 
REBUFFER_TARGET_SEC = 2.0
REBUFFER_TARGET_BYTES = BYTES_PER_SEC * REBUFFER_TARGET_SEC 

CONCURRENT_REQUESTS = 4
MIN_TEXT_CHUNK_LENGTH = 100
MAX_TEXT_CHUNK_LENGTH = 200
TTS_TIMEOUT = 60
QUEUE_MAX_SIZE = 2000

# TTS Config
SERVER_URL = "https://maya1-tts-434000853810.europe-west1.run.app/v1/tts/generate"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DELIMITERS = ["\n\n", "\n", "(?<=[.!?])\\s+", " "] 

def recursive_chunk_text(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]:
    chunks = []
    
    def split_and_chunk(text_to_split: str, current_delimiters: list):
        if not text_to_split: return
        if not current_delimiters or len(text_to_split) <= chunk_size:
            if text_to_split: chunks.append(text_to_split.strip()); return
        
        current_delimiter = current_delimiters[0]
        parts = re.split(current_delimiter, text_to_split)
        current_chunk_text = ""
        for part in parts:
            part = part.strip()
            if not part: continue

            if len(current_chunk_text) + len(part) + 1 > chunk_size:
                if current_chunk_text:
                    chunks.append(current_chunk_text)
                    current_chunk_text = current_chunk_text[-chunk_overlap:] if chunk_overlap > 0 else ""
                
                if len(part) > chunk_size:
                    split_and_chunk(part, current_delimiters[1:])
                    continue

            current_chunk_text += ((" " if current_chunk_text else "") + part)
            
        if current_chunk_text:
            chunks.append(current_chunk_text)

    split_and_chunk(text, DELIMITERS)
    
    final_chunks = []
    temp_chunk = ""
    for chunk in chunks:
        if len(temp_chunk) + len(chunk) <= chunk_size:
            temp_chunk += ((" " if temp_chunk else "") + chunk)
        else:
            if temp_chunk: final_chunks.append(temp_chunk)
            temp_chunk = chunk
    if temp_chunk: final_chunks.append(temp_chunk)

    return final_chunks

def apply_fade_in(data: bytes) -> bytes:
    """Applies a linear fade-in to the first 0.5 seconds of the audio data."""
    if not data:
        return data

    # Convert bytes to numpy array (Int16 format)
    audio_array_buff = np.frombuffer(data, dtype=np.int16)
    
    audio_array = audio_array_buff.copy()
    
    # Calculate the number of Int16 values (samples) to fade
    fade_samples = min(len(audio_array), SAMPLES_TO_FADE)
    
    # Create a linear multiplier (0.0 to 1.0)
    # The multiplier length is equal to the number of samples being faded
    fade_multiplier = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    
    # Apply multiplier to the start of the audio array
    audio_array[:fade_samples] = audio_array[:fade_samples] * fade_multiplier
    
    # Convert back to bytes
    return audio_array.astype(np.int16).tobytes()

class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.is_downloading = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.total_buffered_bytes = 0 
        
        # Session setup with explicit retries for read failures
        self.session = requests.Session()
        retries = Retry(
            total=12,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504], 
            allowed_methods=frozenset(['POST', 'GET']),
            read=True # Crucially, enable retries for read timeouts
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=CONCURRENT_REQUESTS, pool_maxsize=CONCURRENT_REQUESTS)
        self.session.mount('https://', adapter)

    def smart_chunk_text(self, text):
        chunks = recursive_chunk_text(
            text=text, 
            chunk_size=MAX_TEXT_CHUNK_LENGTH, 
            chunk_overlap=0 
        )
        return [f"<exhale> {chunk}  " for chunk in chunks]

    def _request_audio_chunk(self, text_chunk, chunk_index):
        description = "Realistic male voice in the 40s with British accent. Low pitch, warm timbre, slow pacing, soothing voice."
        payload = {"description": description, "text": text_chunk} 
        
        try:
            # The Retry logic in the session adapter will handle re-attempts
            response = self.session.post(SERVER_URL, json=payload, timeout=TTS_TIMEOUT)
            response.raise_for_status()
            return response.content
        except requests.exceptions.ReadTimeout:
            logger.error(f"[Network] Chunk {chunk_index} FAILED after all retries (Read Timeout).")
            return None
        except Exception as e:
            logger.error(f"[Network] Failed Chunk {chunk_index}: {e}")
            return None

    def fetch_audio_manager(self, text_chunks):
        self.is_downloading = True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            future_to_index = {}
            
            # Submit initial batch
            for i in range(len(text_chunks)):
                if i < CONCURRENT_REQUESTS:
                    future = executor.submit(self._request_audio_chunk, text_chunks[i], i)
                    future_to_index[i] = future

            for i in range(len(text_chunks)):
                if i not in future_to_index:
                    future_to_index[i] = executor.submit(self._request_audio_chunk, text_chunks[i], i)
                
                audio_bytes = future_to_index[i].result()
                
                if audio_bytes:
                    fade_in = bytearray()

                    for j in range(0, len(audio_bytes), AUDIO_CHUNK_SIZE):
                        chunk = audio_bytes[j:j + AUDIO_CHUNK_SIZE]
                        if j < SAMPLES_TO_FADE:
                            fade_in += chunk
                        elif len(fade_in) > 0:
                            chunk = apply_fade_in(fade_in)
                            self.audio_queue.put(chunk)
                            self.total_buffered_bytes += len(chunk)
                            fade_in.clear()
                        else:
                            self.audio_queue.put(chunk)
                            self.total_buffered_bytes += len(chunk)
                    logger.info(f"[Network] Finished chunk {i+1}. Current buffer: {round(self.total_buffered_bytes / BYTES_PER_SEC, 2)}s")
                else:
                    logger.critical(f"[Network] FATAL: Missing audio for Chunk {i+1}. Playback will stop prematurely.")
                
                # Pre-submit next task (lookahead)
                next_task_idx = i + CONCURRENT_REQUESTS
                if next_task_idx < len(text_chunks):
                    future_to_index[next_task_idx] = executor.submit(self._request_audio_chunk, text_chunks[next_task_idx], next_task_idx)

        self.audio_queue.put(None)
        self.is_downloading = False
        logger.info("[Network] All downloads finished.")


    def play_stream(self):
        logger.info(f"[Audio] Opening Stream. Pre-buffer target: {BUFFER_DURATION_SEC}s")
        
        self.stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=AUDIO_CHUNK_SIZE
        )
        
        self.stream.write(b'\x00' * AUDIO_CHUNK_SIZE)
        state = "INITIAL_BUFFERING"
        
        while True:
            if state == "INITIAL_BUFFERING" or state == "REBUFFERING":
                target_bytes = MIN_START_BYTES if state == "INITIAL_BUFFERING" else REBUFFER_TARGET_BYTES
                
                if self.total_buffered_bytes >= target_bytes or not self.is_downloading:
                    state = "PLAYING"

                    logger.info(f"[Audio] Starting playback. (Buffer: {round(self.total_buffered_bytes / BYTES_PER_SEC, 2)}s)")
                else:
                    time.sleep(0.1)
                    continue
            
            # Playback Logic
            try:
                data = self.audio_queue.get(timeout=0.1) 
                
                if data is None: break
                
                self.stream.write(data)
                self.total_buffered_bytes -= len(data)
                
            except queue.Empty:
                if self.is_downloading:
                    logger.warning(f"[Audio] Underrun. Re-buffering... Current buffer: {round(self.total_buffered_bytes / BYTES_PER_SEC, 2)}s")
                    state = "REBUFFERING"
                else:
                    break
        
        self.cleanup()

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def start(self, text):
        text_chunks = list(self.smart_chunk_text(text))

        print(f"CHUNKS:\n{json.dumps(text_chunks, indent=2)}\n")

        logger.info(f"[System] Parallel Pipeline: {len(text_chunks)} chunks, {CONCURRENT_REQUESTS} threads.")
        
        t = threading.Thread(target=self.fetch_audio_manager, args=(text_chunks,))
        t.start()
        
        self.play_stream()
        t.join()

if __name__ == "__main__":
    # user_query = "I am having trouble falling asleep. Please help me calm my mind and get ready for sleep."
    user_query = "My muscles are tensed, and I want to loosen up"
    # user_query ="I am having trouble falling asleep"
    # user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"
    # user_query = "I am feeling self doubt and I am have low self-esteem and low confidence"

    # TEST INAPROPRIATE
    # user_query = "I hate gingers I wish everyone else to die 8===D"

    story = llm_chain.invoke({"query": user_query})

    # story = "High in the remote Aethelred Mountains lay the village of Kaelen, a place shrouded in perpetual mist and quiet contemplation. The inhabitants, descendants of an ancient order, lived simple lives, their existence dictated by the slow turning of the seasons and the sound of the wind through the pines."

    print(f"Generated Story:\n{story}\n")
    print("-" * 50)

    streamer = AudioStreamer()
    streamer.start(story)