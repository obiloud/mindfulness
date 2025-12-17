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
import json
from story_generator_pipeline import llm_chain
from voice_generator import voice_character_chain
from langchain_core.runnables import RunnableParallel

# --- CONFIGURATION ---
# PyAudio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
AUDIO_CHUNK_SIZE = 8192 

# DSP Configuration for Smooth Transitions (Crossfade)
CROSSFADE_DURATION_SEC = 0.2  # 200ms overlap
SAMPLES_PER_SEC = RATE * CHANNELS
CROSSFADE_SAMPLES = int(CROSSFADE_DURATION_SEC * SAMPLES_PER_SEC)

# Token Estimation Constants (for budgeting)
TOKEN_PER_WORD = 50           
TOKEN_PER_TAG_SHORT = 150     # <chuckle>
TOKEN_PER_TAG_LONG = 400      # <laugh>
BASE_TOKEN_OVERHEAD = 150      

# Buffering & Concurrency Config 
BUFFER_DURATION_SEC = 6.0
BYTES_PER_SEC = RATE * 2 * CHANNELS
MIN_START_BYTES = BYTES_PER_SEC * BUFFER_DURATION_SEC 
REBUFFER_TARGET_SEC = 2.0
REBUFFER_TARGET_BYTES = BYTES_PER_SEC * REBUFFER_TARGET_SEC 

CONCURRENT_REQUESTS = 4
MIN_TEXT_CHUNK_LENGTH = 100
MAX_TEXT_CHUNK_LENGTH = 250   # Increased slightly for "Medium Text" efficiency
TTS_TIMEOUT = 90
QUEUE_MAX_SIZE = 2000

# TTS Config
SERVER_URL = "https://maya1-tts-434000853810.europe-west1.run.app/v1/tts/generate"
DESCRIPTION_DEFAULT = "Realistic male voice in the 40s with British accent. Low pitch, warm timbre, slow pacing, soothing voice."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DELIMITERS = ["\n\n", "\n", "(?<=[.!?])\\s+", " "] 

# --- Helper Functions ---

def estimate_max_tokens(text: str) -> int:
    """Calculates the recommended max_tokens based on content."""
    words = len(text.split())
    # Count specific high-cost tags
    laughs = len(re.findall(r'<laugh>', text, re.IGNORECASE))
    chuckles = len(re.findall(r'<chuckle>|<excited>', text, re.IGNORECASE))
    
    # Calculate budget
    token_budget = (words * TOKEN_PER_WORD) + \
                   (laughs * TOKEN_PER_TAG_LONG) + \
                   (chuckles * TOKEN_PER_TAG_SHORT) + \
                   BASE_TOKEN_OVERHEAD
                   
    # Clamp to recommended buckets
    if token_budget < 250: return 250
    if token_budget > 2000: return 2000 
    return int(token_budget)

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

# --- AudioStreamer Class ---

class AudioStreamer:
    def __init__(self, tts_description: str = DESCRIPTION_DEFAULT):
        self.audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.is_downloading = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.total_buffered_bytes = 0 
        self.tts_description = tts_description
        
        # State for Crossfading
        self.previous_chunk_tail = None 
        
        self.session = requests.Session()
        retries = Retry(
            total=12,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504], 
            allowed_methods=frozenset(['POST', 'GET']),
            read=True 
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=CONCURRENT_REQUESTS, pool_maxsize=CONCURRENT_REQUESTS)
        self.session.mount('https://', adapter)

    def smart_chunk_text(self, text):
        chunks = recursive_chunk_text(
            text=text, 
            chunk_size=MAX_TEXT_CHUNK_LENGTH, 
            chunk_overlap=0 
        )
        # return [f" mhmm {chunk} " for chunk in chunks] 
        return chunks # Removed "mhmm" for cleaner testing of crossfade

    def _request_audio_chunk(self, text_chunk, chunk_index):
        # NEW: Calculate optimized max_tokens
        max_tokens = estimate_max_tokens(text_chunk)
        
        payload = {
            "description": self.tts_description, 
            "text": text_chunk,
            "max_tokens": max_tokens  # Sending the budget to the server
        }
        
        try:
            response = self.session.post(SERVER_URL, json=payload, timeout=TTS_TIMEOUT)
            response.raise_for_status()
            return response.content
        except requests.exceptions.ReadTimeout:
            logger.error(f"[Network] Chunk {chunk_index} FAILED after all retries (Read Timeout).")
            return None
        except Exception as e:
            logger.error(f"[Network] Failed Chunk {chunk_index}: {e}")
            return None

    def _crossfade_chunks(self, new_chunk_bytes):
        """
        Mixes the end of the previous chunk with the start of the new chunk
        to prevent clicks and ensure smooth transitions.
        """
        if not new_chunk_bytes: return b""
        
        # Convert to float32 for mixing
        new_audio = np.frombuffer(new_chunk_bytes, dtype=np.int16).astype(np.float32)
        
        # 
        
        # Case 1: First chunk (no previous tail)
        if self.previous_chunk_tail is None:
            if len(new_audio) > CROSSFADE_SAMPLES:
                self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
                return new_audio[:-CROSSFADE_SAMPLES].astype(np.int16).tobytes()
            else:
                return new_chunk_bytes

        # Case 2: Standard crossfade
        prev_tail = self.previous_chunk_tail
        
        # Safety check: if new chunk is extremely short, skip fade to avoid index errors
        if len(new_audio) < CROSSFADE_SAMPLES:
             # Just append it, update tail if possible, otherwise reset
            self.previous_chunk_tail = new_audio if len(new_audio) > 0 else None
            return np.concatenate((prev_tail, new_audio)).astype(np.int16).tobytes()
            
        # 1. Extract head of new audio
        new_head = new_audio[:CROSSFADE_SAMPLES]
        
        # 2. Generate fade curves (Linear or Sine)
        fade_out = np.linspace(1.0, 0.0, CROSSFADE_SAMPLES, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, CROSSFADE_SAMPLES, dtype=np.float32)
        
        # 3. Mix overlap
        mixed_section = (prev_tail * fade_out) + (new_head * fade_in)
        
        # 4. Save NEW tail for next time
        self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
        
        # 5. Concatenate: Mixed Section + Body
        body_section = new_audio[CROSSFADE_SAMPLES:-CROSSFADE_SAMPLES]
        result = np.concatenate((mixed_section, body_section))
        
        return result.astype(np.int16).tobytes()

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
                    # NEW: Apply Crossfading instead of simple fade-in
                    processed_bytes = self._crossfade_chunks(audio_bytes)
                    
                    for j in range(0, len(processed_bytes), AUDIO_CHUNK_SIZE):
                        chunk = processed_bytes[j:j + AUDIO_CHUNK_SIZE]
                        self.audio_queue.put(chunk)
                        self.total_buffered_bytes += len(chunk)
                        
                    logger.info(f"[Network] Processed chunk {i+1}. Current buffer: {round(self.total_buffered_bytes / BYTES_PER_SEC, 2)}s")
                else:
                    logger.critical(f"[Network] FATAL: Missing audio for Chunk {i+1}. Playback will stop prematurely.")
                
                next_task_idx = i + CONCURRENT_REQUESTS
                if next_task_idx < len(text_chunks):
                    future_to_index[next_task_idx] = executor.submit(self._request_audio_chunk, text_chunks[next_task_idx], next_task_idx)

        # Flush the final tail (the last bit of audio)
        if self.previous_chunk_tail is not None:
            last_bytes = self.previous_chunk_tail.astype(np.int16).tobytes()
            self.audio_queue.put(last_bytes)
            self.total_buffered_bytes += len(last_bytes)

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
    user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"

    pipeline = RunnableParallel(
        description=voice_character_chain,
        text=llm_chain
    )
    result = pipeline.invoke({"query": user_query})
    print(json.dumps(result, indent=2))

    streamer = AudioStreamer(result['description'])
    streamer.start(result['text'])