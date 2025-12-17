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
from story_generator_pipeline import story_generator_chain
from voice_generator import voice_character_chain
from langchain_core.runnables import RunnableParallel

# --- CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
AUDIO_CHUNK_SIZE = 8192 

# DSP Config
CROSSFADE_DURATION_SEC = 0.1
SAMPLES_PER_SEC = RATE * CHANNELS
CROSSFADE_SAMPLES = int(CROSSFADE_DURATION_SEC * SAMPLES_PER_SEC)

# --- NEW TOKEN BUDGET CALCULATIONS (100 WPM) ---
# Based on: 87.5 tokens/sec * 0.6 sec/word = 52.5 tokens/word
TOKEN_PER_WORD = 53           # Rounded up from 52.5 for safety
TOKEN_PER_TAG_SHORT = 150     # <chuckle> (approx 2-3 sec worth)
TOKEN_PER_TAG_LONG = 400      # <laugh> (approx 5-6 sec worth)
BASE_TOKEN_OVERHEAD = 100     # General safety buffer

# Chunking Config
MAX_WORDS_PER_CHUNK = 35      # Strict limit as requested
MIN_WORDS_PER_CHUNK = 10      # Avoid tiny chunks if possible

# Buffering Config
BUFFER_DURATION_SEC = 6.0
BYTES_PER_SEC = RATE * 2 * CHANNELS
MIN_START_BYTES = BYTES_PER_SEC * BUFFER_DURATION_SEC 
REBUFFER_TARGET_SEC = 2.0
REBUFFER_TARGET_BYTES = BYTES_PER_SEC * REBUFFER_TARGET_SEC 

CONCURRENT_REQUESTS = 4
TTS_TIMEOUT = 60 * 3
QUEUE_MAX_SIZE = 2000

SERVER_URL = "https://maya1-tts-434000853810.europe-west1.run.app/v1/tts/generate"
DESCRIPTION_DEFAULT = "Realistic male voice in the 40s with British accent. Low pitch, warm timbre, slow pacing, soothing voice."

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NEW WORD-BASED SPLITTER ---

def get_word_count(text: str) -> int:
    """Returns the approximate word count."""
    return len(text.split())

def recursive_word_chunker(text: str, max_words: int) -> list[str]:
    """
    Splits text into chunks strictly keeping word count below max_words.
    Prioritizes splitting by Paragraph -> Sentence -> Punctuation -> Space.
    """
    # Delimiters ordered by priority (preserve context)
    delimiters = [
        "\n\n",             # Paragraphs
        r"(?<=[.!?])\s+",   # Sentences (lookbehind for punctuation)
        r"(?<=[,;])\s+",    # Clauses (commas/semicolons)
        " "                 # Words (last resort)
    ]
    
    chunks = []
    
    def split_recursive(text_segment, delimiter_idx):
        # Base case: fits in budget
        if get_word_count(text_segment) <= max_words:
            if text_segment.strip():
                chunks.append(text_segment.strip())
            return

        # Failure case: no more delimiters, but text is still too big
        # (This happens if a single word is somehow massive, or logic fail. We force split.)
        if delimiter_idx >= len(delimiters):
            # Fallback: Hard truncate (should rarely happen with space delimiter)
            words = text_segment.split()
            current_chunk = []
            for word in words:
                if len(current_chunk) + 1 > max_words:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                else:
                    current_chunk.append(word)
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return

        # Recursive Step
        delimiter = delimiters[delimiter_idx]
        parts = re.split(delimiter, text_segment)
        
        # Accumulate parts to form maximal chunks
        current_accumulation = ""
        
        for part in parts:
            part = part.strip()
            if not part: continue
            
            # Check size of (accumulation + new part)
            potential_text = (current_accumulation + " " + part).strip() if current_accumulation else part
            
            if get_word_count(potential_text) <= max_words:
                current_accumulation = potential_text
            else:
                # If adding the part exceeds limit, push current accumulation
                if current_accumulation:
                    chunks.append(current_accumulation)
                    current_accumulation = ""
                
                # Now handle the part that was too big
                # If the part ITSELF is smaller than max, start new accumulation
                if get_word_count(part) <= max_words:
                    current_accumulation = part
                else:
                    # The part itself is huge, recurse on it with finer delimiter
                    split_recursive(part, delimiter_idx + 1)
        
        # Flush remainder
        if current_accumulation:
            chunks.append(current_accumulation)

    # Start recursion
    split_recursive(text, 0)
    
    # Optional: Merge tiny chunks (orphaned words) into previous if space permits
    # This optimization prevents "hanging" words like "The." at the end of a stream.
    optimized_chunks = []
    if chunks:
        current = chunks[0]
        for next_chunk in chunks[1:]:
            if get_word_count(current) + get_word_count(next_chunk) <= max_words:
                current += " " + next_chunk
            else:
                optimized_chunks.append(current)
                current = next_chunk
        optimized_chunks.append(current)
        
    return optimized_chunks

# --- HELPER FUNCTIONS ---

def estimate_max_tokens(text: str) -> int:
    """
    Calculates max_tokens using the 100 WPM formula.
    Formula: (Words * 53) + Emotion Overhead + Base Buffer
    """
    words = get_word_count(text)
    
    laughs = len(re.findall(r'<laugh>', text, re.IGNORECASE))
    chuckles = len(re.findall(r'<chuckle>|<excited>', text, re.IGNORECASE))
    
    token_budget = (words * TOKEN_PER_WORD) + \
                   (laughs * TOKEN_PER_TAG_LONG) + \
                   (chuckles * TOKEN_PER_TAG_SHORT) + \
                   BASE_TOKEN_OVERHEAD
                   
    # Clamp to model limits
    if token_budget < 250: return 250   # Minimum floor
    if token_budget > 2048: return 2048 # Hard Hardware limit
    return int(token_budget)

# --- AUDIO STREAMER (Unchanged Logic, New Splitter) ---

class AudioStreamer:
    def __init__(self, tts_description: str = DESCRIPTION_DEFAULT):
        self.audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.is_downloading = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.total_buffered_bytes = 0 
        self.tts_description = tts_description
        self.previous_chunk_tail = None 
        
        self.session = requests.Session()
        retries = Retry(total=8, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=frozenset(['POST', 'GET']), read=True)
        adapter = HTTPAdapter(max_retries=retries, pool_connections=CONCURRENT_REQUESTS, pool_maxsize=CONCURRENT_REQUESTS)
        self.session.mount('https://', adapter)

    def smart_chunk_text(self, text):
        # CALL THE NEW WORD SPLITTER
        return recursive_word_chunker(text, MAX_WORDS_PER_CHUNK)

    def _request_audio_chunk(self, text_chunk, chunk_index):
        max_tokens = estimate_max_tokens(text_chunk)
        word_count = get_word_count(text_chunk)
        
        # Log specifically to verify strict word limit
        logger.info(f"[Network] Chunk {chunk_index}: {word_count} words | Budget: {max_tokens} tokens")
        
        payload = {
            "description": self.tts_description, 
            "text": text_chunk,
            "max_tokens": max_tokens
        }
        
        try:
            response = self.session.post(SERVER_URL, json=payload, timeout=TTS_TIMEOUT)
            response.raise_for_status()
            return response.content
        except requests.exceptions.ReadTimeout:
            logger.error(f"[Network] Chunk {chunk_index} FAILED (Read Timeout).")
            return None
        except Exception as e:
            logger.error(f"[Network] Failed Chunk {chunk_index}: {e}")
            return None

    def _crossfade_chunks(self, new_chunk_bytes):
        if not new_chunk_bytes: return b""
        
        new_audio = np.frombuffer(new_chunk_bytes, dtype=np.int16).astype(np.float32)
        
        # If chunk is too short (less than 2x crossfade), skip DSP to avoid index errors
        if len(new_audio) < (2 * CROSSFADE_SAMPLES):
            if self.previous_chunk_tail is not None:
                combined = np.concatenate((self.previous_chunk_tail, new_audio))
                self.previous_chunk_tail = None
                return combined.astype(np.int16).tobytes()
            return new_chunk_bytes

        if self.previous_chunk_tail is None:
            self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
            return new_audio[:-CROSSFADE_SAMPLES].astype(np.int16).tobytes()

        prev_tail = self.previous_chunk_tail
        new_head = new_audio[:CROSSFADE_SAMPLES]
        
        fade_out = np.linspace(1.0, 0.0, CROSSFADE_SAMPLES, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, CROSSFADE_SAMPLES, dtype=np.float32)
        
        mixed = (prev_tail * fade_out) + (new_head * fade_in)
        
        self.previous_chunk_tail = new_audio[-CROSSFADE_SAMPLES:]
        body = new_audio[CROSSFADE_SAMPLES:-CROSSFADE_SAMPLES]
        
        return np.concatenate((mixed, body)).astype(np.int16).tobytes()

    def fetch_audio_manager(self, text_chunks):
        self.is_downloading = True
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            future_to_index = {}
            for i in range(len(text_chunks)):
                if i < CONCURRENT_REQUESTS:
                    future = executor.submit(self._request_audio_chunk, text_chunks[i], i)
                    future_to_index[i] = future

            for i in range(len(text_chunks)):
                if i not in future_to_index:
                    future_to_index[i] = executor.submit(self._request_audio_chunk, text_chunks[i], i)
                
                audio_bytes = future_to_index[i].result()
                if audio_bytes:
                    processed = self._crossfade_chunks(audio_bytes)
                    for j in range(0, len(processed), AUDIO_CHUNK_SIZE):
                        self.audio_queue.put(processed[j:j+AUDIO_CHUNK_SIZE])
                        self.total_buffered_bytes += len(processed[j:j+AUDIO_CHUNK_SIZE])
                
                next_idx = i + CONCURRENT_REQUESTS
                if next_idx < len(text_chunks):
                    future_to_index[next_idx] = executor.submit(self._request_audio_chunk, text_chunks[next_idx], next_idx)

        if self.previous_chunk_tail is not None:
            self.audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())

        self.audio_queue.put(None)
        self.is_downloading = False

    def play_stream(self):
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=AUDIO_CHUNK_SIZE)
        state = "BUFFERING"
        while True:
            if state == "BUFFERING":
                target = MIN_START_BYTES if self.total_buffered_bytes == 0 else REBUFFER_TARGET_BYTES
                if self.total_buffered_bytes >= target or not self.is_downloading:
                    state = "PLAYING"
                else:
                    time.sleep(0.05); continue
            try:
                data = self.audio_queue.get(timeout=0.1)
                if data is None: break
                self.stream.write(data)
                self.total_buffered_bytes -= len(data)
            except queue.Empty:
                if self.is_downloading: state = "BUFFERING"
                else: break
        self.cleanup()

    def cleanup(self):
        if self.stream: self.stream.stop_stream(); self.stream.close()
        self.p.terminate()

    def start(self, text):
        text_chunks = list(self.smart_chunk_text(text))
        print(f"DEBUG: Split into {len(text_chunks)} chunks.")
        for i, c in enumerate(text_chunks):
            print(f"  Chunk {i} ({get_word_count(c)} words): {c[:50]}...")
            
        t = threading.Thread(target=self.fetch_audio_manager, args=(text_chunks,))
        t.start()
        self.play_stream()
        t.join()

if __name__ == "__main__":
    # Test with a long, slow-paced text
    user_query = "I want to strengthen my inner self, defeat negative self-talk and doubts, and fix low self-esteem and self-doubt."
    # user_query = "My muscles are tensed, and I want to loosen up"
    pipeline = RunnableParallel(description=voice_character_chain, text=story_generator_chain)
    result = pipeline.invoke({"query": user_query})
    
    streamer = AudioStreamer(result['description'])
    streamer.start(result['text'])