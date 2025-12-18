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
from story_generator_pipeline import story_generator_chain
from voice_generator import voice_character_chain
from langchain_core.runnables import RunnableParallel

# --- CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
AUDIO_CHUNK_SIZE = 8192 

# DSP Config
CROSSFADE_DURATION_SEC = 0.05
SAMPLES_PER_SEC = RATE * CHANNELS
CROSSFADE_SAMPLES = int(CROSSFADE_DURATION_SEC * SAMPLES_PER_SEC)

# Token Budget (100 WPM)
TOKEN_PER_WORD = 53
TOKEN_PER_TAG_SHORT = 150
TOKEN_PER_TAG_LONG = 400
BASE_TOKEN_OVERHEAD = 100

MAX_WORDS_PER_CHUNK = 35 

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

# --- UTILS ---

def get_word_count(text: str) -> int:
    return len(text.split())

def generate_silent_bytes(duration_sec: float) -> bytes:
    """Generates pure digital silence (zeros) for the specified duration."""
    num_samples = int(duration_sec * RATE * CHANNELS)
    # create array of zeros (int16)
    silent_array = np.zeros(num_samples, dtype=np.int16)
    return silent_array.tobytes()

def parse_pause_tags(text: str):
    """
    Splits text by [PAUSE:X] tags.
    Returns a list of mixed types: [str, float, str, float...]
    where float represents silence duration in seconds.
    """
    # Regex to find [PAUSE:2] or [PAUSE:0.5]
    pattern = r'\[PAUSE:(\d+(?:\.\d+)?)\]'
    parts = re.split(pattern, text)
    
    parsed_sequence = []
    
    # re.split returns [text, duration, text, duration...]
    # We need to reconstruct this carefully
    i = 0
    while i < len(parts):
        text_segment = parts[i].strip()
        if text_segment:
            parsed_sequence.append(text_segment)
        
        # If there is a next part, it is the captured group (duration)
        if i + 1 < len(parts):
            try:
                duration = float(parts[i+1])
                parsed_sequence.append(duration)
            except ValueError:
                pass # Should not happen with strict regex
        i += 2
        
    return parsed_sequence

def recursive_word_chunker(text: str, max_words: int) -> list[str]:
    # (Existing chunker logic - abbreviated for brevity)
    # Note: This runs on the text segments *between* pauses
    delimiters = ["\n\n", r"(?<=[.!?])\s+", r"(?<=[,;])\s+", " "]
    chunks = []
    
    def split_recursive(text_segment, delimiter_idx):
        if get_word_count(text_segment) <= max_words:
            if text_segment.strip(): chunks.append(text_segment.strip())
            return
        if delimiter_idx >= len(delimiters): # Hard split fallback
            words = text_segment.split()
            current = []
            for w in words:
                if len(current) + 1 > max_words:
                    chunks.append(" ".join(current)); current = [w]
                else: current.append(w)
            if current: chunks.append(" ".join(current))
            return
            
        delimiter = delimiters[delimiter_idx]
        parts = re.split(delimiter, text_segment)
        current_accumulation = ""
        for part in parts:
            part = part.strip()
            if not part: continue
            potential = (current_accumulation + " " + part).strip() if current_accumulation else part
            if get_word_count(potential) <= max_words:
                current_accumulation = potential
            else:
                if current_accumulation: chunks.append(current_accumulation); current_accumulation = ""
                if get_word_count(part) <= max_words: current_accumulation = part
                else: split_recursive(part, delimiter_idx + 1)
        if current_accumulation: chunks.append(current_accumulation)

    split_recursive(text, 0)
    
    # Merge optimization
    final = []
    if chunks:
        cur = chunks[0]
        for nxt in chunks[1:]:
            if get_word_count(cur) + get_word_count(nxt) <= max_words: cur += " " + nxt
            else: final.append(cur); cur = nxt
        final.append(cur)
    return final

# --- ESTIMATOR ---

def estimate_max_tokens(text: str) -> int:
    words = get_word_count(text)
    
    laughs = len(re.findall(r'<laugh>|<exhale>|<sigh>', text, re.IGNORECASE))
    chuckles = len(re.findall(r'<chuckle>|<excited>|<gasp>', text, re.IGNORECASE))
    
    token_budget = (words * TOKEN_PER_WORD) + \
                   (laughs * TOKEN_PER_TAG_LONG) + \
                   (chuckles * TOKEN_PER_TAG_SHORT) + \
                   BASE_TOKEN_OVERHEAD
    if token_budget < 250: return 250
    if token_budget > 2048: return 2048
    return int(token_budget)

# --- STREAMER ---

class AudioStreamer:
    def __init__(self, tts_description: str = DESCRIPTION_DEFAULT):
        self.audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.is_downloading = False
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.total_buffered_bytes = 0 
        self.tts_description = tts_description
        self.previous_chunk_tail = bytearray(CROSSFADE_SAMPLES) 
        
        self.session = requests.Session()
        retries = Retry(total=8, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=frozenset(['POST', 'GET']), read=True)
        adapter = HTTPAdapter(max_retries=retries, pool_connections=CONCURRENT_REQUESTS, pool_maxsize=CONCURRENT_REQUESTS)
        self.session.mount('https://', adapter)

    def prepare_pipeline(self, text):
        """
        1. Parses PAUSE tags.
        2. Chunks text segments.
        3. Returns a flat list of items: [{'type': 'text', 'content': '...'}, {'type': 'pause', 'duration': 2.0}]
        """
        raw_sequence = parse_pause_tags(text)
        pipeline_items = []
        
        for item in raw_sequence:
            if isinstance(item, float):
                # It's a pause
                pipeline_items.append({'type': 'pause', 'duration': item})
            elif isinstance(item, str):
                # It's text, chunk it further
                chunks = recursive_word_chunker(item, MAX_WORDS_PER_CHUNK)
                for c in chunks:
                    pipeline_items.append({'type': 'text', 'content': c})
                    
        return pipeline_items

    def _request_audio_chunk(self, text_chunk, chunk_index):
        # (Same network logic)
        max_tokens = estimate_max_tokens(text_chunk)
        payload = {"description": self.tts_description, "text": text_chunk, "max_tokens": max_tokens}
        try:
            response = self.session.post(SERVER_URL, json=payload, timeout=TTS_TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"[Network] Failed Chunk {chunk_index}: {e}")
            return None

    def _crossfade_chunks(self, new_chunk_bytes):
        if not new_chunk_bytes: return b""
        new_audio = np.frombuffer(new_chunk_bytes, dtype=np.int16).astype(np.float32)
        
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

    def fetch_audio_manager(self, pipeline_items):
        self.is_downloading = True
        
        # We need to manage futures manually because the list contains mixed types (text vs pause)
        # We only submit futures for 'text' items.
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            future_to_index = {}
            
            # Helper to submit a task
            def submit_task(idx):
                if idx < len(pipeline_items) and pipeline_items[idx]['type'] == 'text':
                    return executor.submit(self._request_audio_chunk, pipeline_items[idx]['content'], idx)
                return None

            # Initial submission
            for i in range(len(pipeline_items)):
                if i < CONCURRENT_REQUESTS:
                    future = submit_task(i)
                    if future: future_to_index[i] = future

            # Process loop
            for i in range(len(pipeline_items)):
                item = pipeline_items[i]
                
                if item['type'] == 'pause':
                    # 1. Flush any pending crossfade tail so the silence starts cleanly
                    if self.previous_chunk_tail is not None:
                        self.audio_queue.put(self.previous_chunk_tail.astype(np.int16).tobytes())
                        self.previous_chunk_tail = bytearray(CROSSFADE_SAMPLES)  # Reset crossfader
                    
                    # 2. Inject Silence
                    duration = item['duration']
                    logger.info(f"[System] Injecting {duration}s silence (Client-side).")
                    silent_bytes = generate_silent_bytes(duration)
                    
                    # Chunk the silence into queue to allow smooth playback
                    for j in range(0, len(silent_bytes), AUDIO_CHUNK_SIZE):
                        self.audio_queue.put(silent_bytes[j:j+AUDIO_CHUNK_SIZE])
                        self.total_buffered_bytes += len(silent_bytes[j:j+AUDIO_CHUNK_SIZE])
                        
                    continue # Skip to next item (no network request to wait for)

                # If text, wait for result
                if i not in future_to_index:
                    future = submit_task(i)
                    if future: future_to_index[i] = future
                
                if i in future_to_index:
                    audio_bytes = future_to_index[i].result()
                    if audio_bytes:
                        processed = self._crossfade_chunks(audio_bytes)
                        for j in range(0, len(processed), AUDIO_CHUNK_SIZE):
                            self.audio_queue.put(processed[j:j+AUDIO_CHUNK_SIZE])
                            self.total_buffered_bytes += len(processed[j:j+AUDIO_CHUNK_SIZE])
                
                # Lookahead submission
                next_idx = i + CONCURRENT_REQUESTS
                if next_idx < len(pipeline_items):
                    future = submit_task(next_idx)
                    if future: future_to_index[next_idx] = future

        # Final flush
        if self.previous_chunk_tail is not None:
            if not isinstance(self.previous_chunk_tail, bytearray):
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
                else: time.sleep(0.05); continue
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
        # 1. Prepare pipeline (Split text and pauses)
        pipeline = self.prepare_pipeline(text)
        
        print(f"PIPELINE PLAN:")
        for idx, item in enumerate(pipeline):
            if item['type'] == 'text':
                print(f"  {idx}: [TTS] {item['content'][:30]}...")
            else:
                print(f"  {idx}: [SILENCE] {item['duration']}s")
            
        t = threading.Thread(target=self.fetch_audio_manager, args=(pipeline,))
        t.start()
        self.play_stream()
        t.join()

if __name__ == "__main__":
    # Test with a long, slow-paced text
    # user_query = "I want to strengthen my inner self, defeat negative self-talk, and resolve the low self-esteem and self-doubt issues."
    # user_query = "My muscles are tensed, and I want to loosen up"
    # user_query = "I am having a job interview tomorrow and I am anxious about it, help me focus and relax"
    # user_query = "I need a meditation session with vivid imagery of tranquil walk through nature to put me to sleep"
    user_query = "I wish to hear a vivid advanture story from a sail boat expedition around the lighthouse and rocky shores, told by a skipper, to gide me to sleep"
    pipeline = RunnableParallel(description=voice_character_chain, text=story_generator_chain)
    result = pipeline.invoke({"query": user_query})
    
    print(json.dumps(result))

    streamer = AudioStreamer(result['description'])
    streamer.start(result['text'])
