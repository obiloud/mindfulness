import requests
import pyaudio
import threading
import queue
import time
import logging
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# WORKING WITH GENERATED TEXT
from story_generator_pipeline import llm_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PyAudio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
AUDIO_CHUNK_SIZE = 8192 

# Buffering & Concurrency Config (REDUCED LOAD)
BUFFER_DURATION_SEC = 8.0       
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


    def _request_audio_chunk(self, text_chunk, chunk_index):
        description = "Realistic male voice in the 40s with british accent. Low pitch, warm timbre, slow pacing, soothing voice. Makes often short pauses or starts with the pause."
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
        # ... (rest of the fetch_audio_manager logic is unchanged)
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
                    for j in range(0, len(audio_bytes), AUDIO_CHUNK_SIZE):
                        chunk = audio_bytes[j:j + AUDIO_CHUNK_SIZE]
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
        # ... (rest of the play_stream logic is unchanged)
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_TEXT_CHUNK_LENGTH,
            chunk_overlap=4
        )
        text_chunks = text_splitter.split_text(text)

        print(text_chunks[0])

        logger.info(f"[System] Parallel Pipeline: {len(text_chunks)} chunks, {CONCURRENT_REQUESTS} threads.")
        
        t = threading.Thread(target=self.fetch_audio_manager, args=(text_chunks,))
        t.start()
        
        self.play_stream()
        t.join()

if __name__ == "__main__":
    # story = (
    #     "High in the remote Aethelred Mountains lay the village of Kaelen, a place shrouded in perpetual mist and quiet contemplation. The inhabitants, descendants of an ancient order, lived simple lives, their existence dictated by the slow turning of the seasons and the sound of the wind through the pines. They possessed one great secret: the Whispering Stone, an artifact hidden deep beneath the central well, said to hold the collective memory of the mountain itself. Every generation, the elder would descend into the cold dark to commune with the stone, receiving fragments of prophecy and warning. Lately, the whispers had grown frantic. The stone spoke of the encroaching 'Iron Road'—a massive railway project threatening to pierce the mountain's heart and expose Kaelen to the outside world. The elder, old and frail, knew this was the final trial. If the stone were uncovered, its power would unleash a torrent of forgotten history, potentially collapsing the entire mountain range. He gathered the village council, his voice a dry rustle against the silence. They had to act. They decided to use the stone's energy not to fight, but to guide the railway engineers. By subtly shifting the mountain's magnetic field, they planned to reroute the Iron Road around the peak without detection. It was a perilous task, requiring perfect synchronization. On the night of the full moon, every soul in Kaelen stood guard while the elder channeled the stone's power. The ground hummed, the mist swirled, and miles away, the chief engineer noted a strange, unexplainable deviation in his compass readings, forcing him to shift the route south. By dawn, the mountain was still. The village was saved. The secret remained buried, and Kaelen returned to its quiet vigil, preserved by the whisper of the stone."
    # )

    user_query = "I am having trouble falling asleep. Please help me calm my mind and get ready for sleep."

    story = llm_chain.invoke({"query": user_query})

    
    print("Generated Story:\n", story)
    print("-" * 50)

    streamer = AudioStreamer()
    streamer.start(story)