import numpy as np
import pyaudio

# --- Configuration ---
SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
STD_DEV = 5 # Very small standard deviation for subtle noise

# --- PyAudio setup ---
p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    # Generate a new chunk of subtle white noise
    noise_data = np.random.normal(0, STD_DEV, frame_count).astype(np.int16)
    
    # In a real application, you'd check the input stream 'in_data' for silence
    # and either pass the original audio or this noise data.
    # For this example, we just output noise.
    
    return (noise_data.tobytes(), pyaudio.paContinue)

# Open stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                stream_callback=callback)

# Start the stream
stream.start_stream()

# Keep the stream running (e.g., using a while loop or an input prompt)
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    pass

# Stop and close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()
