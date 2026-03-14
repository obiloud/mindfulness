// Global state for the audio pipeline
let audioCtx = null;
let nextStartTime = 0;
const SAMPLE_RATE = 44100;

export function toggle_audio_state(should_play) {
    if (!audioCtx) return;
    // Resuming/Suspending the context is the most efficient way to pause 
    // a scheduled timeline without losing our place.
    should_play ? audioCtx.resume() : audioCtx.suspend();
}

export function init_cartesia_stream(transcript, apiKey, dispatch) {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE,
        });
    }

    // Crucial for iOS: resume must happen inside the user-gesture call stack
    audioCtx.resume();
    nextStartTime = audioCtx.currentTime;

    // Use the latest stable 2026 versioning
    const url = `wss://api.cartesia.ai/tts/websocket?api_key=${apiKey}&cartesia_version=2024-06-10`;
    const socket = new WebSocket(url);

    socket.onopen = () => {
        const request = {
            model_id: "sonic-3", // Routes to latest stable snapshot
            transcript: transcript,
            language: "en",      // Mandatory for sonic-3 models
            voice: {
                mode: "id",
                id: "79a36f69-74f1-4177-8547-0e6d5e7542d1",
            },
            output_format: {
                container: "raw",
                encoding: "pcm_s16le",
                sample_rate: SAMPLE_RATE,
            }
        };
        socket.send(JSON.stringify(request));
    };

    socket.onmessage = (event) => {
        const response = JSON.parse(event.data);
        if (response.type === "chunk" && response.data) {
            handleAudioChunk(response.data);
        }
        if (response.done) {
            socket.close();
            dispatch({ type: "HandleFFIEvent", data: "AudioEnded" });
        }
    };

    socket.onerror = (err) => {
        console.error("Cartesia Error:", err);
        dispatch({ type: "HandleFFIEvent", data: "SocketError" });
    };
}

function handleAudioChunk(base64Data) {
    const binaryString = window.atob(base64Data);
    const len = binaryString.length;
    const bytes = new Int16Array(len / 2);

    for (let i = 0; i < len; i += 2) {
        bytes[i / 2] = (binaryString.charCodeAt(i + 1) << 8) | binaryString.charCodeAt(i);
    }

    const float32Data = new Float32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
        float32Data[i] = bytes[i] / 32768.0;
    }

    const buffer = audioCtx.createBuffer(1, float32Data.length, SAMPLE_RATE);
    buffer.copyToChannel(float32Data, 0);

    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);

    // Drifting protection: If nextStartTime is too far in the past, 
    // reset it to the current context time to avoid "catch-up" bursts.
    const lookahead = 0.1; // 100ms safety buffer
    if (nextStartTime < audioCtx.currentTime) {
        nextStartTime = audioCtx.currentTime + lookahead;
    }

    source.start(nextStartTime);
    nextStartTime += buffer.duration;
}