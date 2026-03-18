let audioCtx = null;
let analyser = null;
let nextStartTime = 0;

export function init_audio() {
    // Cartesia sonic-3 uses a 24,000Hz sample rate 
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });

    // Create the persistent analyser bridge
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.8;

    // Connect the analyser to the speakers
    analyser.connect(audioCtx.destination);
}

export function get_analyser() {
    return analyser;
}

export function play_chunk(base64Data) {
    if (!audioCtx || !analyser) return;

    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Interpret bytes as 32-bit Floats (f32le) as per your implementation 
    const floatData = new Float32Array(bytes.buffer);

    const audioBuffer = audioCtx.createBuffer(1, floatData.length, 24000);
    audioBuffer.getChannelData(0).set(floatData);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;

    // Connect chunks to the analyser instead of the final destination
    source.connect(analyser);

    const currentTime = audioCtx.currentTime;
    if (nextStartTime < currentTime) {
        nextStartTime = currentTime;
    }
    source.start(nextStartTime);
    nextStartTime += audioBuffer.duration;
}