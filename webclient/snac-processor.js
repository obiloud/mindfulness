class SNACProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 2 seconds @ 24kHz. 
        this.bufferSize = 48000;
        this.buffer = new Float32Array(this.bufferSize);
        this.readIndex = 0;
        this.writeIndex = 0;
        this.isBuffering = true;

        // FIXED: Lower threshold (approx 200ms) to avoid deadlock
        // 24000 * 0.2 = 4,800 samples
        this.bufferThreshold = 24000 * 0.5;

        this.port.onmessage = (e) => {
            if (e.data.type === 'PCM_DATA') {
                this.writeToBuffer(e.data.data);
            }
        };
    }

    writeToBuffer(data) {
        for (let i = 0; i < data.length; i++) {
            this.buffer[this.writeIndex] = data[i];
            this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
        }
    }

    process(inputs, outputs) {
        const output = outputs[0][0];
        if (!output) return true;

        // Calculate how many samples are currently waiting
        const available = (this.writeIndex - this.readIndex + this.bufferSize) % this.bufferSize;

        // 1. Buffering Logic
        if (this.isBuffering) {
            if (available >= this.bufferThreshold) {
                this.isBuffering = false;
                console.log("Buffer ready, starting playback...");
            } else {
                output.fill(0);
                return true;
            }
        }

        // 2. Underflow Protection
        if (available < output.length) {
            console.warn("Underflow detected, re-buffering...");
            this.isBuffering = true;
            output.fill(0);
            return true;
        }

        // 3. Playback
        for (let i = 0; i < output.length; i++) {
            output[i] = this.buffer[this.readIndex];
            this.readIndex = (this.readIndex + 1) % this.bufferSize;
        }

        return true;
    }
}

registerProcessor('snac-processor', SNACProcessor);