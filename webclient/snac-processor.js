class SNACProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // 2 seconds @ 24kHz is 48,000 samples. Correct.
        this.buffer = new Float32Array(48000);
        this.readIndex = 0;
        this.writeIndex = 0;
        this.isBuffering = true;

        // Change from 0.34 (340ms) to 0.1 (100ms)
        this.bufferThreshold = 24000 * 0.1;

        this.port.onmessage = (e) => {
            if (e.data.type === 'PCM_DATA') {
                const data = e.data.data;

                // Use .set() for fast block copying
                if (this.writeIndex + data.length < this.buffer.length) {
                    this.buffer.set(data, this.writeIndex);
                    this.writeIndex += data.length;
                } else {
                    // Handle circular wrap-around
                    const firstPart = this.buffer.length - this.writeIndex;
                    this.buffer.set(data.subarray(0, firstPart), this.writeIndex);
                    this.buffer.set(data.subarray(firstPart), 0);
                    this.writeIndex = data.length - firstPart;
                }
            }
        };
    }

    writeToBuffer(data) {
        for (let i = 0; i < data.length; i++) {
            this.buffer[this.writeIndex] = data[i];
            this.writeIndex = (this.writeIndex + 1) % this.buffer.length;
        }
    }

    process(inputs, outputs) {
        const output = outputs[0][0];
        let available = (this.writeIndex - this.readIndex + this.buffer.length) % this.buffer.length;

        // If we have data, play it. Only enter 'isBuffering' if we are COMPLETELY empty.
        if (available < output.length) {
            if (!this.isBuffering) {
                console.warn("Underflow! Coasting...");
                this.isBuffering = true;
            }
            output.fill(0);
            return true;
        }

        // Once we've started playing, don't stop unless we hit 0 samples
        for (let i = 0; i < output.length; i++) {
            output[i] = this.buffer[this.readIndex];
            this.readIndex = (this.readIndex + 1) % this.buffer.length;
        }
        return true;
    }
}

registerProcessor('snac-processor', SNACProcessor);