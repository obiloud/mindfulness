// snac-processor.js
class SNACProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.audioBuffer = new Float32Array(0);

        this.port.onmessage = (e) => {
            if (e.data.type === 'PCM_DATA') {
                const newAudio = e.data.data;
                if (!(newAudio instanceof Float32Array)) {
                    console.error("Worklet expected Float32Array, got:", typeof newAudio);
                    return;
                }
                this.appendAudio(e.data.data);
            } else if (e.data.type === 'PAUSE') {
                const silentSamples = new Float32Array(Math.floor(24000 * e.data.duration));
                this.appendAudio(silentSamples);
            }
        };
    }

    appendAudio(newAudio) {
        const combined = new Float32Array(this.audioBuffer.length + newAudio.length);
        combined.set(this.audioBuffer);
        combined.set(newAudio, this.audioBuffer.length);
        this.audioBuffer = combined;
    }

    process(inputs, outputs) {
        const output = outputs[0][0];
        if (this.audioBuffer.length >= output.length) {
            output.set(this.audioBuffer.subarray(0, output.length));
            this.audioBuffer = this.audioBuffer.subarray(output.length);
        }
        return true;
    }
}
registerProcessor('snac-processor', SNACProcessor);