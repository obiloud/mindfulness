// src/lib/CartesiaTTSClient.ts

import Cartesia from '@cartesia/cartesia-js';

class CartesiaTTSClient {
    private websocket: any = null;
    private isInitialized = false;
    private readonly API_KEY: string;
    private readonly MODEL_ID: string;
    private readonly VOICE_ID: string;

    constructor() {
        this.API_KEY = import.meta.env.VITE_CARTESIA_API_KEY;
        this.MODEL_ID = import.meta.env.VITE_CARTESIA_MODEL_ID || 'sonic-3';
        this.VOICE_ID = import.meta.env.VITE_CARTESIA_VOICE_ID || '6ccbfb76-1fc6-48f7-b71d-91ac6298247b';
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        try {
            const client = new Cartesia({
                apiKey: this.API_KEY,
            });

            this.websocket = await client.tts.websocket();
            this.isInitialized = true;
            console.log('✅ Connected to Cartesia TTS WebSocket');
        } catch (error) {
            console.error('❌ Failed to initialize Cartesia TTS connection:', error);
            throw error;
        }
    }

    async sendTranscript(transcript: string): Promise<void> {
        if (!this.isInitialized) {
            await this.initialize();
        }

        const sampleRate = 44100;
        const audioCtx = new AudioContext({ sampleRate });
        let nextStartTime = audioCtx.currentTime;

        for await (const event of this.websocket.generate({
            model_id: this.MODEL_ID,
            transcript: transcript,
            voice: { mode: 'id', id: this.VOICE_ID },
            output_format: { container: 'raw', encoding: 'pcm_f32le', sample_rate: sampleRate },
        })) {
            if (event.type === 'chunk' && event.audio) {
                const floats = new Float32Array(
                    event.audio.buffer,
                    event.audio.byteOffset,
                    event.audio.byteLength / 4,
                );

                const audioBuffer = audioCtx.createBuffer(1, floats.length, sampleRate);
                audioBuffer.getChannelData(0).set(floats);

                const source = audioCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioCtx.destination);

                // Schedule this chunk right after the previous one
                const startTime = Math.max(nextStartTime, audioCtx.currentTime);
                source.start(startTime);
                nextStartTime = startTime + audioBuffer.duration;
            }
        }

        this.websocket.close();
    }
}

export const cartesiaTTSClient = new CartesiaTTSClient();