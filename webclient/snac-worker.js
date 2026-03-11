// Use the full build that includes WebGPU (JSEP) support
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.min.js');

let session = null;
let queue = [];
let isFirstChunk = true;
let isStreamActive = false;

onmessage = async (e) => {
    if (e.data.type === 'INIT') {
        try {
            console.log("Worker: Initializing WebGPU...");

            // Point to the WebGPU/WASM binaries
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

            ort.env.webgpu.profilingMode = 'default';

            session = await ort.InferenceSession.create(e.data.modelUrl, {
                executionProviders: ['webgpu'],
                // Use 'all' to ensure maximum optimization on the M2's architecture
                graphOptimizationLevel: 'all',
                preferredOutputLocation: 'gpu-buffer',
                // New: Force high-performance power preference for smoother real-time audio
                freeDimensionOverrides: { "batch_size": 1 }
            });

            console.log("✅ Worker: SNAC Loaded with WebGPU Acceleration");

            while (queue.length > 0) {
                await performDecode(queue.shift());
            }
        } catch (err) {
            console.error("WebGPU Init Failed, falling back to WASM:", err);
            // Fallback logic
            ort.env.wasm.simd = true;
            session = await ort.InferenceSession.create(e.data.modelUrl, {
                executionProviders: ['wasm']
            });
        }
    } else if (e.data.type === 'DECODE') {
        if (!session) {
            queue.push(e.data.tokens);
        } else {

            // CHECK: Is 'data' a list of lists?
            if (Array.isArray(e.data.tokens[0])) {
                // It's a batch! Process each frame inside
                for (const frame of e.data.tokens) {
                    try {
                        await performDecode(frame);
                    } catch (err) {
                        console.error("Frame Decode Error:", err);
                    }
                }
            } else {
                // It's a single frame (fallback)
                await performDecode(e.data.tokens);
            }


        }
    }
};

async function performDecode(tokens, isEndOfStream = false) {
    let inputTensor = null;
    let audioTensor = null;
    try {
        const start = performance.now();

        // 1. Create Input
        inputTensor = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, 7]);

        // 2. Run Inference
        const results = await session.run({ "codes": inputTensor });

        // 3. Move Data to CPU
        audioTensor = results.audio;
        let rawPcm = await audioTensor.getData();

        // 4. Ensure it is Float32 for the Worklet
        let pcm = new Float32Array(rawPcm);

        // Logic: Only fade at the true boundaries of the speech
        const isFirst = !isStreamActive;
        isStreamActive = !isEndOfStream;

        pcm = applyBoundaryFades(pcm, isFirst, isEndOfStream);

        // 6. Post with Transferable (Zero-copy)
        postMessage({
            type: 'PCM_DATA',
            data: pcm,
            time: Math.round(performance.now() - start)
        }, [pcm.buffer]);

    } catch (err) {
        console.error("Decode Error:", err);
    } finally {
        // 7. CRITICAL: Clean up BOTH tensors to free WebGPU memory
        if (inputTensor) inputTensor.dispose();
        if (audioTensor) audioTensor.dispose();
    }
}

function applyBoundaryFades(pcm, fadeIn, fadeOut) {
    const fadeSamples = 48; // ~2ms is enough to prevent clicks without being audible

    if (fadeIn) {
        for (let i = 0; i < fadeSamples; i++) pcm[i] *= (i / fadeSamples);
    }

    if (fadeOut) {
        for (let i = 0; i < fadeSamples; i++) {
            pcm[pcm.length - 1 - i] *= (i / fadeSamples);
        }
    }
    // Middle chunks are returned raw, preventing the "pumping" effect
    return pcm;
}