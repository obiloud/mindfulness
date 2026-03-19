export function mount_visualizer(containerId, analyser) {
    const container = document.getElementById(containerId);
    if (!container || !analyser) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    container.appendChild(canvas);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    let phase = 0; // Controls the gradient movement

    function resize() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    function draw() {
        requestAnimationFrame(draw);
        phase += 0.005; // Adjust this for faster/slower color cycling

        analyser.getByteTimeDomainData(dataArray);

        // 1. Smooth background trail
        ctx.fillStyle = 'rgba(15, 23, 42, 0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const baseRadius = Math.min(centerX, centerY) * 0.6;

        // 2. Animated Gradient Coordinates
        // Using sin/cos to move the gradient's focus point in a slow circle
        const x1 = centerX + Math.cos(phase) * centerX;
        const y1 = centerY + Math.sin(phase) * centerY;
        const x2 = centerX - Math.cos(phase) * centerX;
        const y2 = centerY - Math.sin(phase) * centerY;

        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        gradient.addColorStop(0, '#3098d0');   // Emerald 500
        gradient.addColorStop(0.5, '#32e2a2');
        gradient.addColorStop(1, '#7f5fd7');   // Blue 500

        ctx.beginPath();
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';

        // 3. Dynamic Glow
        // The glow "pulses" slightly with the phase
        const glowIntensity = 10 + Math.sin(phase * 2) * 5;
        ctx.shadowBlur = glowIntensity;
        ctx.shadowColor = 'rgba(16, 185, 129, 0.6)';

        for (let i = 0; i <= 360; i += 2) {
            const rad = (i * Math.PI) / 180;
            const index = Math.floor((i / 360) * dataArray.length);

            // Normalize PCM data (128 is silence)
            const amplitude = (dataArray[index] - 128) / 128;

            // The wave reacts to the voice
            const r = baseRadius + (amplitude * 50);

            const x = centerX + r * Math.cos(rad);
            const y = centerY + r * Math.sin(rad);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.closePath();
        ctx.stroke();
    }

    draw();
}