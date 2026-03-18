export function mount_visualizer(containerId, analyser) {
    const container = document.getElementById(containerId);
    if (!container || !analyser) return;

    // Create canvas and append to Lustre's container
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    container.appendChild(canvas);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    function resize() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    function draw() {
        requestAnimationFrame(draw);

        // Get the latest audio bytes
        analyser.getByteTimeDomainData(dataArray);

        // Deep forest background with trailing "blur"
        ctx.fillStyle = 'rgba(15, 42, 41, 0.74)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const baseRadius = Math.min(centerX, centerY) * 0.6;

        ctx.beginPath();
        ctx.strokeStyle = '#1f9f72'; // Emerald-400
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';

        for (let i = 0; i <= 360; i++) {
            const rad = (i * Math.PI) / 180;

            // Map audio data to the radius
            const index = Math.floor((i / 360) * dataArray.length);
            const amplitude = (dataArray[index] - 128) / 128; // Normalize to -1.0 ... 1.0

            // Smooth out the movement for a "breathing" feel
            const r = baseRadius + (amplitude * 40);

            const x = centerX + r * Math.cos(rad);
            const y = centerY + r * Math.sin(rad);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        ctx.closePath();
        ctx.stroke();

        // Add a soft glow
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'rgba(19, 48, 40, 0.3)';
    }

    draw();
}