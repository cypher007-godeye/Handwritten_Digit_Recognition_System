const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const fileInput = document.getElementById('fileInput');

const placeholder = document.getElementById('placeholder');
const resultArea = document.getElementById('resultArea');
const predictedDigit = document.getElementById('predictedDigit');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const errorArea = document.getElementById('errorArea');
const errorMessage = document.getElementById('errorMessage');
const historyList = document.getElementById('historyList');

let isDrawing = false;
let predictionsCount = 0;

// Initialize Canvas
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'white';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing Logic
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
    const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

clearBtn.addEventListener('click', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResults();
});

async function sendPrediction(imageB64) {
    showLoading();
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageB64 })
        });
        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            showResult(data.digit, data.confidence);
            addToHistory(data.digit, data.confidence);
        }
    } catch (e) {
        showError("Neural link severed. Server unreachable.");
    }
}

predictBtn.addEventListener('click', () => {
    const imageB64 = canvas.toDataURL('image/png');
    sendPrediction(imageB64);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        sendPrediction(event.target.result);
    };
    reader.readAsDataURL(file);
});

function showLoading() {
    placeholder.classList.remove('hidden');
    resultArea.classList.add('hidden');
    errorArea.classList.add('hidden');
    placeholder.innerHTML = '<div class="text-5xl mb-4 animate-spin">⚛️</div><p class="text-violet-400 animate-pulse">Analyzing Neural Patterns...</p>';
}

function showResult(digit, confidence) {
    placeholder.classList.add('hidden');
    resultArea.classList.remove('hidden');
    errorArea.classList.add('hidden');

    predictedDigit.innerText = digit;
    // Add pop animation by resetting it
    predictedDigit.classList.remove('prediction-animate');
    void predictedDigit.offsetWidth;
    predictedDigit.classList.add('prediction-animate');

    confidenceText.innerText = `Confidence: ${confidence}`;
    confidenceBar.style.width = confidence;
}

function showError(msg) {
    placeholder.classList.add('hidden');
    resultArea.classList.add('hidden');
    errorArea.classList.remove('hidden');
    errorMessage.innerText = msg;
}

function hideResults() {
    placeholder.classList.remove('hidden');
    resultArea.classList.add('hidden');
    errorArea.classList.add('hidden');
    placeholder.innerHTML = '<div class="text-8xl mb-6 opacity-20">🧠</div><p class="text-lg italic">Awaiting neural input...</p>';
}

function addToHistory(digit, confidence) {
    if (predictionsCount === 0) {
        historyList.innerHTML = '';
    }
    predictionsCount++;

    const item = document.createElement('div');
    item.className = 'history-item p-3 rounded-xl flex justify-between items-center text-xs font-mono';
    item.innerHTML = `
        <span class="text-slate-400">#${predictionsCount} Digit: <b class="text-white">${digit}</b></span>
        <span class="text-violet-400 font-bold">${confidence}</span>
    `;

    historyList.prepend(item);
}
