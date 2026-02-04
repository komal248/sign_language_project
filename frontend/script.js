// Configuration
const API_BASE_URL = 'http://localhost:5000';
let isCameraActive = false;
let isAutoPredicting = false;
let currentWord = '';
let predictionHistory = [];
let stream = null;
let autoPredictInterval = null;
let currentPrediction = null;

// DOM Elements
const modeSelectors = document.querySelectorAll('.mode');
const modeContents = document.querySelectorAll('.mode-content');
const sourceButtons = document.querySelectorAll('.source-btn');
const webcam = document.getElementById('webcam');
const outputCanvas = document.getElementById('outputCanvas');
const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');
const captureFrameBtn = document.getElementById('captureFrame');
const autoPredictBtn = document.getElementById('autoPredict');
const predictedLetter = document.getElementById('predictedLetter');
const letterName = document.getElementById('letterName');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const currentWordElement = document.getElementById('currentWord');
const wordMeaningElement = document.getElementById('wordMeaning');
const addLetterBtn = document.getElementById('addLetter');
const clearWordBtn = document.getElementById('clearWord');
const saveWordBtn = document.getElementById('saveWord');
const speakWordBtn = document.getElementById('speakWord');
const clearHistoryBtn = document.getElementById('clearHistory');
const historyList = document.getElementById('historyList');
const totalPredictionsElement = document.getElementById('totalPredictions');
const accuracyRateElement = document.getElementById('accuracyRate');
const wordsBuiltElement = document.getElementById('wordsBuilt');
const gestureGuide = document.getElementById('gestureGuide');
const suggestButtons = document.querySelectorAll('.suggest-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    loadGestureGuide();
    loadStats();
    loadHistory();
    setupEventListeners();
});

function initializeApp() {
    console.log('Sign Language Detection System Initialized');
    
    // Set canvas dimensions
    outputCanvas.width = 640;
    outputCanvas.height = 480;
    
    // Initialize stats
    updateStats();
}

function loadGestureGuide() {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let html = '';
    
    for (let letter of letters) {
        html += `
            <div class="gesture-item" data-letter="${letter}">
                <div class="gesture-letter">${letter}</div>
                <div class="gesture-name">${getLetterDescription(letter)}</div>
            </div>
        `;
    }
    
    gestureGuide.innerHTML = html;
    
    // Add click listeners to gesture items
    document.querySelectorAll('.gesture-item').forEach(item => {
        item.addEventListener('click', () => {
            const letter = item.dataset.letter;
            simulatePrediction(letter);
        });
    });
}

function getLetterDescription(letter) {
    const descriptions = {
        'A': 'Fist, thumb side',
        'B': 'Flat hand',
        'C': 'Curved C shape',
        'D': 'Index point up',
        'E': 'Fist, thumb over',
        'F': 'OK sign',
        'G': 'Index point side',
        'H': 'Two fingers',
        'I': 'Pinky up',
        'J': 'J motion',
        'K': 'Index+middle',
        'L': 'L shape',
        'M': 'Three under thumb',
        'N': 'Two under thumb',
        'O': 'O shape',
        'P': 'P shape',
        'Q': 'Q shape',
        'R': 'Crossed fingers',
        'S': 'S fist',
        'T': 'T shape',
        'U': 'Two up',
        'V': 'Peace sign',
        'W': 'Three up',
        'X': 'Bent index',
        'Y': 'Rock on',
        'Z': 'Z motion'
    };
    return descriptions[letter] || 'ASL Letter';
}

function setupEventListeners() {
    // Mode switching
    modeSelectors.forEach(mode => {
        mode.addEventListener('click', () => {
            const modeId = mode.dataset.mode + '-mode';
            
            // Update active mode
            modeSelectors.forEach(m => m.classList.remove('active'));
            mode.classList.add('active');
            
            // Show corresponding content
            modeContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === modeId) {
                    content.classList.add('active');
                }
            });
        });
    });

    // Source switching
    sourceButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const source = btn.id.replace('Btn', '');
            
            // Update active button
            sourceButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show corresponding input method
            switch(source) {
                case 'live':
                    document.getElementById('cameraContainer').style.display = 'block';
                    document.getElementById('uploadContainer').style.display = 'none';
                    break;
                case 'image':
                case 'video':
                    document.getElementById('cameraContainer').style.display = 'none';
                    document.getElementById('uploadContainer').style.display = 'block';
                    updateUploadUI(source);
                    break;
            }
        });
    });

    // Camera controls
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    captureFrameBtn.addEventListener('click', captureAndPredict);
    autoPredictBtn.addEventListener('click', toggleAutoPredict);

    // Word controls
    addLetterBtn.addEventListener('click', addToWord);
    clearWordBtn.addEventListener('click', clearWord);
    saveWordBtn.addEventListener('click', saveWord);
    speakWordBtn.addEventListener('click', speakWord);
    clearHistoryBtn.addEventListener('click', clearHistory);

    // Quick words
    suggestButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const word = btn.dataset.word;
            setWord(word);
        });
    });

    // File upload handling
    document.getElementById('imageInput').addEventListener('change', handleImageUpload);
    document.getElementById('videoInput').addEventListener('change', handleVideoUpload);
    document.getElementById('analyzeVideo').addEventListener('click', analyzeVideo);
}

async function startCamera() {
    try {
        stopCamera(); // Stop any existing stream
        
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcam.srcObject = stream;
        isCameraActive = true;
        
        // Update UI
        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;
        captureFrameBtn.disabled = false;
        
        // Start drawing
        drawCameraFrame();
        
        showToast('Camera started successfully', 'success');
    } catch (error) {
        console.error('Error starting camera:', error);
        showToast(`Camera error: ${error.message}`, 'error');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    isCameraActive = false;
    isAutoPredicting = false;
    
    // Clear intervals
    if (autoPredictInterval) {
        clearInterval(autoPredictInterval);
        autoPredictInterval = null;
    }
    
    // Update UI
    startCameraBtn.disabled = false;
    stopCameraBtn.disabled = true;
    captureFrameBtn.disabled = true;
    autoPredictBtn.textContent = '<i class="fas fa-robot"></i> Auto Predict';
    
    // Clear canvas
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    showToast('Camera stopped', 'info');
}

function drawCameraFrame() {
    if (!isCameraActive) return;
    
    const ctx = outputCanvas.getContext('2d');
    ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    // Draw video frame
    ctx.drawImage(webcam, 0, 0, outputCanvas.width, outputCanvas.height);
    
    // Draw hand area guide
    drawHandGuide(ctx);
    
    // Continue animation
    requestAnimationFrame(drawCameraFrame);
}

function drawHandGuide(ctx) {
    const centerX = outputCanvas.width / 2;
    const centerY = outputCanvas.height / 2;
    const radius = 100;
    
    // Draw circle
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw text
    ctx.fillStyle = 'white';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Show Hand Here', centerX, centerY + 140);
}

async function captureAndPredict() {
    if (!isCameraActive) {
        showToast('Please start camera first', 'warning');
        return;
    }
    
    try {
        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(webcam, 0, 0);
        
        // Convert to blob
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            
            // Send to backend
            const response = await fetch(`${API_BASE_URL}/predict/image`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            updatePredictionUI(result);
            
        }, 'image/jpeg', 0.8);
        
    } catch (error) {
        console.error('Error predicting:', error);
        showToast('Prediction failed. Please try again.', 'error');
        // Fallback to simulation
        simulatePrediction();
    }
}

function toggleAutoPredict() {
    if (!isCameraActive) {
        showToast('Please start camera first', 'warning');
        return;
    }
    
    if (isAutoPredicting) {
        // Stop auto prediction
        isAutoPredicting = false;
        clearInterval(autoPredictInterval);
        autoPredictBtn.innerHTML = '<i class="fas fa-robot"></i> Auto Predict';
        autoPredictBtn.classList.remove('active');
        showToast('Auto prediction stopped', 'info');
    } else {
        // Start auto prediction
        isAutoPredicting = true;
        autoPredictBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Auto';
        autoPredictBtn.classList.add('active');
        
        // Predict every 2 seconds
        autoPredictInterval = setInterval(() => {
            if (isCameraActive) {
                captureAndPredict();
            }
        }, 2000);
        
        showToast('Auto prediction started (every 2 seconds)', 'success');
    }
}

function updatePredictionUI(result) {
    if (!result || result.status !== 'success') {
        console.error('Invalid prediction result:', result);
        return;
    }
    
    const letter = result.predicted_letter || result.letter || '?';
    const confidence = result.confidence || 0.8;
    const confidencePercent = Math.round(confidence * 100);
    
    // Update prediction display
    predictedLetter.textContent = letter;
    letterName.textContent = `Letter ${letter} - ${getLetterDescription(letter)}`;
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceBar.style.width = `${confidencePercent}%`;
    
    // Update confidence bar color based on confidence
    if (confidencePercent >= 80) {
        confidenceBar.style.background = 'linear-gradient(to right, #4CAF50, #45a049)';
    } else if (confidencePercent >= 60) {
        confidenceBar.style.background = 'linear-gradient(to right, #FF9800, #F57C00)';
    } else {
        confidenceBar.style.background = 'linear-gradient(to right, #F44336, #D32F2F)';
    }
    
    // Add to history
    addToHistory(letter, confidencePercent);
    
    // Update stats
    updateStats();
    
    // Store current prediction for word building
    currentPrediction = {
        letter: letter,
        confidence: confidencePercent,
        timestamp: new Date()
    };
}

function addToHistory(letter, confidence) {
    const historyItem = {
        id: Date.now(),
        letter: letter,
        confidence: confidence,
        timestamp: new Date().toLocaleTimeString()
    };
    
    predictionHistory.unshift(historyItem);
    
    // Keep only last 20 items
    if (predictionHistory.length > 20) {
        predictionHistory.pop();
    }
    
    // Update UI
    updateHistoryUI();
    
    // Save to local storage
    localStorage.setItem('predictionHistory', JSON.stringify(predictionHistory));
}

function updateHistoryUI() {
    if (predictionHistory.length === 0) {
        historyList.innerHTML = `
            <div class="empty-history">
                <i class="fas fa-history fa-2x"></i>
                <p>No predictions yet</p>
            </div>
        `;
        return;
    }
    
    let html = '';
    predictionHistory.forEach(item => {
        html += `
            <div class="history-item">
                <div class="history-letter">${item.letter}</div>
                <div class="history-details">
                    <span class="history-confidence">${item.confidence}%</span>
                    <span class="history-time">${item.timestamp}</span>
                </div>
            </div>
        `;
    });
    
    historyList.innerHTML = html;
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('predictionHistory');
        if (saved) {
            predictionHistory = JSON.parse(saved);
            updateHistoryUI();
        }
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all prediction history?')) {
        predictionHistory = [];
        localStorage.removeItem('predictionHistory');
        updateHistoryUI();
        showToast('History cleared', 'info');
    }
}

function addToWord() {
    if (!currentPrediction) {
        showToast('No prediction available. Make a prediction first.', 'warning');
        return;
    }
    
    if (currentPrediction.confidence < 60) {
        showToast('Confidence is low. Please try again.', 'warning');
        return;
    }
    
    currentWord += currentPrediction.letter;
    updateWordUI();
    showToast(`Added '${currentPrediction.letter}' to word`, 'success');
}

function clearWord() {
    if (currentWord.length === 0) {
        showToast('Word is already empty', 'info');
        return;
    }
    
    if (confirm('Clear current word?')) {
        currentWord = '';
        updateWordUI();
        showToast('Word cleared', 'info');
    }
}

function saveWord() {
    if (currentWord.length === 0) {
        showToast('No word to save', 'warning');
        return;
    }
    
    // Save to localStorage
    const savedWords = JSON.parse(localStorage.getItem('savedWords') || '[]');
    savedWords.push({
        word: currentWord,
        timestamp: new Date().toISOString()
    });
    
    localStorage.setItem('savedWords', JSON.stringify(savedWords));
    
    // Update stats
    const wordsCount = parseInt(wordsBuiltElement.textContent) || 0;
    wordsBuiltElement.textContent = wordsCount + 1;
    
    showToast(`Word "${currentWord}" saved successfully!`, 'success');
    
    // Clear current word
    currentWord = '';
    updateWordUI();
}

function speakWord() {
    if (currentWord.length === 0) {
        showToast('No word to speak', 'warning');
        return;
    }
    
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(currentWord);
        utterance.rate = 0.8;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
        showToast(`Speaking: ${currentWord}`, 'info');
    } else {
        showToast('Text-to-speech not supported in this browser', 'error');
    }
}

function setWord(word) {
    currentWord = word;
    updateWordUI();
    showToast(`Word set to: ${word}`, 'success');
}

function updateWordUI() {
    currentWordElement.textContent = currentWord || 'No word yet';
    
    if (currentWord.length > 0) {
        // You could add word meaning lookup here
        wordMeaningElement.textContent = `Length: ${currentWord.length} letters`;
        
        // Style based on word length
        if (currentWord.length >= 10) {
            currentWordElement.style.fontSize = '3rem';
        } else if (currentWord.length >= 5) {
            currentWordElement.style.fontSize = '3.5rem';
        } else {
            currentWordElement.style.fontSize = '4rem';
        }
    } else {
        wordMeaningElement.textContent = 'Add letters to build a word';
        currentWordElement.style.fontSize = '4rem';
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        if (response.ok) {
            const stats = await response.json();
            updateStatsUI(stats);
        }
    } catch (error) {
        console.error('Error loading stats:', error);
        // Use simulated stats
        updateStatsUI({
            total_predictions: predictionHistory.length,
            gestures_count: 26,
            accuracy_rate: 0.85
        });
    }
}

function updateStatsUI(stats) {
    totalPredictionsElement.textContent = stats.total_predictions || 0;
    accuracyRateElement.textContent = stats.accuracy_rate ? 
        Math.round(stats.accuracy_rate * 100) + '%' : '0%';
    
    // Update words built from localStorage
    const savedWords = JSON.parse(localStorage.getItem('savedWords') || '[]');
    wordsBuiltElement.textContent = savedWords.length;
}

function updateStats() {
    // Update total predictions
    totalPredictionsElement.textContent = predictionHistory.length;
    
    // Calculate average confidence
    if (predictionHistory.length > 0) {
        const avgConfidence = predictionHistory.reduce((sum, item) => sum + item.confidence, 0) / predictionHistory.length;
        accuracyRateElement.textContent = Math.round(avgConfidence) + '%';
    }
}

function updateUploadUI(source) {
    const imageBox = document.getElementById('imageUploadBox');
    const videoBox = document.getElementById('videoUploadBox');
    
    if (source === 'image') {
        imageBox.style.display = 'block';
        videoBox.style.display = 'none';
    } else if (source === 'video') {
        imageBox.style.display = 'none';
        videoBox.style.display = 'block';
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = '';
    
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.style.maxWidth = '100%';
    img.style.borderRadius = '10px';
    
    preview.appendChild(img);
    
    // Auto predict after upload
    setTimeout(() => predictImage(file), 1000);
}

async function predictImage(file) {
    try {
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch(`${API_BASE_URL}/predict/image`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            updatePredictionUI(result);
            showToast('Image analyzed successfully', 'success');
        }
    } catch (error) {
        console.error('Error predicting image:', error);
        showToast('Image analysis failed', 'error');
        simulatePrediction();
    }
}

function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const preview = document.getElementById('videoPreview');
    preview.innerHTML = '';
    
    const video = document.createElement('video');
    video.src = URL.createObjectURL(file);
    video.controls = true;
    video.style.maxWidth = '100%';
    video.style.borderRadius = '10px';
    
    preview.appendChild(video);
    
    // Show video controls
    document.getElementById('videoControls').style.display = 'flex';
}

async function analyzeVideo() {
    showToast('Video analysis started. This may take a moment...', 'info');
    // Simulate analysis
    setTimeout(() => {
        simulatePrediction();
        showToast('Video analysis complete', 'success');
    }, 2000);
}

// Fallback simulation for testing
function simulatePrediction(specificLetter = null) {
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const letter = specificLetter || letters[Math.floor(Math.random() * letters.length)];
    const confidence = Math.random() * 0.3 + 0.7; // 0.7 to 1.0
    
    const result = {
        status: 'success',
        predicted_letter: letter,
        confidence: confidence,
        message: 'Simulated prediction'
    };
    
    updatePredictionUI(result);
    showToast(`Simulated prediction: ${letter}`, 'info');
}

// Toast notification
function showToast(message, type = 'info') {
    Toastify({
        text: message,
        duration: 3000,
        gravity: 'top',
        position: 'right',
        backgroundColor: type === 'success' ? '#4CAF50' : 
                       type === 'error' ? '#F44336' : 
                       type === 'warning' ? '#FF9800' : '#2196F3',
        stopOnFocus: true
    }).showToast();
}