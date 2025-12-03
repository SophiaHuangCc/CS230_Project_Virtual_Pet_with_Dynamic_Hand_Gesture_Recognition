/**
 * Virtual Pet Web App - Frontend JavaScript
 * Handles webcam capture, frame buffering, API communication, and animation playback
 */

// Configuration
const CLIP_LEN = 16; // Number of frames per clip
const TARGET_FPS = 30; // Target frames per second
const FRAME_INTERVAL = 1000 / TARGET_FPS; // ~33ms per frame
const INFERENCE_INTERVAL = 500; // Run inference every 500ms

// Gesture class IDs (matching backend config)
const GESTURE_CLASSES = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27];

// DOM elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const status = document.getElementById('status');
const gestureInfo = document.getElementById('gestureInfo');
const petAnimation = document.getElementById('petAnimation');
const petPlaceholder = document.getElementById('petPlaceholder');
const predictionInfo = document.getElementById('predictionInfo');
const gestureList = document.getElementById('gestureList');

// State
let stream = null;
let frameBuffer = [];
let isCapturing = false;
let lastInferenceTime = 0;
let currentGestureId = null;
let animationInterval = null;

// Initialize gesture list
function initGestureList() {
    GESTURE_CLASSES.forEach((gestureId, index) => {
        const item = document.createElement('div');
        item.className = 'gesture-item';
        item.id = `gesture-${gestureId}`;
        item.innerHTML = `
            <div class="gesture-id">${gestureId}</div>
            <div class="gesture-label">Gesture ${index + 1}</div>
        `;
        gestureList.appendChild(item);
    });
}

// Convert video frame to base64
function frameToBase64(videoElement) {
    const ctx = canvas.getContext('2d');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    ctx.drawImage(videoElement, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8).split(',')[1]; // Remove data:image/jpeg;base64, prefix
}

// Add frame to buffer
function addFrameToBuffer() {
    if (!isCapturing || webcam.readyState !== webcam.HAVE_ENOUGH_DATA) {
        return;
    }

    const base64Frame = frameToBase64(webcam);
    frameBuffer.push(base64Frame);

    // Keep buffer size at CLIP_LEN
    if (frameBuffer.length > CLIP_LEN) {
        frameBuffer.shift(); // Remove oldest frame
    }

    // Update status
    status.textContent = `Buffering: ${frameBuffer.length}/${CLIP_LEN} frames`;
}

// Send frames to backend for prediction
async function predictGesture() {
    if (frameBuffer.length !== CLIP_LEN) {
        return;
    }

    const now = Date.now();
    if (now - lastInferenceTime < INFERENCE_INTERVAL) {
        return; // Throttle inference
    }
    lastInferenceTime = now;

    try {
        status.textContent = 'Processing...';
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frames: frameBuffer
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        handlePrediction(result);

    } catch (error) {
        console.error('Prediction error:', error);
        status.textContent = `Error: ${error.message}`;
    }
}

// Handle prediction result
function handlePrediction(result) {
    const { predicted_class, class_id, confidence, all_scores } = result;

    // Update status
    status.textContent = `Gesture detected!`;
    gestureInfo.textContent = `Gesture ${class_id} (${(confidence * 100).toFixed(1)}%)`;
    gestureInfo.classList.add('show');

    // Update prediction info
    predictionInfo.innerHTML = `
        <strong>Predicted:</strong> Gesture ${class_id}<br>
        <span class="confidence">Confidence: ${(confidence * 100).toFixed(1)}%</span>
    `;

    // Update gesture list highlighting
    document.querySelectorAll('.gesture-item').forEach(item => {
        item.classList.remove('active');
    });
    const activeItem = document.getElementById(`gesture-${class_id}`);
    if (activeItem) {
        activeItem.classList.add('active');
    }

    // Load and play animation if gesture changed
    if (currentGestureId !== class_id) {
        currentGestureId = class_id;
        loadAnimation(class_id);
    }
}

// Load animation for a gesture
function loadAnimation(gestureId) {
    // Backend handles format detection, just call the endpoint
    const animationUrl = `/animations/${gestureId}`;
    
    // Reset video element
    petAnimation.onloadeddata = () => {
        petPlaceholder.classList.add('hidden');
        petAnimation.classList.add('show');
        petAnimation.play().catch(err => {
            console.error('Error playing animation:', err);
            // If play fails, show placeholder
            petPlaceholder.classList.remove('hidden');
            petAnimation.classList.remove('show');
        });
    };
    
    petAnimation.onerror = () => {
        console.warn(`Animation not found for gesture ${gestureId}`);
        petPlaceholder.classList.remove('hidden');
        petAnimation.classList.remove('show');
    };
    
    // Load the video
    petAnimation.src = animationUrl;
    petAnimation.load(); // Force reload
}

// Start webcam capture
async function startCapture() {
    try {
        status.textContent = 'Requesting camera access...';
        
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });

        webcam.srcObject = stream;
        isCapturing = true;
        frameBuffer = [];
        lastInferenceTime = 0;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        status.textContent = 'Camera active';

        // Start frame capture loop
        animationInterval = setInterval(() => {
            addFrameToBuffer();
            predictGesture();
        }, FRAME_INTERVAL);

    } catch (error) {
        console.error('Error accessing camera:', error);
        status.textContent = `Camera error: ${error.message}`;
        alert('Could not access camera. Please allow camera permissions and try again.');
    }
}

// Stop webcam capture
function stopCapture() {
    isCapturing = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }

    webcam.srcObject = null;
    frameBuffer = [];
    currentGestureId = null;

    startBtn.disabled = false;
    stopBtn.disabled = true;
    status.textContent = 'Camera stopped';
    gestureInfo.classList.remove('show');
    predictionInfo.innerHTML = '';
    
    // Reset gesture list highlighting
    document.querySelectorAll('.gesture-item').forEach(item => {
        item.classList.remove('active');
    });

    // Hide animation
    petPlaceholder.classList.remove('hidden');
    petAnimation.classList.remove('show');
    petAnimation.src = '';
}

// Check backend health on load
async function checkBackendHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.model_loaded) {
            status.textContent = 'Ready - Model loaded';
        } else {
            status.textContent = 'Warning - Model not loaded';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        status.textContent = 'Error - Backend not available';
    }
}

// Event listeners
startBtn.addEventListener('click', startCapture);
stopBtn.addEventListener('click', stopCapture);

// Initialize on page load
window.addEventListener('load', () => {
    initGestureList();
    checkBackendHealth();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCapture();
});

