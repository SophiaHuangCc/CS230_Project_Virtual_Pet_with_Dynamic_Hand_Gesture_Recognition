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

// Gesture ID to action name mapping
const GESTURE_ACTIONS = {
    10: "Action 1",
    11: "Action 2",
    16: "Action 3",
    17: "Action 4",
    18: "Action 5",
    23: "Action 6",
    24: "Action 7",
    25: "Action 8",
    26: "Action 9",
    27: "Action 10"
};

// DOM elements
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const stopBtn = document.getElementById('stopBtn');
const recordBtn = document.getElementById('recordBtn');
const status = document.getElementById('status');
const gestureInfo = document.getElementById('gestureInfo');
const petAnimation = document.getElementById('petAnimation');
const petPlaceholder = document.getElementById('petPlaceholder');
const predictionInfo = document.getElementById('predictionInfo');
const gestureList = document.getElementById('gestureList');
const currentAction = document.getElementById('currentAction');
const actionStatus = document.getElementById('actionStatus');
const actionDetails = document.getElementById('actionDetails');

// State
let stream = null;
let frameBuffer = [];
let isCapturing = false;
let lastInferenceTime = 0;
let currentGestureId = null;
let currentGestureConfidence = null; // Store confidence for display during animation
let animationInterval = null;
let isAnimationPlaying = false; // Track if animation is currently playing
let mediaRecorder = null;
let isRecording = false;
let recordedChunks = [];

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
    // Skip inference if animation is currently playing
    if (isAnimationPlaying) {
        return;
    }

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

    // Load and play animation if gesture changed and no animation is currently playing
    if (currentGestureId !== class_id && !isAnimationPlaying) {
        currentGestureId = class_id;
        currentGestureConfidence = confidence;
        loadAnimation(class_id);
    }
}

// Load animation for a gesture
function loadAnimation(gestureId) {
    // Backend handles format detection, just call the endpoint
    const animationUrl = `/animations/${gestureId}`;
    
    // Set flag to pause inference while animation plays (but don't update display yet)
    isAnimationPlaying = true;
    recordBtn.disabled = true; // Disable recording while animation plays
    
    // Handle animation loaded and ready to play
    petAnimation.onloadeddata = () => {
        // Show "Gesture Detected" with gesture info while animation is playing
        const actionName = GESTURE_ACTIONS[gestureId] || `Gesture ${gestureId}`;
        const confidenceText = currentGestureConfidence 
            ? ` (${(currentGestureConfidence * 100).toFixed(1)}% confidence)`
            : '';
        currentAction.className = 'current-action playing';
        actionStatus.textContent = '🎯 Gesture Detected';
        actionStatus.className = 'action-status playing';
        actionDetails.innerHTML = `
            <div class="action-name">${actionName}</div>
            <div class="gesture-id">Hand Gesture: ${gestureId}${confidenceText}</div>
            <div class="detection-status">⚠️ Gesture detection paused - Animation playing</div>
        `;
        status.textContent = 'Playing animation...';
        
        petPlaceholder.classList.add('hidden');
        petAnimation.classList.add('show');
        petAnimation.play().catch(err => {
            console.error('Error playing animation:', err);
            // If play fails, show placeholder and resume detection
            petPlaceholder.classList.remove('hidden');
            petAnimation.classList.remove('show');
            isAnimationPlaying = false;
            recordBtn.disabled = false; // Re-enable recording if play fails
            status.textContent = 'Camera active';
            updateActionDisplayForDetection();
        });
    };
    
    // Handle animation ended - resume gesture detection
    petAnimation.onended = () => {
        isAnimationPlaying = false;
        recordBtn.disabled = false; // Re-enable recording after animation
        status.textContent = 'Camera active - Ready for next gesture';
        updateActionDisplayForDetection();
        // Clear the gesture info overlay after a short delay
        setTimeout(() => {
            gestureInfo.classList.remove('show');
        }, 1000);
    };
    
    // Handle animation errors
    petAnimation.onerror = () => {
        console.warn(`Animation not found for gesture ${gestureId}`);
        petPlaceholder.classList.remove('hidden');
        petAnimation.classList.remove('show');
        isAnimationPlaying = false;
        recordBtn.disabled = false; // Re-enable recording if animation fails
        status.textContent = 'Camera active';
        updateActionDisplayForDetection();
    };
    
    // Load the video
    petAnimation.src = animationUrl;
    petAnimation.load(); // Force reload
}

// Update action display when detecting gestures
function updateActionDisplayForDetection() {
    currentAction.className = 'current-action detecting';
    actionStatus.textContent = '👁️ Detecting Gestures';
    actionStatus.className = 'action-status detecting';
    actionDetails.innerHTML = `
        <div class="detection-status">Ready to detect hand gestures</div>
    `;
}

// Stop webcam capture
function stopCapture() {
    isCapturing = false;

    // Stop any ongoing recording
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;

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
    currentGestureConfidence = null;
    isAnimationPlaying = false; // Reset animation state

    stopBtn.disabled = true;
    recordBtn.disabled = false;
    status.textContent = 'Camera stopped';
    gestureInfo.classList.remove('show');
    predictionInfo.innerHTML = '';
    
    // Reset action display
    currentAction.className = 'current-action';
    actionStatus.textContent = 'Waiting for gesture...';
    actionStatus.className = 'action-status';
    actionDetails.innerHTML = '';
    
    // Reset gesture list highlighting
    document.querySelectorAll('.gesture-item').forEach(item => {
        item.classList.remove('active');
    });

    // Hide animation
    petPlaceholder.classList.remove('hidden');
    petAnimation.classList.remove('show');
    petAnimation.src = '';
    // Remove event listeners to prevent memory leaks
    petAnimation.onended = null;
    petAnimation.onerror = null;
    petAnimation.onloadeddata = null;
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

// Extract frames from video blob
async function extractFramesFromVideo(videoBlob) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const url = URL.createObjectURL(videoBlob);
        video.src = url;
        video.muted = true;
        
        video.onloadedmetadata = () => {
            video.currentTime = 0;
            const frames = [];
            const duration = video.duration;
            const frameInterval = duration / CLIP_LEN; // Extract 16 frames evenly spaced
            
            let frameIndex = 0;
            
            const captureFrame = () => {
                if (frameIndex >= CLIP_LEN) {
                    URL.revokeObjectURL(url);
                    resolve(frames);
                    return;
                }
                
                // Set video time to capture frame at this position
                video.currentTime = frameIndex * frameInterval;
                
                video.onseeked = () => {
                    // Draw frame to canvas
                    const ctx = canvas.getContext('2d');
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.drawImage(video, 0, 0);
                    
                    // Convert to base64
                    const base64Frame = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
                    frames.push(base64Frame);
                    
                    frameIndex++;
                    captureFrame();
                };
            };
            
            captureFrame();
        };
        
        video.onerror = (error) => {
            URL.revokeObjectURL(url);
            reject(error);
        };
    });
}

// Record 4-second video and predict gesture
async function recordAndPredict() {
    if (isRecording || isAnimationPlaying) {
        return;
    }
    
    try {
        // Start camera if not already started
        if (!stream) {
            status.textContent = 'Requesting camera access...';
            recordBtn.disabled = true;
            
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });
                
                webcam.srcObject = stream;
                isCapturing = true;
                stopBtn.disabled = false;
                status.textContent = 'Camera started. Ready to record...';
            } catch (error) {
                console.error('Error accessing camera:', error);
                status.textContent = `Camera error: ${error.message}`;
                recordBtn.disabled = false;
                alert('Could not access camera. Please allow camera permissions and try again.');
                return;
            }
        }
        
        // Now start recording
        isRecording = true;
        recordBtn.disabled = true;
        recordBtn.textContent = 'Recording...';
        status.textContent = 'Recording 4 seconds...';
        
        // Pause continuous detection during recording (if it was running)
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        
        // Create MediaRecorder
        recordedChunks = [];
        const options = { mimeType: 'video/webm;codecs=vp9' };
        
        // Try different codecs if vp9 is not supported
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'video/webm;codecs=vp8';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'video/webm';
            }
        }
        
        mediaRecorder = new MediaRecorder(stream, options);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            try {
                status.textContent = 'Processing video...';
                
                // Create blob from recorded chunks
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                
                // Extract frames from video
                const frames = await extractFramesFromVideo(blob);
                
                if (frames.length !== CLIP_LEN) {
                    throw new Error(`Expected ${CLIP_LEN} frames, got ${frames.length}`);
                }
                
                // Send to backend for prediction
                status.textContent = 'Predicting gesture...';
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        frames: frames
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                handlePrediction(result);
                
            } catch (error) {
                console.error('Recording/prediction error:', error);
                status.textContent = `Error: ${error.message}`;
                alert(`Error processing recording: ${error.message}`);
            } finally {
                isRecording = false;
                recordBtn.disabled = false;
                recordBtn.textContent = 'Record (4s)';
                
                // Resume continuous detection if camera is still active
                if (isCapturing && !isAnimationPlaying) {
                    animationInterval = setInterval(() => {
                        addFrameToBuffer();
                        predictGesture();
                    }, FRAME_INTERVAL);
                }
            }
        };
        
        // Start recording
        mediaRecorder.start();
        
        // Stop recording after 4 seconds
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        }, 4000);
        
    } catch (error) {
        console.error('Recording error:', error);
        status.textContent = `Recording error: ${error.message}`;
        isRecording = false;
        recordBtn.disabled = false;
        recordBtn.textContent = 'Record (4s)';
    }
}

// Event listeners
stopBtn.addEventListener('click', stopCapture);
recordBtn.addEventListener('click', recordAndPredict);

// Initialize on page load
window.addEventListener('load', () => {
    initGestureList();
    checkBackendHealth();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCapture();
});

