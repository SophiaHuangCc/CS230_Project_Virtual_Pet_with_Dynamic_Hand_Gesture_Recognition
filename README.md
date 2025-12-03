# Dynamic Hand Gesture Recognition - Virtual Pet Web Application

This document explains how to set up the local development environment for the **Dynamic Hand Gesture Recognition for Digital Pet Keeping** project, including the FastAPI backend and web interface.

We use **mamba/miniconda** for environment management.  
Development is done locally on CPU (MacBook Pro), and training will later run on AWS GPU instances using provided credits.

---

## 1. Create and Activate the Environment

```bash
# Create a new environment named cs230 with Python 3.10
mamba create -n cs230 python=3.10 -y

# Activate the environment
mamba activate cs230
```

---

## 2. Install Dependencies (CPU-Only Setup for macOS)

```bash
# Install PyTorch CPU version
mamba install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install supporting libraries
mamba install opencv tqdm pillow matplotlib pandas numpy -y

# Install web framework dependencies
pip install fastapi uvicorn[standard] python-multipart httpx pytest
```

If you see dependency solver issues, try:
```bash
mamba update --all -y
```

**Alternative**: Install all Python dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 3. Verify Installation

Run the following commands to confirm your installation is correct:

```bash
# Show PyTorch environment info
python -m torch.utils.collect_env

# Check CUDA availability (should be False on Mac)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check versions for torchvision and OpenCV
python -c "import torchvision, cv2; print('torchvision', torchvision.__version__, '| OpenCV', cv2.__version__)"
```

Expected output on macOS:
```
CUDA available: False
torchvision 0.xx.x | OpenCV 4.x.x
```

---

## 4. Optional: Validate Data Loader Scripts

Once dependencies are installed and the dataset/CSV are available, verify the preprocessing and data loading scripts work correctly:

```bash
python build_hgd_index.py
python sanity_hgd_loader.py
```

Expected output:
```
Wrote hgd_index.json with 1701 items
Clip shape (C,T,H,W): (3,32,112,112)
Wrote preview_example.png
```

If you see these messages and a preview image appears, your dataset pipeline is working properly.

---

## 5. Notes and Tips

- **Activate environment each session**:
  ```bash
  mamba activate cs230
  ```
- **Export the environment**:
  ```bash
  conda env export > environment.yml
  ```
- **Recreate from environment file**:
  ```bash
  mamba env create -f environment.yml
  ```
- **When training on AWS GPU**, install CUDA-enabled PyTorch instead:
  ```bash
  mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
  ```

---

## 6. Repository Structure

```
CS230_Project_Virtual_Pet_with_Dynamic_Hand_Gesture_Recognition/
│
├── data/
│   ├── hand_gesture_dataset_videos/
│   │   ├── class_01/ ... class_27/
│   ├── hand_gesture_timing_stats.csv
│
├── runs/
│   └── train/
│       └── exp*/weights/best.pt  # Model checkpoints
│
├── static/
│   ├── animations/              # Animation videos for gestures
│   │   ├── gesture_10.mp4       # (add animation files here)
│   │   ├── gesture_11.mp4
│   │   └── ...
│   ├── index.html               # Frontend web interface
│   ├── app.js                   # Frontend JavaScript (webcam, API calls)
│   └── style.css                # Frontend styles
│
├── tests/
│   ├── test_app.py              # FastAPI endpoint tests
│   ├── test_inference.py        # Model inference tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   └── conftest.py              # Pytest configuration
│
├── build_hgd_index.py           # Build dataset index
├── dataset_hgd_preproc.py       # Dataset preprocessing
├── train.py                     # Model training script
├── app.py                       # FastAPI backend application
├── inference.py                 # Model inference module
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── BACKEND_IMPLEMENTATION.md    # Backend implementation details
└── README.md
```

---

## 7. Training the Model

```bash
python train.py --pretrained --epochs 40 --batch_size 16 --clip_len 16 --resize 112 --arch r3d_18 --freeze_until layer3
```

The trained model checkpoint will be saved in `runs/train/exp*/weights/best.pt`.

---

## 8. Running the Web Application

### 8.1 Start the FastAPI Server

```bash
# Make sure you're in the project root directory
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`.

### 8.2 Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

### 8.3 Using the Application

1. **Start Camera**: Click the "Start Camera" button and allow camera permissions
2. **Perform Gestures**: Make hand gestures in front of your webcam
3. **View Results**: 
   - The detected gesture and confidence will be displayed
   - The corresponding animation will play in the pet display area
   - The gesture class list will highlight the active gesture

### 8.4 Adding Animation Files

Place animation video files in `static/animations/` with the following naming convention:
- `gesture_10.mp4`, `gesture_11.mp4`, `gesture_16.mp4`, etc.
- Supported formats: `.mp4`, `.webm`, `.gif`, `.mov`
- The application will automatically detect and play the appropriate animation

---

## 9. API Endpoints

### 9.1 Health Check
```http
GET /health
```
Returns:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 9.2 Predict Gesture
```http
POST /predict
Content-Type: application/json

{
  "frames": ["base64_encoded_image1", "base64_encoded_image2", ...]
}
```

Returns:
```json
{
  "predicted_class": 0,
  "class_id": 10,
  "confidence": 0.95,
  "all_scores": [0.95, 0.02, ...]
}
```

### 9.3 Get Animation
```http
GET /animations/{gesture_id}
```
Returns the animation video file for the specified gesture ID (10, 11, 16, 17, 18, 23, 24, 25, 26, 27).

### 9.4 Interactive API Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 10. Testing

### 10.1 Run All Tests

```bash
pytest tests/ -v
```

### 10.2 Run Specific Test Suites

```bash
# Test preprocessing
pytest tests/test_preprocessing.py -v

# Test inference
pytest tests/test_inference.py -v

# Test API endpoints
pytest tests/test_app.py -v
```

### 10.3 Basic Backend Verification

```bash
python test_backend_basic.py
```

---

## 11. Application Architecture

### Backend (FastAPI)
- **`app.py`**: Main FastAPI application with endpoints
- **`inference.py`**: Model loading and inference logic
- **`config.py`**: Configuration settings (model paths, class mappings, preprocessing params)

### Frontend (HTML/CSS/JavaScript)
- **`static/index.html`**: Main web interface
- **`static/app.js`**: Webcam capture, frame buffering, API communication
- **`static/style.css`**: UI styling

### Key Features
- **Frame Buffering**: Collects 16 frames at ~30fps before inference
- **Real-time Processing**: Sends frames to backend every 500ms
- **Animation Playback**: Automatically loads and plays gesture-specific animations
- **Visual Feedback**: Shows detected gesture, confidence, and highlights active gesture class

---

## 12. Model Configuration

The application uses:
- **Architecture**: R3D-18 (3D ResNet)
- **Input**: 16 frames, resized to 112x112
- **Classes**: 10 gesture classes (mapped from original IDs: 10, 11, 16, 17, 18, 23, 24, 25, 26, 27)
- **Normalization**: Kinetics-400 mean/std
- **Preprocessing**: Letterbox resize (preserves aspect ratio)

---

## 13. Troubleshooting

### Model Not Loading
- Ensure a model checkpoint exists at `runs/train/exp*/weights/best.pt`
- Check the console output for model loading errors
- Verify the model architecture matches the checkpoint

### Camera Not Working
- Ensure camera permissions are granted in your browser
- Try using HTTPS (some browsers require secure context for camera access)
- Check browser console for errors

### Animations Not Playing
- Verify animation files exist in `static/animations/`
- Check file naming: `gesture_{id}.mp4` (e.g., `gesture_10.mp4`)
- Ensure video format is supported (mp4, webm, gif, mov)

### API Errors
- Check that the server is running: `http://localhost:8000/health`
- Verify model is loaded (check `/health` endpoint)
- Check browser console and server logs for error messages

---

## 14. Development Notes

- The application uses **Test-Driven Development (TDD)** principles
- All core components have corresponding test files
- See `BACKEND_IMPLEMENTATION.md` for detailed backend documentation
- Frontend automatically handles frame buffering and throttles inference requests

---

## 15. Authors

CS230 - Dynamic Hand Gesture Recognition Team  
Stanford University  
Fall 2025

---

**End of README**
