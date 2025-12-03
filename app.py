"""
FastAPI application for virtual pet web app.
"""
import base64
import io
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import find_model_checkpoint, ANIMATIONS_DIR, CLIP_LEN
from inference import ModelInference

# Global model instance (loaded on startup)
model_inference: Optional[ModelInference] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on application startup."""
    global model_inference
    try:
        model_path = find_model_checkpoint()
        print(f"Loading model from: {model_path}")
        model_inference = ModelInference(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        model_inference = None
    yield
    # Cleanup on shutdown (if needed)
    model_inference = None


app = FastAPI(title="Virtual Pet Gesture Recognition API", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web application."""
    index_path = Path("static/index.html")
    if index_path.exists():
        return index_path.read_text()
    else:
        raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model_inference is not None}


@app.post("/predict")
async def predict(frames: dict):
    """
    Predict gesture from video frames.
    
    Request body:
        {
            "frames": [base64_encoded_image1, base64_encoded_image2, ...]
        }
    
    Returns:
        {
            "predicted_class": int (0-9),
            "class_id": int (10-27, original class ID),
            "confidence": float (0-1),
            "all_scores": [float, ...] (10 scores)
        }
    """
    # Validate request structure first (before checking model)
    if "frames" not in frames:
        raise HTTPException(status_code=400, detail="Missing 'frames' field in request")
    
    frame_strings = frames["frames"]
    
    if len(frame_strings) != CLIP_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {CLIP_LEN} frames, got {len(frame_strings)}"
        )
    
    # Decode base64 frames to numpy arrays (validate before checking model)
    decoded_frames = []
    for i, frame_str in enumerate(frame_strings):
        try:
            # Decode base64
            img_data = base64.b64decode(frame_str)
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to numpy array (H, W, C) uint8
            frame_array = np.array(img, dtype=np.uint8)
            decoded_frames.append(frame_array)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode frame {i}: {str(e)}"
            )
    
    # Check if model is loaded after all validation
    if model_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Run inference
    try:
        result = model_inference.predict(decoded_frames)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/animations/{gesture_id}")
async def get_animation(gesture_id: int):
    """
    Serve animation video for a gesture.
    
    Args:
        gesture_id: Original class ID (10-27)
    
    Returns:
        Video file or 404 if not found
    """
    # Check if gesture_id is valid
    valid_ids = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
    if gesture_id not in valid_ids:
        raise HTTPException(status_code=404, detail=f"Invalid gesture_id: {gesture_id}")
    
    # Look for animation file
    animation_path = ANIMATIONS_DIR / f"gesture_{gesture_id}.mp4"
    
    if not animation_path.exists():
        # Try alternative formats
        for ext in ['.webm', '.gif', '.mov']:
            alt_path = ANIMATIONS_DIR / f"gesture_{gesture_id}{ext}"
            if alt_path.exists():
                animation_path = alt_path
                break
        else:
            raise HTTPException(status_code=404, detail=f"Animation not found for gesture_id: {gesture_id}")
    
    return FileResponse(animation_path)


# Mount static files (for frontend)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


