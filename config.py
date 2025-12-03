"""
Configuration settings for the virtual pet web application.
"""
from pathlib import Path

# Model configuration
MODEL_ARCH = "r3d_18"
NUM_CLASSES = 10
KEEP_CLASSES = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]  # Original class IDs
DROPOUT = 0.5

# Model checkpoint path (will search for best.pt in runs/train/exp*)
DEFAULT_MODEL_PATH = "runs/train/exp14/weights/best.pt"

# Preprocessing configuration
CLIP_LEN = 16  # Number of frames per clip
RESIZE_SIZE = (112, 112)  # (H, W) spatial size

# K400 normalization constants (from dataset_hgd_preproc.py)
K400_MEAN = [0.43216, 0.394666, 0.37645]
K400_STD = [0.22803, 0.22145, 0.216989]

# Frame buffer configuration
FRAME_BUFFER_SIZE = 16
INFERENCE_FPS = 30.0

# Animation configuration
ANIMATIONS_DIR = Path("static/animations")

# Class mapping: model output index (0-9) -> original class ID
CLASS_INDEX_TO_ID = {i: cls_id for i, cls_id in enumerate(KEEP_CLASSES)}
CLASS_ID_TO_INDEX = {cls_id: i for i, cls_id in enumerate(KEEP_CLASSES)}

def find_model_checkpoint():
    """
    Find the best model checkpoint in runs/train/exp*/weights/best.pt.
    Returns the path to the most recent checkpoint, or DEFAULT_MODEL_PATH.
    """
    runs_dir = Path("runs/train")
    if not runs_dir.exists():
        return DEFAULT_MODEL_PATH
    
    # Find all best.pt files
    checkpoints = list(runs_dir.glob("exp*/weights/best.pt"))
    if not checkpoints:
        return DEFAULT_MODEL_PATH
    
    # Return the most recently modified one
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


