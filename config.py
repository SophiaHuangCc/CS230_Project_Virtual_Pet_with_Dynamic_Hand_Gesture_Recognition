"""
Configuration settings for the virtual pet web application.
"""
from pathlib import Path
import torch

# Model configuration
MODEL_ARCH = "r3d_18"
NUM_CLASSES = 10
KEEP_CLASSES = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]  # Original class IDs
DROPOUT = 0.5

# Model checkpoint path (will search for best.pt in runs/train/exp*)
DEFAULT_MODEL_PATH = "runs/train/exp14/weights/best.pt"

# Checkpoint selection strategy: "most_recent", "best_accuracy", "exp14", "exp23", etc.
# Set to a specific experiment (e.g., "exp14") to use that checkpoint, or use a strategy
CHECKPOINT_SELECTION = "most_recent"  # Options: "most_recent", "best_accuracy", or "exp<N>"

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


def _get_checkpoint_val_acc(checkpoint_path: Path):
    """Extract validation accuracy from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            return checkpoint.get('val_acc', None)
    except Exception:
        pass
    return None


def find_model_checkpoint():
    """
    Find the best model checkpoint in runs/train/exp*/weights/best.pt.
    
    Selection strategies:
    - "most_recent": Returns the most recently modified checkpoint
    - "best_accuracy": Returns the checkpoint with highest validation accuracy
    - "exp<N>": Returns a specific experiment checkpoint (e.g., "exp14")
    
    Returns the path to the selected checkpoint, or DEFAULT_MODEL_PATH.
    """
    runs_dir = Path("runs/train")
    if not runs_dir.exists():
        return DEFAULT_MODEL_PATH
    
    # Find all best.pt files
    checkpoints = list(runs_dir.glob("exp*/weights/best.pt"))
    if not checkpoints:
        return DEFAULT_MODEL_PATH
    
    # Handle specific experiment selection
    if CHECKPOINT_SELECTION.startswith("exp"):
        specific_path = runs_dir / CHECKPOINT_SELECTION / "weights" / "best.pt"
        if specific_path.exists():
            return str(specific_path)
        else:
            print(f"Warning: {CHECKPOINT_SELECTION} checkpoint not found, falling back to most recent")
    
    # Handle "best_accuracy" strategy
    if CHECKPOINT_SELECTION == "best_accuracy":
        best_checkpoint = None
        best_acc = -1.0
        for ckpt_path in checkpoints:
            val_acc = _get_checkpoint_val_acc(ckpt_path)
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                best_checkpoint = ckpt_path
        if best_checkpoint:
            return str(best_checkpoint)
        # Fall through to most_recent if no accuracies found
    
    # Default: most_recent or fallback
    return str(max(checkpoints, key=lambda p: p.stat().st_mtime))


