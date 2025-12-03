"""
Model inference module for gesture classification.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch.nn.functional as F

from config import (
    MODEL_ARCH, NUM_CLASSES, KEEP_CLASSES, DROPOUT,
    RESIZE_SIZE, CLIP_LEN, K400_MEAN, K400_STD,
    CLASS_INDEX_TO_ID
)
from train import TransferModel_1, get_device


# K400 normalization tensors
K400_MEAN_TENSOR = torch.tensor(K400_MEAN).view(3, 1, 1, 1)
K400_STD_TENSOR = torch.tensor(K400_STD).view(3, 1, 1, 1)


def resize_letterbox(video, target_size=(112, 112)):
    """
    Resize video with letterbox (preserve aspect ratio).
    video: (T, C, H, W) or (C, T, H, W)
    target_size: (new_h, new_w)
    returns video resized with preserved aspect ratio and padding
    """
    is_CT = video.shape[0] == 3  # detect layout
    if is_CT:  # (C, T, H, W)
        video = video.permute(1, 0, 2, 3)  # -> (T, C, H, W)

    T, C, H, W = video.shape
    new_h, new_w = target_size
    scale = min(new_h / H, new_w / W)
    resized_h, resized_w = int(round(H * scale)), int(round(W * scale))

    # resize first
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)

    # pad to target
    pad_h = new_h - resized_h
    pad_w = new_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom))  # pad=(left,right,top,bottom)

    if is_CT:
        video = video.permute(1, 0, 2, 3)  # back to (C, T, H, W)
    return video


def preprocess_frames(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Preprocess frames for model inference.
    
    Args:
        frames: List of 16 frames, each as numpy array (H, W, C) uint8 [0-255]
    
    Returns:
        Preprocessed tensor of shape (C, T, H, W) = (3, 16, 112, 112)
    """
    if len(frames) != CLIP_LEN:
        raise ValueError(f"Expected {CLIP_LEN} frames, got {len(frames)}")
    
    if len(frames) == 0:
        raise ValueError("Empty frame list")
    
    # Convert to tensor: list of (H, W, C) -> (T, C, H, W)
    # Each frame is (H, W, C), convert to (C, H, W) then stack
    video_list = []
    for frame in frames:
        # frame is (H, W, C), convert to (C, H, W) and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        video_list.append(frame_tensor)
    
    # Stack to (T, C, H, W)
    video = torch.stack(video_list, dim=0)  # (T, C, H, W)
    
    # Resize with letterbox
    video = resize_letterbox(video, target_size=RESIZE_SIZE)  # (T, C, 112, 112)
    
    # Convert to (C, T, H, W) format
    video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
    
    # Normalize with K400 mean/std
    mean = K400_MEAN_TENSOR.to(dtype=video.dtype, device=video.device)
    std = K400_STD_TENSOR.to(dtype=video.dtype, device=video.device)
    video = (video - mean) / std
    
    return video


def load_model(checkpoint_path: str) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
    
    Returns:
        Tuple of (model, device)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Get device
    device = get_device(force_cpu=False, use_gpu=False)
    
    # Load checkpoint first to check number of classes
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine number of classes from checkpoint
    checkpoint_num_classes = NUM_CLASSES  # Default to our config
    if isinstance(checkpoint, dict):
        if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
            checkpoint_num_classes = checkpoint['args'].get('num_classes', NUM_CLASSES)
        elif 'num_classes' in checkpoint:
            checkpoint_num_classes = checkpoint['num_classes']
    
    # Load model architecture with checkpoint's number of classes
    model, _ = TransferModel_1(MODEL_ARCH, checkpoint_num_classes, pretrained=False, dropout=DROPOUT)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'], strict=True)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint, strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    
    model.eval()
    model = model.to(device)
    
    return model, device


def predict_gesture(model: torch.nn.Module, device: torch.device, frames: List[np.ndarray], model_num_classes: int = None) -> Dict:
    """
    Predict gesture from frames.
    
    Args:
        model: Loaded model
        device: Device to run inference on
        frames: List of 16 frames, each as numpy array (H, W, C) uint8
        model_num_classes: Number of classes the model was trained with (for mapping)
    
    Returns:
        Dictionary with:
            - predicted_class: int (0-9, mapped to our 10 classes)
            - class_id: int (10-27, original class ID)
            - confidence: float (0-1)
            - all_scores: List[float] (confidence scores for all 10 classes)
    """
    # Preprocess frames
    video_tensor = preprocess_frames(frames)
    video_tensor = video_tensor.unsqueeze(0).to(device)  # Add batch dimension: (1, C, T, H, W)
    
    # Run inference
    with torch.no_grad():
        logits = model(video_tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)  # (1, num_classes)
    
    # Handle class mapping if model has 27 classes but we only care about 10
    if model_num_classes == 27:
        # Map from 27 classes (0-26, representing classes 1-27) to our 10 classes
        # KEEP_CLASSES = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
        # In 0-indexed: [9, 10, 15, 16, 17, 22, 23, 24, 25, 26]
        keep_indices = [cls_id - 1 for cls_id in KEEP_CLASSES]  # Convert to 0-indexed
        
        # Extract probabilities for the 10 classes we care about
        keep_probs = probs[0, keep_indices]  # (10,)
        # Renormalize
        keep_probs = keep_probs / keep_probs.sum()
        
        # Get prediction from the 10 classes
        predicted_idx_10 = keep_probs.argmax().item()  # 0-9
        confidence = keep_probs[predicted_idx_10].item()
        class_id = CLASS_INDEX_TO_ID[predicted_idx_10]
        all_scores = keep_probs.cpu().numpy().tolist()
    else:
        # Model already has 10 classes
        predicted_idx_10 = logits.argmax(dim=1).item()  # 0-9
        confidence = probs[0, predicted_idx_10].item()
        class_id = CLASS_INDEX_TO_ID[predicted_idx_10]
        all_scores = probs[0].cpu().numpy().tolist()
    
    return {
        'predicted_class': int(predicted_idx_10),
        'class_id': int(class_id),
        'confidence': float(confidence),
        'all_scores': all_scores
    }


class ModelInference:
    """
    Wrapper class for model inference.
    """
    def __init__(self, checkpoint_path: str):
        """
        Initialize model inference.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        self.model, self.device = load_model(checkpoint_path)
        # Determine number of classes from model's final layer
        if hasattr(self.model, 'fc'):
            # R3D-18 has fc as Sequential with Linear at index 1
            if isinstance(self.model.fc, nn.Sequential):
                self.model_num_classes = self.model.fc[1].out_features
            else:
                self.model_num_classes = self.model.fc.out_features
        else:
            self.model_num_classes = NUM_CLASSES
    
    def predict(self, frames: List[np.ndarray]) -> Dict:
        """
        Predict gesture from frames.
        
        Args:
            frames: List of 16 frames, each as numpy array (H, W, C) uint8
        
        Returns:
            Dictionary with prediction results
        """
        return predict_gesture(self.model, self.device, frames, self.model_num_classes)

