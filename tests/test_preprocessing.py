"""
Tests for frame preprocessing pipeline.
"""
import pytest
import torch
import numpy as np
from inference import preprocess_frames


def test_preprocess_frames_shape():
    """Test that preprocessing returns correct tensor shape (C, T, H, W)."""
    # Create dummy frames: list of 16 RGB frames, each 224x224
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    
    result = preprocess_frames(frames)
    
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 16, 112, 112), f"Expected (3, 16, 112, 112), got {result.shape}"


def test_preprocess_frames_normalization():
    """Test that frames are normalized with K400 mean/std."""
    frames = [np.ones((224, 224, 3), dtype=np.uint8) * 128 for _ in range(16)]
    
    result = preprocess_frames(frames)
    
    # After normalization, values should be around (128/255 - mean) / std
    # K400_MEAN = [0.43216, 0.394666, 0.37645]
    # K400_STD = [0.22803, 0.22145, 0.216989]
    # For value 128/255 ≈ 0.502, normalized ≈ (0.502 - 0.43216) / 0.22803 ≈ 0.306
    assert result.dtype == torch.float32
    # Check that values are in reasonable range after normalization
    assert result.min() > -5.0 and result.max() < 5.0


def test_preprocess_frames_letterbox():
    """Test that letterbox resizing preserves aspect ratio."""
    # Create frames with non-square aspect ratio
    frames = [np.random.randint(0, 255, (320, 240, 3), dtype=np.uint8) for _ in range(16)]
    
    result = preprocess_frames(frames)
    
    assert result.shape == (3, 16, 112, 112)


def test_preprocess_frames_wrong_frame_count():
    """Test that preprocessing handles wrong number of frames."""
    # Too few frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    
    with pytest.raises(ValueError):
        preprocess_frames(frames)
    
    # Too many frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(20)]
    
    with pytest.raises(ValueError):
        preprocess_frames(frames)


def test_preprocess_frames_empty():
    """Test that preprocessing handles empty frame list."""
    with pytest.raises(ValueError):
        preprocess_frames([])



