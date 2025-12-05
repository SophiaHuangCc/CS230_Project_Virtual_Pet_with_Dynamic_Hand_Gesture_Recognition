"""
Tests for model inference module.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from inference import load_model, predict_gesture, ModelInference


def test_load_model_architecture():
    """Test that model loads with correct architecture (R3D-18)."""
    # This test will fail if model file doesn't exist, which is expected in TDD
    model_path = Path("runs/train/exp14/weights/best.pt")
    
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    
    # Force CPU for testing to avoid MPS issues
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    model, device = load_model(str(model_path))
    
    assert model is not None
    # Check that model has correct architecture
    assert hasattr(model, 'fc')
    
    # Check the actual output size by doing a forward pass
    # Use CPU to avoid MPS conv3d issues in tests
    model_cpu = model.cpu()
    dummy_input = torch.randn(1, 3, 16, 112, 112)
    with torch.no_grad():
        output = model_cpu(dummy_input)
    
    # The checkpoint has 27 classes, so output should be 27
    # (The mapping to 10 classes happens in predict_gesture)
    assert output.shape[1] in [10, 27], f"Expected 10 or 27 classes, got {output.shape[1]}"


def test_load_model_device():
    """Test that model loads on appropriate device."""
    model_path = Path("runs/train/exp14/weights/best.pt")
    
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    
    # Force CPU for testing to avoid MPS issues
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    model, device = load_model(str(model_path))
    
    assert device is not None
    assert isinstance(device, torch.device)


def test_predict_gesture_output_format():
    """Test that predict_gesture returns correct output format."""
    model_path = Path("runs/train/exp14/weights/best.pt")
    
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    
    # Force CPU for testing to avoid MPS issues
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    model, device = load_model(str(model_path))
    
    # Determine model's number of classes
    if hasattr(model, 'fc'):
        if isinstance(model.fc, nn.Sequential):
            model_num_classes = model.fc[1].out_features
        else:
            model_num_classes = model.fc.out_features
    else:
        model_num_classes = 27  # Default assumption
    
    # Create dummy preprocessed frames
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    
    result = predict_gesture(model, device, frames, model_num_classes)
    
    assert isinstance(result, dict)
    assert 'predicted_class' in result
    assert 'confidence' in result
    assert 'class_id' in result  # Original class ID (10-27)
    assert 'all_scores' in result
    
    # Check predicted_class is in range 0-9
    assert 0 <= result['predicted_class'] < 10
    # Check class_id is in KEEP_CLASSES [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
    assert result['class_id'] in [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
    # Check confidence is between 0 and 1
    assert 0.0 <= result['confidence'] <= 1.0
    # Check all_scores has 10 values
    assert len(result['all_scores']) == 10


def test_model_inference_class():
    """Test ModelInference class initialization and usage."""
    model_path = Path("runs/train/exp14/weights/best.pt")
    
    if not model_path.exists():
        pytest.skip(f"Model checkpoint not found at {model_path}")
    
    # Force CPU for testing to avoid MPS issues
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    inference = ModelInference(str(model_path))
    
    assert inference.model is not None
    assert inference.device is not None
    
    # Test prediction
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    result = inference.predict(frames)
    
    assert isinstance(result, dict)
    assert 'predicted_class' in result
    assert 'confidence' in result
    assert 'class_id' in result


def test_model_inference_missing_checkpoint():
    """Test that ModelInference handles missing checkpoint gracefully."""
    with pytest.raises(FileNotFoundError):
        ModelInference("nonexistent/path/best.pt")


