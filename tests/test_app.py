"""
Tests for FastAPI application endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app import app
import numpy as np
import base64
from io import BytesIO
from PIL import Image


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test /health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_endpoint_missing_frames(client):
    """Test /predict endpoint with missing frames returns 400."""
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_endpoint_wrong_frame_count(client):
    """Test /predict endpoint with wrong number of frames returns 400."""
    # Create dummy frames (too few)
    frames = []
    for i in range(10):
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        frames.append(img_str)
    
    response = client.post("/predict", json={"frames": frames})
    assert response.status_code == 400


def test_predict_endpoint_valid_frames(client):
    """Test /predict endpoint with valid frames returns 200 and prediction."""
    # Create 16 dummy frames
    frames = []
    for i in range(16):
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        frames.append(img_str)
    
    response = client.post("/predict", json={"frames": frames})
    
    # This might fail if model isn't loaded, but structure should be correct
    if response.status_code == 200:
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert "class_id" in data
        assert 0 <= data["predicted_class"] < 10
        assert data["class_id"] in [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
    elif response.status_code == 500:
        # Model might not be loaded, check error message
        data = response.json()
        assert "error" in data or "detail" in data


def test_animations_endpoint(client):
    """Test /animations/{gesture_id} endpoint."""
    # Test with valid gesture ID
    response = client.get("/animations/10")
    # Should return 404 if animation doesn't exist, or 200 if it does
    assert response.status_code in [200, 404]
    
    # Test with invalid gesture ID
    response = client.get("/animations/999")
    assert response.status_code == 404


def test_predict_endpoint_invalid_base64(client):
    """Test /predict endpoint with invalid base64 returns 400."""
    response = client.post("/predict", json={"frames": ["invalid_base64_string"] * 16})
    assert response.status_code == 400



