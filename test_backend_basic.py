#!/usr/bin/env python3
"""
Basic test script to verify backend components work correctly.
Run this after installing dependencies to verify the implementation.
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import config
        print("✓ config imported")
        
        import inference
        print("✓ inference imported")
        
        import app
        print("✓ app imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_config():
    """Test config module."""
    print("\nTesting config...")
    try:
        from config import KEEP_CLASSES, NUM_CLASSES, CLIP_LEN, RESIZE_SIZE
        assert NUM_CLASSES == 10, f"Expected 10 classes, got {NUM_CLASSES}"
        assert len(KEEP_CLASSES) == 10, f"Expected 10 classes, got {len(KEEP_CLASSES)}"
        assert CLIP_LEN == 16, f"Expected 16 frames, got {CLIP_LEN}"
        assert RESIZE_SIZE == (112, 112), f"Expected (112, 112), got {RESIZE_SIZE}"
        print("✓ Config values correct")
        return True
    except Exception as e:
        print(f"✗ Config test error: {e}")
        return False


def test_model_checkpoint():
    """Test model checkpoint finding."""
    print("\nTesting model checkpoint...")
    try:
        from config import find_model_checkpoint
        path = find_model_checkpoint()
        print(f"✓ Found checkpoint: {path}")
        if Path(path).exists():
            print(f"✓ Checkpoint file exists")
        else:
            print(f"⚠ Checkpoint file not found (this is OK if model hasn't been trained yet)")
        return True
    except Exception as e:
        print(f"✗ Checkpoint test error: {e}")
        return False


def main():
    """Run all basic tests."""
    print("=" * 50)
    print("Basic Backend Tests")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_config())
    results.append(test_model_checkpoint())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ All basic tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())



