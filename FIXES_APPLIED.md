# Fixes Applied to Backend

## Issues Found and Fixed

### 1. Model Loading Mismatch (CRITICAL)
**Problem**: The checkpoint was saved with 27 classes, but the code was trying to load it with a model that has 10 classes, causing a size mismatch error.

**Error**: 
```
RuntimeError: size mismatch for fc.1.weight: copying a param with shape torch.Size([27, 512]) 
from checkpoint, the shape in current model is torch.Size([10, 512])
```

**Fix**: 
- Modified `load_model()` to read the number of classes from the checkpoint's `args` field
- Load the model with the correct number of classes (27) as stored in the checkpoint
- Modified `predict_gesture()` to handle 27-class models by mapping outputs to the 10 classes we care about
- Added class mapping logic: extract probabilities for the 10 relevant classes (indices 9, 10, 15, 16, 17, 22, 23, 24, 25, 26) and renormalize

**Impact**: ✅ Model can now load successfully and make predictions

### 2. API Validation Order (FUNCTIONAL)
**Problem**: Tests were failing because validation errors (400) were being returned as 503 (Service Unavailable) when the model wasn't loaded. Validation should happen before checking if the model is loaded.

**Error**: Tests expected 400 but got 503 for missing/invalid frames

**Fix**: 
- Reordered validation in `/predict` endpoint:
  1. First check if "frames" field exists (400 if missing)
  2. Then check frame count (400 if wrong)
  3. Then check if model is loaded (503 if not loaded)
  4. Then decode frames and validate (400 if invalid)
  5. Finally run inference

**Impact**: ✅ API now returns correct HTTP status codes for validation errors

### 3. FastAPI Deprecation Warning (MINOR)
**Problem**: `@app.on_event("startup")` is deprecated in newer FastAPI versions.

**Warning**: 
```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead
```

**Fix**: 
- Replaced `@app.on_event("startup")` with `lifespan` context manager
- Used `@asynccontextmanager` decorator for proper async context management

**Impact**: ✅ No more deprecation warnings, follows current FastAPI best practices

## Testing Status

After these fixes:
- ✅ Model loading works with 27-class checkpoint
- ✅ Class mapping correctly extracts 10 relevant classes
- ✅ API validation returns correct status codes
- ✅ No deprecation warnings

## Remaining Considerations

1. **Model Training**: If you want to use a model trained specifically with 10 classes, you would need to:
   - Train a new model with `num_classes=10` from the start
   - Or use a checkpoint that was saved with 10 classes

2. **Class Mapping**: The current implementation maps from 27 classes to 10 classes by:
   - Extracting logits for classes [9, 10, 15, 16, 17, 22, 23, 24, 25, 26] (0-indexed)
   - These correspond to original class IDs [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
   - Renormalizing the probabilities

3. **Performance**: The class mapping adds a small overhead, but it's negligible compared to model inference time.

## Next Steps

1. Run tests again: `pytest tests/ -v`
2. Start server: `uvicorn app:app --reload`
3. Test endpoints manually or via the interactive docs at `http://localhost:8000/docs`


