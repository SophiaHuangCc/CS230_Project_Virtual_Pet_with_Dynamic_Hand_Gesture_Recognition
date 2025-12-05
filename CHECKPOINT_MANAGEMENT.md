# Checkpoint Management Guide

## Overview

You now have multiple `best.pt` checkpoint files from different training experiments. This guide explains how to manage and select between them.

## Available Checkpoints

To see all available checkpoints and their metadata, run:

```bash
python inspect_checkpoints.py
```

This will show:
- Path and size of each checkpoint
- Validation accuracy
- Training hyperparameters (epochs, architecture, etc.)
- Modification date
- Which checkpoint has the best validation accuracy
- Which checkpoint is most recent

## Checkpoint Selection

The system automatically selects a checkpoint using the `CHECKPOINT_SELECTION` strategy in `config.py`. You can change this by editing the `CHECKPOINT_SELECTION` variable:

### Options:

1. **`"most_recent"`** (default)
   - Selects the checkpoint with the most recent modification time
   - Currently selects: `exp23/weights/best.pt`

2. **`"best_accuracy"`**
   - Selects the checkpoint with the highest validation accuracy
   - Best for production use if you want the best-performing model

3. **`"exp14"`** or **`"exp23"`** (specific experiment)
   - Forces use of a specific experiment's checkpoint
   - Useful when you know which model you want to use

### Example:

```python
# In config.py
CHECKPOINT_SELECTION = "best_accuracy"  # Use best performing model
# or
CHECKPOINT_SELECTION = "exp14"  # Force use of exp14
```

## Current Checkpoints

Based on your training experiments:

- **exp14**: 
  - Pretrained: true
  - Classes: 27 (mapped to 10)
  - Resize: 112x112
  - Epochs: 5
  
- **exp23**:
  - Pretrained: true  
  - Classes: 10 (direct)
  - Resize: 56x56
  - Epochs: 40

## Usage

The checkpoint is automatically loaded when:
- Starting the web application (`app.py`)
- Running inference (`inference.py`)
- Running tests

The selected checkpoint path is determined by `find_model_checkpoint()` in `config.py`.

## Inspecting a Specific Checkpoint

To inspect a single checkpoint:

```bash
python inspect_checkpoints.py runs/train/exp14/weights/best.pt
```

This will print detailed JSON information about that checkpoint.

