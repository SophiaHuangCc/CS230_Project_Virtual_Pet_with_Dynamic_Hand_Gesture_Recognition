# Dynamic Hand Gesture Recognition - Environment Setup

This document explains how to set up the local development environment for the **Dynamic Hand Gesture Recognition for Digital Pet Keeping** project.

We use **mamba/miniconda** for environment management.  
Development is done locally on CPU (MacBook Pro), and training will later run on AWS GPU instances using provided credits.

---

## 1. Create and Activate the Environment

```bash
# Create a new environment named cs230 with Python 3.10
mamba create -n cs230 python=3.10 -y

# Activate the environment
mamba activate cs230
```

---

## 2. Install Dependencies (CPU-Only Setup for macOS)

```bash
# Install PyTorch CPU version
mamba install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install supporting libraries
mamba install opencv tqdm pillow matplotlib pandas numpy -y
```

If you see dependency solver issues, try:
```bash
mamba update --all -y
```

---

## 3. Verify Installation

Run the following commands to confirm your installation is correct:

```bash
# Show PyTorch environment info
python -m torch.utils.collect_env

# Check CUDA availability (should be False on Mac)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check versions for torchvision and OpenCV
python -c "import torchvision, cv2; print('torchvision', torchvision.__version__, '| OpenCV', cv2.__version__)"
```

Expected output on macOS:
```
CUDA available: False
torchvision 0.xx.x | OpenCV 4.x.x
```

---

## 4. Optional: Validate Data Loader Scripts

Once dependencies are installed and the dataset/CSV are available, verify the preprocessing and data loading scripts work correctly:

```bash
python build_hgd_index.py
python sanity_hgd_loader.py
```

Expected output:
```
Wrote hgd_index.json with 1701 items
Clip shape (C,T,H,W): (3,32,112,112)
Wrote preview_example.png
```

If you see these messages and a preview image appears, your dataset pipeline is working properly.

---

## 5. Notes and Tips

- **Activate environment each session**:
  ```bash
  mamba activate cs230
  ```
- **Export the environment**:
  ```bash
  conda env export > environment.yml
  ```
- **Recreate from environment file**:
  ```bash
  mamba env create -f environment.yml
  ```
- **When training on AWS GPU**, install CUDA-enabled PyTorch instead:
  ```bash
  mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
  ```

---

## 6. Repository Structure (recommended)

```
hand_gesture_project/
│
├── data/
│   ├── hand_gestures_dataset_videos/
│   ├── hand_gesture_timing_stats.csv
│
├── build_hgd_index.py
├── sanity_hgd_loader.py
├── train_model.py
├── environment.yml
└── README.md
```

---

## 7. Authors

CS230 - Dynamic Hand Gesture Recognition Team  
Stanford University  
Fall 2025

---

**End of README**# CS230_Project_Virtual_Pet_with_Dynamic_Hand_Gesture_Recognition
