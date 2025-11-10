import os
from pathlib import Path
import torchvision.io as io
from tqdm import tqdm

root = Path("/Users/sophiahuang/Desktop/CS230/project/project_ws/data/hand_gesture_dataset_videos")

summary = []

for path in tqdm(list(root.rglob("*.avi"))):
    try:
        video, _, info = io.read_video(str(path), pts_unit="sec")
        T, H, W, C = video.shape
        summary.append({
            "path": str(path),
            "frames": T,
            "height": H,
            "width": W,
            "fps": info["video_fps"]
        })
    except Exception as e:
        print(f"Error reading {path}: {e}")

print(f"\nTotal videos scanned: {len(summary)}")
print("Sample entry:", summary[0])

# Compute dataset-level stats
import pandas as pd
df = pd.DataFrame(summary)
print("\nFrame statistics across dataset:")
print(df[["frames", "height", "width"]].describe())