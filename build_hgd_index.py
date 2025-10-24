# build_hgd_index.py
import csv, json
from pathlib import Path

ROOT = Path("data/hand_gesture_dataset_videos")  # change if needed
CSV  = Path("data/hand_gesture_timing_stats.csv")

def class_to_label(c):  # CSV classes 1..27 -> labels 0..26
    return int(c) - 1

def find_three_videos(class_dir, user_folder):
    udir = class_dir / user_folder
    vids = sorted([p for p in udir.glob("*.avi")])
    assert len(vids) == 3, f"Expected 3 videos under {udir}, found {len(vids)}"
    return vids

items = []
with CSV.open() as f:
    r = csv.DictReader(f)
    for row in r:
        cls = f'class_{int(row["class"]):02d}'
        user = f'User{int(row["user"])}_'
        label = class_to_label(row["class"])
        vids = find_three_videos(ROOT/cls, user)

        for i in (1,2,3):
            s = int(row[f"start_frame_{i}"])
            e = int(row[f"end_frame_{i}"])
            items.append({
                "video_path": str(vids[i-1]),
                "label": label,
                "start_frame": s,
                "end_frame": e,
                "class": int(row["class"]),
                "user": int(row["user"]),
                "rep": i
            })

Path("hgd_index.json").write_text(json.dumps(items, indent=2))
print(f"Wrote hgd_index.json with {len(items)} items")  # expect 1701 = 567*3