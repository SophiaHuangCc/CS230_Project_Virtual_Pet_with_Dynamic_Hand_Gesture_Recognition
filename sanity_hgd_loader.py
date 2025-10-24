# sanity_hgd_loader.py
import json, math, torch
from pathlib import Path
from torchvision.io import read_video
import torch.nn.functional as F
from PIL import Image

FPS = 30.0

def sample_indices(n_src, n_out):
    if n_src <= n_out:
        idx = list(range(n_src)) + [n_src-1]*(n_out-n_src)
    else:
        step = n_src / n_out
        idx = [int(i*step) for i in range(n_out)]
    return torch.tensor(idx)

def load_clip(item, T=32, size=(112,112)):
    s_f, e_f = item["start_frame"], item["end_frame"]
    start_sec = s_f / FPS
    end_sec   = max(start_sec + 1e-3, e_f / FPS)
    video, _, _ = read_video(item["video_path"], start_pts=start_sec, end_pts=end_sec, pts_unit='sec')
    # (t,h,w,c) uint8 -> float -> (t,c,h,w)
    video = (video.float()/255.0).permute(0,3,1,2)
    if size:
        video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
    idx = sample_indices(video.shape[0], T)
    clip = video.index_select(0, idx).permute(1,0,2,3)  # (C,T,H,W)
    return clip

def save_contact_sheet(clip, out="preview.png", n=8):
    # clip: (C,T,H,W) in [0,1]
    C,T,H,W = clip.shape
    step = max(1, T//n)
    frames = [clip[:,t,:,:].permute(1,2,0) for t in range(0, T, step)][:n]  # (H,W,C)
    frames = [ (f.clamp(0,1)*255).byte().cpu().numpy() for f in frames ]
    imgs = [Image.fromarray(frames[i]) for i in range(len(frames))]
    grid = Image.new("RGB", (W*len(imgs), H))
    for i,img in enumerate(imgs): grid.paste(img, (i*W, 0))
    grid.save(out)

items = json.loads(Path("hgd_index.json").read_text())
print("Total items:", len(items))

# Basic distribution checks
import collections
label_counts = collections.Counter([it["label"] for it in items])
print("Num classes:", len(label_counts), "min/max counts:", min(label_counts.values()), max(label_counts.values()))

# Decode one example
ex = items[0]
clip = load_clip(ex, T=32, size=(112,112))
print("Clip shape (C,T,H,W):", tuple(clip.shape), "label:", ex["label"], "class:", ex["class"], "user:", ex["user"], "rep:", ex["rep"])

# Save a quick preview strip
save_contact_sheet(clip, out="preview_example.png")
print("Wrote preview_example.png")