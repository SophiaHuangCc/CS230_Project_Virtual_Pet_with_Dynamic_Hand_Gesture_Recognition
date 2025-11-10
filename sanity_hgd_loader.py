# # sanity_hgd_loader.py
# import json, math, torch
# from pathlib import Path
# from torchvision.io import read_video
# import torch.nn.functional as F
# from PIL import Image

# FPS = 30.0

# def sample_indices(n_src, n_out):
#     if n_src <= n_out:
#         idx = list(range(n_src)) + [n_src-1]*(n_out-n_src)
#     else:
#         step = n_src / n_out
#         idx = [int(i*step) for i in range(n_out)]
#     return torch.tensor(idx)

# def load_clip(item, T=32, size=(112,112)):
#     s_f, e_f = item["start_frame"], item["end_frame"]
#     start_sec = s_f / FPS
#     end_sec   = max(start_sec + 1e-3, e_f / FPS)
#     video, _, _ = read_video(item["video_path"], start_pts=start_sec, end_pts=end_sec, pts_unit='sec')
#     # (t,h,w,c) uint8 -> float -> (t,c,h,w)
#     video = (video.float()/255.0).permute(0,3,1,2)
#     if size:
#         video = F.interpolate(video, size=size, mode="bilinear", align_corners=False)
#     idx = sample_indices(video.shape[0], T)
#     clip = video.index_select(0, idx).permute(1,0,2,3)  # (C,T,H,W)
#     return clip

# def save_contact_sheet(clip, out="preview.png", n=8):
#     # clip: (C,T,H,W) in [0,1]
#     C,T,H,W = clip.shape
#     step = max(1, T//n)
#     frames = [clip[:,t,:,:].permute(1,2,0) for t in range(0, T, step)][:n]  # (H,W,C)
#     frames = [ (f.clamp(0,1)*255).byte().cpu().numpy() for f in frames ]
#     imgs = [Image.fromarray(frames[i]) for i in range(len(frames))]
#     grid = Image.new("RGB", (W*len(imgs), H))
#     for i,img in enumerate(imgs): grid.paste(img, (i*W, 0))
#     grid.save(out)

# items = json.loads(Path("hgd_index.json").read_text())
# print("Total items:", len(items))

# # Basic distribution checks
# import collections
# label_counts = collections.Counter([it["label"] for it in items])
# print("Num classes:", len(label_counts), "min/max counts:", min(label_counts.values()), max(label_counts.values()))

# # Decode one example
# ex = items[0]
# clip = load_clip(ex, T=32, size=(112,112))
# print("Clip shape (C,T,H,W):", tuple(clip.shape), "label:", ex["label"], "class:", ex["class"], "user:", ex["user"], "rep:", ex["rep"])

# # Save a quick preview strip
# save_contact_sheet(clip, out="preview_example.png")
# print("Wrote preview_example.png")

# sanity_visualize_preproc.py
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from torchvision.io import read_video

# ---- import your dataset + constants ----
from dataset_hgd_preproc import HGDClips, IMGNET_MEAN, IMGNET_STD, FPS

# ---------- helpers ----------
def denorm_clip(clip_CT_HW):
    """(C,T,H,W) -> (C,T,H,W) in [0,1] for visualization."""
    mean = IMGNET_MEAN.to(dtype=clip_CT_HW.dtype, device=clip_CT_HW.device)
    std  = IMGNET_STD.to(dtype=clip_CT_HW.dtype, device=clip_CT_HW.device)
    x = clip_CT_HW * std + mean
    return x.clamp(0,1)

def to_pil(img_CHW):
    """(C,H,W) in [0,1] -> PIL.Image"""
    arr = (img_CHW.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)

def contact_sheet(clip_CT_HW, n=8):
    """Sample T frames uniformly and make a horizontal grid."""
    C,T,H,W = clip_CT_HW.shape
    step = max(1, T//n)
    frames = [clip_CT_HW[:, t, :, :] for t in range(0, T, step)][:n]
    frames = [to_pil(f) for f in frames]
    grid = Image.new("RGB", (W*len(frames), H))
    for i, im in enumerate(frames):
        grid.paste(im, (i*W, 0))
    return grid

def letterbox_params(src_h, src_w, dst_h, dst_w):
    scale = min(dst_h/src_h, dst_w/src_w)
    rh, rw = int(round(src_h*scale)), int(round(src_w*scale))
    pad_h, pad_w = dst_h - rh, dst_w - rw
    top = pad_h // 2
    left = pad_w // 2
    bottom = pad_h - top
    right = pad_w - left
    return rh, rw, top, bottom, left, right, scale

def draw_padding_overlay(pil_img, top, bottom, left, right, color=(255,0,0), alpha=90):
    """Shade the padding regions on a PIL image."""
    W, H = pil_img.size
    overlay = Image.new('RGBA', (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    # top band
    if top > 0:    draw.rectangle([0, 0, W, top], fill=(*color, alpha))
    # bottom band
    if bottom > 0: draw.rectangle([0, H-bottom, W, H], fill=(*color, alpha))
    # left band
    if left > 0:   draw.rectangle([0, 0, left, H], fill=(*color, alpha))
    # right band
    if right > 0:  draw.rectangle([W-right, 0, W, H], fill=(*color, alpha))
    return Image.alpha_composite(pil_img.convert("RGBA"), overlay).convert("RGB")

# ---------- main ----------
def main():
    index_path = Path("hgd_index.json")
    assert index_path.exists(), "hgd_index.json not found."

    items = json.loads(index_path.read_text())
    ex = items[0]  # visualize first example (change index if you want)
    print("Example path:", ex["video_path"])

    # --- decode raw segment (original) ---
    s_f, e_f = ex["start_frame"], ex["end_frame"]
    start_sec = s_f / FPS
    end_sec   = max(start_sec + 1e-3, e_f / FPS)
    raw, _, info = read_video(ex["video_path"], start_pts=start_sec, end_pts=end_sec, pts_unit='sec')  # (T,H,W,C), uint8
    T_raw, H_raw, W_raw, _ = raw.shape
    print(f"Raw segment shape: (T,H,W,C)=({T_raw},{H_raw},{W_raw},3)  fps={info['video_fps']}")
    print(f"Raw aspect ratio: {W_raw}:{H_raw} = {W_raw/H_raw:.4f}")

    # --- build a dataset that uses your letterbox resize + NO jitter for determinism ---
    # size matches your train script default; change if needed
    ds = HGDClips(index_json=str(index_path),
                  clip_len=16,            # choose a clip length to visualize
                  size=(112,112),         # your target size
                  train=False,            # no augs (temporal_jitter/flip/brightness off)
                  temporal_jitter=False,
                  random_flip=False,
                  brightness=0.0)

    clip_CT_HW, label = ds[0]  # (C,T,H,W) after resize+normalize
    C,T,H,W = clip_CT_HW.shape
    print(f"Processed clip shape (C,T,H,W)=({C},{T},{H},{W})")
    print(f"Processed aspect ratio: {W}:{H} = {W/H:.4f}")

    # --- denormalize for viewing ---
    clip_vis = denorm_clip(clip_CT_HW)  # [0,1]

    # --- save a contact sheet of processed frames ---
    grid = contact_sheet(clip_vis, n=8)
    grid.save("processed_contact_sheet.png")
    print("Wrote processed_contact_sheet.png")

    # --- show raw first frame (center of raw segment for fairness) ---
    t0 = min(T_raw//2, T_raw-1)  # pick a central frame
    raw_first = torch.from_numpy(raw[t0].permute(2,0,1).numpy()).float()/255.0  # (C,H,W) [0,1]
    to_pil(raw_first).save("orig_first_frame.png")
    print("Wrote orig_first_frame.png")

    # --- reconstruct letterbox on that same raw frame to visualize padding explicitly ---
    # resize raw frame proportionally to target and pad
    dst_h, dst_w = H, W
    rh, rw, top, bottom, left, right, scale = letterbox_params(H_raw, W_raw, dst_h, dst_w)
    raw_first_bchw = raw_first.unsqueeze(0)  # (1,C,H,W)
    resized = F.interpolate(raw_first_bchw, size=(rh, rw), mode="bilinear", align_corners=False)[0]
    canvas = torch.zeros((3, dst_h, dst_w), dtype=resized.dtype)
    canvas[:, top:top+rh, left:left+rw] = resized
    lb = to_pil(canvas)

    # draw padding overlays
    lb_pad = draw_padding_overlay(lb, top, bottom, left, right, color=(255,0,0), alpha=80)
    lb_pad.save("resized_first_frame_with_padding.png")
    print("Wrote resized_first_frame_with_padding.png")
    print(f"Letterbox params: resized_h={rh}, resized_w={rw}, top={top}, bottom={bottom}, left={left}, right={right}, scale={scale:.6f}")

    print("Done.")

if __name__ == "__main__":
    main()