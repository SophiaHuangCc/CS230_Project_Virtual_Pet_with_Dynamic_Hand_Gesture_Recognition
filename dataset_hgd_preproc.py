# dataset_hgd_preproc.py
import json, math, torch
from pathlib import Path
from torchvision.io import read_video
import torch.nn.functional as F

IMGNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
IMGNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
FPS = 30.0

def uniform_indices(n_src, n_out, jitter=False):
    if n_src <= 0: raise ValueError("Empty decoded segment")
    if n_src <= n_out:
        idx = list(range(n_src)) + [n_src-1] * (n_out - n_src)
    else:
        step = n_src / n_out
        if jitter:
            idx = []
            for i in range(n_out):
                a = int(i*step)
                b = int((i+1)*step) - 1
                b = max(a, b)
                idx.append(torch.randint(a, b+1, (1,)).item())
        else:
            idx = [int(i*step) for i in range(n_out)]
    return torch.tensor(idx)

def resize_letterbox(video, target_size=(112,112)):
    """
    video: (T,C,H,W) or (C,T,H,W)
    target_size: (new_h, new_w)
    returns video resized with preserved aspect ratio and padding
    """
    is_CT = video.shape[0] == 3  # detect layout
    if is_CT:  # (C,T,H,W)
        video = video.permute(1,0,2,3)  # -> (T,C,H,W)

    T, C, H, W = video.shape
    new_h, new_w = target_size
    scale = min(new_h/H, new_w/W)
    resized_h, resized_w = int(round(H*scale)), int(round(W*scale))
    
    # resize first
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    
    # pad to target
    pad_h = new_h - resized_h
    pad_w = new_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom))  # pad=(left,right,top,bottom)
    
    if is_CT:
        video = video.permute(1,0,2,3)  # back to (C,T,H,W)
    return video

class HGDClips(torch.utils.data.Dataset):
    def __init__(
        self,
        index_json,
        clip_len=32,
        size=(112,112),
        train=True,
        temporal_jitter=True,
        random_flip=True,
        brightness=0.0,             
    ):
        self.items = json.loads(Path(index_json).read_text())
        self.clip_len = clip_len
        self.size = size
        self.train = train

        # enable augs only for train
        self.temporal_jitter = (temporal_jitter if train else False)
        self.random_flip     = (random_flip     if train else False)
        self.brightness      = (float(brightness) if train else 0.0)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        s_f, e_f = it["start_frame"], it["end_frame"]
        start_sec = s_f / FPS
        end_sec   = max(start_sec + 1e-3, e_f / FPS)

        # Decode segment: (T,H,W,C) uint8
        video, _, info = read_video(it["video_path"], start_pts=start_sec, end_pts=end_sec, pts_unit='sec')
        # to float [0,1], -> (T,C,H,W)
        video = (video.float() / 255.0).permute(0,3,1,2)

        # Spatial resize
        if self.size is not None:
            # video = F.interpolate(video, size=self.size, mode="bilinear", align_corners=False)
            video = resize_letterbox(video, target_size=self.size)

        # Random horizontal flip
        if self.random_flip and torch.rand(()) < 0.5:
            video = torch.flip(video, dims=[3])  # flip W

        # Brightness jitter (multiply, then clamp to [0,1])
        if self.brightness > 0.0:
            lo, hi = 1.0 - self.brightness, 1.0 + self.brightness
            factor = torch.empty(()).uniform_(lo, hi).item()
            video = torch.clamp(video * factor, 0.0, 1.0)

        # Temporal sampling -> fixed T
        idx = uniform_indices(video.shape[0], self.clip_len, jitter=self.temporal_jitter)
        clip = video.index_select(0, idx)      # (T,C,H,W)
        clip = clip.permute(1,0,2,3)           # (C,T,H,W)

        # Normalize
        mean = IMGNET_MEAN.to(dtype=clip.dtype, device=clip.device)
        std  = IMGNET_STD.to(dtype=clip.dtype, device=clip.device)
        clip = (clip - mean) / std

        label = it["label"]
        return clip, label
