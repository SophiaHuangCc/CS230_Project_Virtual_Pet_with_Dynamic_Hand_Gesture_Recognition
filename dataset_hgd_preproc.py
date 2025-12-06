import json
import torch
import random
from pathlib import Path
from torchvision.io import read_video
import torchvision.transforms.functional as TF 
from torchvision import transforms
import torch.nn.functional as F

K400_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
K400_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3,1,1,1)
FPS = 30.0

def resize_letterbox(video, target_size=(112,112)):
    is_CT = video.shape[0] == 3
    if is_CT: video = video.permute(1,0,2,3)
    T, C, H, W = video.shape
    new_h, new_w = target_size
    scale = min(new_h/H, new_w/W)
    resized_h, resized_w = int(round(H*scale)), int(round(W*scale))
    video = F.interpolate(video, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = new_h - resized_h
    pad_w = new_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom))
    if is_CT: video = video.permute(1,0,2,3)
    return video

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

class HGDClips(torch.utils.data.Dataset):
    def __init__(
        self,
        index_json,
        clip_len=32,
        size=(112,112),
        train=True,
        temporal_jitter=True,
        rotate_angle=5.0,
        color_jitter=0.2,   # Controls brightness, contrast, saturation
        keep_classes=None,
    ):
        items = json.loads(Path(index_json).read_text())

        # Optional class filtering
        if keep_classes is not None:
            keep_classes = list(keep_classes)
            class_to_new = {cls: i for i, cls in enumerate(keep_classes)}
            filtered = []
            for it in items:
                cls_id = it["class"]
                if cls_id in class_to_new:
                    it = dict(it)
                    it["label"] = class_to_new[cls_id]
                    filtered.append(it)
            self.items = filtered
            self.keep_classes = keep_classes
            self.class_to_new = class_to_new
        else:
            self.items = items

        self.clip_len = clip_len
        self.size = size
        self.train = train

        self.temporal_jitter = (temporal_jitter if train else False)
        self.rotate_angle    = (rotate_angle    if train else 0.0)
        
        # Initialize ColorJitter Object
        # This doesn't apply the transform yet, it just sets the ranges
        if train and color_jitter > 0:
            self.jitter_tf = transforms.ColorJitter(
                brightness=color_jitter, 
                contrast=color_jitter, 
                saturation=color_jitter, 
                hue=0.1 # Keep hue shift small to avoid unrealistic colors
            )
        else:
            self.jitter_tf = None

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
            video = resize_letterbox(video, target_size=self.size)

        # Random Rotation (Consistent across frames)
        if self.rotate_angle > 0:
            angle = random.uniform(-self.rotate_angle, self.rotate_angle)
            video = torch.stack([
                TF.rotate(frame, angle, interpolation=TF.InterpolationMode.BILINEAR) 
                for frame in video
            ])

        # Consistent Color Jitter (Consistent across frames)
        if self.jitter_tf is not None:
            fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
                self.jitter_tf.brightness, 
                self.jitter_tf.contrast, 
                self.jitter_tf.saturation, 
                self.jitter_tf.hue
            )

            jittered_frames = []
            for frame in video:
                for fn_id in fn_idx:
                    if fn_id == 0: frame = TF.adjust_brightness(frame, b)
                    elif fn_id == 1: frame = TF.adjust_contrast(frame, c)
                    elif fn_id == 2: frame = TF.adjust_saturation(frame, s)
                    elif fn_id == 3: frame = TF.adjust_hue(frame, h)
                jittered_frames.append(frame)
            
            video = torch.stack(jittered_frames)

        # Temporal sampling -> fixed T
        idx = uniform_indices(video.shape[0], self.clip_len, jitter=self.temporal_jitter)
        clip = video.index_select(0, idx)      # (T,C,H,W)
        clip = clip.permute(1,0,2,3)           # (C,T,H,W)

        # Normalize
        mean = K400_MEAN.to(dtype=clip.dtype, device=clip.device)
        std  = K400_STD.to(dtype=clip.dtype, device=clip.device)
        clip = (clip - mean) / std

        label = it["label"]
        return clip, label