#!/usr/bin/env python3
"""
train.py — minimal CPU-friendly training script for HGD

- Uses dataset_hgd_preproc.HGDClips (on-the-fly video decode, resize, normalize)
- Builds a small 3D CNN (much lighter than R3D-18 for CPU dev)
- Random 80/20 train/val split (reproducible with --seed)
- Prints epoch loss/accuracy; saves best checkpoint as best.pt
"""

import argparse
import os
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from torchvision.models.video import r3d_18, r2plus1d_18
from torchvision.models.video import R3D_18_Weights, R2Plus1D_18_Weights
import torch.nn.functional as F

import json
import numpy as np
import datetime as _dt
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# ---- import your dataset class ----
from dataset_hgd_preproc import HGDClips


# ------------------------------
# Tiny 3D CNN (CPU-friendly)
# ------------------------------
class Tiny3DCNN(nn.Module):
    def __init__(self, num_classes: int = 27, in_ch: int = 3):
        super().__init__()
        # Downsample using stride in convs + AvgPool3d (MPS-friendly)
        self.features = nn.Sequential(
            # block 1
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=(1,2,2), padding=1),  # spatial /2
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1,2,2)),  # spatial /2 again (total /4)

            # block 2
            nn.Conv3d(32, 64, kernel_size=3, stride=(2,1,1), padding=1),     # temporal /2
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1,2,2)),  # spatial /2 (total /8 vs input)

            # block 3
            nn.Conv3d(64, 128, kernel_size=3, stride=(2,1,1), padding=1),    # temporal /2 again
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((1,2,2)),  # spatial /2 again
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),   # (B,128,1,1,1)
            nn.Flatten(),              # (B,128)
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
    

# ------------------------------
# Transfer Learning with r3d_18
# ------------------------------
    
def TransferModel_1(arch, num_classes, pretrained=False, dropout=0.5):
    if arch == "r3d_18":
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r3d_18(weights=weights)
    elif arch == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r2plus1d_18(weights=weights)
    else:
        raise ValueError("Use --arch i3d only with PyTorchVideo (see below).")

    # Replace final classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model, weights

def set_trainable(module, trainable: bool, freeze_bn_affine: bool = True):
    module.train(trainable)
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            # keep BN in eval when "frozen"
            m.eval()
            if freeze_bn_affine:
                if m.weight is not None: m.weight.requires_grad = False
                if m.bias is not None:   m.bias.requires_grad = False
        for p in m.parameters(recurse=False):
            p.requires_grad = trainable

def freeze_until(model, stage: str = "layer3", freeze_bn_affine: bool = True):
    # stage ∈ {"stem","layer1","layer2","layer3","none"} meaning:
    # freeze everything up to and including that stage
    # torchvision video resnets expose: model.stem, model.layer1..layer4, model.fc
    order = ["stem", "layer1", "layer2", "layer3", "layer4"]
    if stage not in {"stem","layer1","layer2","layer3","none"}:
        raise ValueError("stage must be one of: stem, layer1, layer2, layer3, none")
    stop_idx = {"stem":0, "layer1":1, "layer2":2, "layer3":3, "none":-1}[stage]
    for i, name in enumerate(order):
        mod = getattr(model, name)
        set_trainable(mod, trainable=(i > stop_idx), freeze_bn_affine=freeze_bn_affine)
    # classifier head always trainable
    set_trainable(model.fc, trainable=True, freeze_bn_affine=False)

def make_optimizer(model, base_lr=3e-4, weight_decay=1e-3, unfreeze_last=False):
    params = []
    # head
    params.append({"params": model.fc.parameters(), "lr": base_lr})
    if unfreeze_last:
        params.append({"params": getattr(model, "layer4").parameters(), "lr": base_lr * 0.1})
    return torch.optim.AdamW(params, weight_decay=weight_decay)

# ------------------------------
# 3DResNet
# ------------------------------
def conv3x3x3(in_ch, out_ch, stride=(1,1,1), groups=1, dilation=1):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups, dilation=dilation)

def conv1x3x3(in_ch, out_ch, stride=(1,1,1)):
    return nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), bias=False)

def conv3x1x1(in_ch, out_ch, stride=(1,1,1)):
    return nn.Conv3d(in_ch, out_ch, kernel_size=(3,1,1), stride=stride, padding=(1,0,0), bias=False)

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=(1,1,1), downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
    def forward(self, x):
        id = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            id = self.downsample(x)
        out = self.relu(out + id)
        return out

class BasicBlock2p1D(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=(1,1,1), downsample=None):
        super().__init__()
        s_t, s_h, s_w = stride
        self.conv_t = conv3x1x1(in_ch, out_ch, (s_t,1,1))
        self.bn_t = nn.BatchNorm3d(out_ch)
        self.conv_sp = conv1x3x3(out_ch, out_ch, (1,s_h,s_w))
        self.bn_sp = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        id = x
        out = self.relu(self.bn_t(self.conv_t(x)))
        out = self.bn_sp(self.conv_sp(out))
        if self.downsample is not None:
            id = self.downsample(x)
        out = self.relu(out + id)
        return out

class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

def make_stage(block_cls, in_ch, out_ch, blocks, stride_t=1, stride_sp=2):
    layers = []
    stride = (stride_t, stride_sp, stride_sp)
    downsample = None
    if stride != (1,1,1) or in_ch != out_ch * block_cls.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(in_ch, out_ch * block_cls.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch * block_cls.expansion),
        )
    layers.append(block_cls(in_ch, out_ch, stride=stride, downsample=downsample))
    in_ch = out_ch * block_cls.expansion
    for _ in range(1, blocks):
        layers.append(block_cls(in_ch, out_ch))
    return nn.Sequential(*layers), in_ch

class ResNet3D(nn.Module):
    def __init__(self, block="r3d", layers=(2,2,2,2), num_classes=27, in_ch=3, width=64, dropout=0.5):
        super().__init__()
        block_cls = BasicBlock3D if block=="r3d" else BasicBlock2p1D
        self.stem = Stem(in_ch, width)
        c = width
        self.layer1, c = make_stage(block_cls, c, width,   layers[0], stride_t=1, stride_sp=1)
        self.layer2, c = make_stage(block_cls, c, width*2, layers[1], stride_t=2, stride_sp=2)
        self.layer3, c = make_stage(block_cls, c, width*4, layers[2], stride_t=2, stride_sp=2)
        self.layer4, c = make_stage(block_cls, c, width*8, layers[3], stride_t=2, stride_sp=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=False, use_gpu=False):
    if force_cpu:
        print("[info] Forcing CPU mode.")
        return torch.device("cpu")

    if use_gpu and torch.cuda.is_available():
        print(f"[info] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[info] Using Apple MPS backend.")
        return torch.device("mps")

    print("[info] Defaulting to CPU.")
    return torch.device("cpu")



def accuracy_top1(logits, targets):
    preds = logits.argmax(1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# ------------------------------
# Training / Eval loops
# ------------------------------
def force_bn_eval(model, freeze_affine=True):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()  # no running-stat updates
            if freeze_affine:
                if m.weight is not None: m.weight.requires_grad = False
                if m.bias   is not None: m.bias.requires_grad   = False

def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    force_bn_eval(model, freeze_affine=True)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for clips, labels in pbar:
        # if total_count == 0:
        #     print(f"[debug] batch clips={tuple(clips.shape)} {clips.dtype}; labels={labels.dtype} range=[{labels.min().item()},{labels.max().item()}]")
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # if total_count == 0:
        #     print("[DEBUG] shape of clips after to(device):", tuple(clips.shape))
        #     print("[DEBUG] shape of labels after to(device):", tuple(labels.shape))

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(clips)
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_count += batch_size

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "acc": f"{(total_correct/total_count):.3f}"
        })

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc

# ------------------------------
# Evaluation metrics
# ------------------------------
def _next_run_dir(base="runs/train"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    # exp, exp2, exp3…
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("exp")]
    if not existing:
        return base / "exp"
    nums = [int(p.name[3:]) for p in existing if p.name[3:].isdigit()]
    n = (max(nums) + 1) if nums else 2
    return base / f"exp{n}"

@torch.no_grad()
@torch.no_grad()
def _collect_logits_labels(model, loader, device):
    model.eval()
    all_logits, all_labels, sample_clips = [], [], []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        # keep labels on CPU and long
        labels = labels.long()                     # <— IMPORTANT
        logits = model(clips)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        # keep only a few clips to avoid huge RAM use
        sample_clips.append(clips.detach().cpu()[:2])
    return torch.cat(all_logits), torch.cat(all_labels), torch.cat(sample_clips)

def _confmat(num_classes, preds, labels):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for p, t in zip(preds.view(-1), labels.view(-1)):
        cm[t, p] += 1
    return cm

def _metrics_from_confmat(cm):
    # cm[i, j] = count of true=i, pred=j
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    tn = cm.sum().float() - (tp + fp + fn)
    eps = 1e-9

    precision = (tp / (tp + fp + eps)).numpy()
    recall    = (tp / (tp + fn + eps)).numpy()
    f1        = (2 * precision * recall / (precision + recall + eps))
    iou       = (tp / (tp + fp + fn + eps)).numpy()

    macro = {
        "precision_macro": float(np.nanmean(precision)),
        "recall_macro":    float(np.nanmean(recall)),
        "f1_macro":        float(np.nanmean(f1)),
        "iou_macro":       float(np.nanmean(iou)),
    }
    micro_acc = float(tp.sum().item() / max(1, cm.sum().item()))
    return precision, recall, f1, iou, macro, micro_acc

def _plot_confmat(cm, class_names, out_png):
    cm_np = cm.numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_np, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label', xlabel='Predicted label', title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_np.max() / 2.0 if cm_np.size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm_np[i, j]),
                    ha="center", va="center",
                    color="white" if cm_np[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)

def _clip_contact_sheet(clip_CT_HW, n=8):  # clip is (C,T,H,W)
    C,T,H,W = clip_CT_HW.shape
    step = max(1, T//n)
    frames = [clip_CT_HW[:, t, :, :] for t in range(0, T, step)][:n]  # list of (C,H,W)
    grid = vutils.make_grid(frames, nrow=len(frames))  # (C, H, n*W)
    return grid  # tensor [0..1] if inputs were normalized differently you may want to denorm

@torch.no_grad()
def evaluate_full(model, loader, criterion, device, num_classes, class_names, save_dir, epoch, max_bad=12):
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    logits, labels, clips = _collect_logits_labels(model, loader, device)
    # sanity checks to catch bad labels early
    assert logits.ndim == 2 and labels.ndim == 1, f"shapes {logits.shape} vs {labels.shape}"
    assert labels.dtype == torch.long, f"labels dtype {labels.dtype} (need long)"
    assert logits.shape[0] == labels.shape[0], f"N mismatch {logits.shape[0]} vs {labels.shape[0]}"
    mn, mx = int(labels.min()), int(labels.max())
    assert 0 <= mn and mx < logits.shape[1], f"label range [{mn},{mx}] out of [0,{logits.shape[1]-1}]"
    
    loss = criterion(logits, labels).item()

    preds = logits.argmax(1)
    cm = _confmat(num_classes, preds, labels)
    prec, rec, f1, iou, macro, acc = _metrics_from_confmat(cm)

    # Save metrics
    metrics = {
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc,
        **macro,
        "per_class": [
            {"idx": i, "precision": float(prec[i]), "recall": float(rec[i]),
             "f1": float(f1[i]), "iou": float(iou[i]),
             "name": class_names[i] if class_names else str(i)}
            for i in range(num_classes)
        ]
    }
    (save_dir / "metrics").mkdir(exist_ok=True)
    json.dump(metrics, open(save_dir / "metrics" / f"epoch_{epoch:03d}.json", "w"), indent=2)

    # Save confusion matrix plot
    _plot_confmat(cm, class_names or [str(i) for i in range(num_classes)],
                  save_dir / f"confmat_epoch_{epoch:03d}.png")

    # Qualitative: save a few wrong predictions
    # wrong_idx = (preds != labels).nonzero(as_tuple=False).flatten().tolist()
    # (save_dir / "qual").mkdir(exist_ok=True)
    # for k, idx in enumerate(wrong_idx[:max_bad]):
    #     grid = _clip_contact_sheet(clips[idx].cpu())  # (C,H,W*)
    #     vutils.save_image(grid, save_dir / "qual" / f"wrong_{epoch:03d}_{k:03d}_t{int(labels[idx])}_p{int(preds[idx])}.png")
    # return acc, loss, cm
    # Qualitative: save a few wrong predictions
    wrong_idx = (preds != labels).nonzero(as_tuple=False).flatten().tolist()
    (save_dir / "qual").mkdir(exist_ok=True)

    num_clips = clips.size(0)
    for k, idx in enumerate(wrong_idx[:max_bad]):
        if idx >= num_clips:
            # we only kept a subset of clips in _collect_logits_labels, so skip out-of-range indices
            continue
        grid = _clip_contact_sheet(clips[idx].cpu())  # (C,H,W*)
        vutils.save_image(
            grid,
            save_dir / "qual" / f"wrong_{epoch:03d}_{k:03d}_t{int(labels[idx])}_p{int(preds[idx])}.png"
        )

    return acc, loss, cm

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a tiny 3D CNN on HGD (CPU-friendly).")
    parser.add_argument("--index", type=str, default="hgd_index.json", help="Path to index JSON built from CSV.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=16, help="Temporal frames per clip (keep small on CPU).")
    parser.add_argument("--resize", type=int, default=112, help="Spatial size (H=W). 112 is good for CPU dev.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0, help="macOS often happier with 0.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if MPS/GPU is available.")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDA GPU if available.")
    parser.add_argument("--save_path", type=str, default="best.pt")
    # add to argparse
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--temporal_jitter", type=int, default=2)
    parser.add_argument("--brightness", type=float, default=0.2)  # ±20%
    parser.add_argument("--arch", type=str, default="r3d_18",
                    choices=["r3d_18", "r2plus1d_18", "i3d"],
                    help="Backbone architecture")
    
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained Kinetics-400 weights when available")
    parser.add_argument("--num_classes", type=int, default=27)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--freeze_until", type=str, default="layer3",
                    choices=["none","stem","layer1","layer2","layer3"],
                    help="Freeze backbone up to and including this stage")
    parser.add_argument("--freeze_bn_affine", action="store_true",
                        help="Also freeze BN gamma/beta in frozen blocks")
    parser.add_argument("--unfreeze_last", action="store_true",
                        help="Also fine-tune layer4 with a smaller LR")


    args = parser.parse_args()

    # --- define the 10 classes we care about ---
    KEEP_CLASSES = [10, 11, 16, 17, 18, 23, 24, 25, 26, 27]
    args.num_classes = len(KEEP_CLASSES)  # 10

    set_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu, use_gpu=args.use_gpu)
    print(f"[info] device: {device}")

    # Logging setup
    # run_dir = _next_run_dir("runs/train")
    # (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    # (run_dir / "metrics").mkdir(exist_ok=True)
    # (run_dir / "qual").mkdir(exist_ok=True)
    # json.dump(vars(args), open(run_dir / "hparams.json", "w"), indent=2)
    # print(f"[info] logging to: {run_dir}")
    # class_names = [f"c{i}" for i in range(args.num_classes)]

    # Dataset instances: one with train=True (augs on), one with train=False (no augs)
    # size = (args.resize, args.resize)
    # full_train_ds = HGDClips(args.index, clip_len=args.clip_len, size=size, train=True)
    # full_eval_ds  = HGDClips(args.index, clip_len=args.clip_len, size=size, train=False)

    # Logging setup: 10 classes
    run_dir = _next_run_dir("runs/train")
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "qual").mkdir(exist_ok=True)
    json.dump(vars(args), open(run_dir / "hparams.json", "w"), indent=2)
    print(f"[info] logging to: {run_dir}")

    # Human-readable class names for the 10 kept gestures
    # (maps index 0..9 back to original class ids)
    class_names = [f"cls{cls_id:02d}" for cls_id in KEEP_CLASSES]

    # Dataset instances: one with train=True (augs on), one with train=False (no augs)
    size = (args.resize, args.resize)
    full_train_ds = HGDClips(
        args.index,
        clip_len=args.clip_len,
        size=size,
        train=True,
        keep_classes=KEEP_CLASSES,     # <-- NEW
    )
    full_eval_ds  = HGDClips(
        args.index,
        clip_len=args.clip_len,
        size=size,
        train=False,
        keep_classes=KEEP_CLASSES,     # <-- NEW
    )

    # Split indices reproducibly
    n_total = len(full_train_ds)
    n_val = int(math.ceil(n_total * args.val_split))
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)

    # Wrap as Subset so train/eval get different augment settings
    train_ds = Subset(full_train_ds, train_idx.indices if hasattr(train_idx, "indices") else train_idx)
    val_ds   = Subset(full_eval_ds,  val_idx.indices if hasattr(val_idx, "indices") else val_idx)

    print(f"[info] dataset sizes: train={len(train_ds)}  val={len(val_ds)}  (total {n_total})")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    # Model / loss / opt
    # Tiny3DCNN
    # model = model = Tiny3DCNN(num_classes=27).to(device)
    # Transfer Learning （exp13 pretrained r3d_18=0.343)
    model, weights = TransferModel_1(args.arch, args.num_classes, args.pretrained, args.dropout)
    # Freeze more (strongest freeze)
    freeze_until(model, stage="layer3")  # freezes stem, layer1, layer2, layer3
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay, unfreeze_last=False)
    model = model.to(device)
    # Baseline ResNet
    # model = ResNet3D(block="r3d", layers=(2,2,2,2), num_classes=27).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 10 classes
    # model = ResNet3D(block="r3d", layers=(2,2,2,2), num_classes=args.num_classes).to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            t0 = time.time()

            # 1) Train for one epoch
            train_loss, train_acc = run_one_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )

            # 2) Full eval with metrics + artifacts on val set
            val_acc, val_loss_exact, cm = evaluate_full(
                model, val_loader, criterion, device,
                num_classes=args.num_classes,
                class_names=class_names,
                save_dir=run_dir,
                epoch=epoch,
                max_bad=12,
            )

            dt = time.time() - t0
            print(
                f"[epoch {epoch}] "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss_exact:.4f}  val_acc={val_acc:.3f}  "
                f"time={dt:.1f}s"
            )

            # 3) Append a one-line CSV log for THIS epoch (like YOLO)
            with open(run_dir / "results.csv", "a") as f:
                if epoch == 1 and f.tell() == 0:
                    f.write(
                        "epoch,train_loss,train_acc,"
                        "val_loss,val_acc,"
                        "precision_macro,recall_macro,f1_macro,iou_macro,"
                        "time\n"
                    )
                # load the macro metrics we just wrote
                mpath = run_dir / "metrics" / f"epoch_{epoch:03d}.json"
                m = json.load(open(mpath))
                f.write(
                    f"{epoch},"
                    f"{train_loss:.6f},{train_acc:.6f},"
                    f"{m['loss']:.6f},{m['accuracy']:.6f},"
                    f"{m['precision_macro']:.6f},"
                    f"{m['recall_macro']:.6f},"
                    f"{m['f1_macro']:.6f},"
                    f"{m['iou_macro']:.6f},"
                    f"{dt:.3f}\n"
                )

            # 4) Save best weights based on *validation accuracy* (YOLO-style)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_path = run_dir / "weights" / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "args": vars(args),
                    },
                    best_path,
                )
                print(
                    f"[info] ↑ new best at epoch {best_epoch} "
                    f"val_acc={best_val_acc:.3f}; saved to {best_path}"
                )

    except KeyboardInterrupt:
        print("\n[warn] Training interrupted by user (Ctrl+C).")

    total_time = time.time() - start_time
    print(f"\n[done] best_val_acc={best_val_acc:.3f}  total_time={total_time/60:.1f} min")


if __name__ == "__main__":
    main()