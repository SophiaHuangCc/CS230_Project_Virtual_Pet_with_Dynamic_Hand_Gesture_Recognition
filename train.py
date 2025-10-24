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
# Utilities
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    # Prefer Apple Silicon MPS if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_top1(logits, targets):
    preds = logits.argmax(1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# ------------------------------
# Training / Eval loops
# ------------------------------
def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a tiny 3D CNN on HGD (CPU-friendly).")
    parser.add_argument("--index", type=str, default="hgd_index.json", help="Path to index JSON built from CSV.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_len", type=int, default=16, help="Temporal frames per clip (keep small on CPU).")
    parser.add_argument("--resize", type=int, default=112, help="Spatial size (H=W). 112 is good for CPU dev.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0, help="macOS often happier with 0.")
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU even if MPS is available.")
    parser.add_argument("--save_path", type=str, default="best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)
    print(f"[info] device: {device}")

    # Dataset instances: one with train=True (augs on), one with train=False (no augs)
    size = (args.resize, args.resize)
    full_train_ds = HGDClips(args.index, clip_len=args.clip_len, size=size, train=True)
    full_eval_ds  = HGDClips(args.index, clip_len=args.clip_len, size=size, train=False)

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
    model = Tiny3DCNN(num_classes=27).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    start_time = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            t0 = time.time()
            train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_acc     = run_one_epoch(model, val_loader,   criterion, optimizer, device, train=False)
            dt = time.time() - t0

            print(f"[epoch {epoch}] "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"time={dt:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "args": vars(args),
                }, args.save_path)
                print(f"[info] ↑ new best val_acc={best_val_acc:.3f}; checkpoint saved to {args.save_path}")

    except KeyboardInterrupt:
        print("\n[warn] Training interrupted by user (Ctrl+C).")

    total_time = time.time() - start_time
    print(f"\n[done] best_val_acc={best_val_acc:.3f}  total_time={total_time/60:.1f} min")


if __name__ == "__main__":
    main()