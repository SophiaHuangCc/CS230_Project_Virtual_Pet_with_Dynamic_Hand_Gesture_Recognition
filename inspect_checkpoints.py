#!/usr/bin/env python3
"""
Utility script to inspect and compare model checkpoints.
"""
import torch
from pathlib import Path
from datetime import datetime
import json


def inspect_checkpoint(checkpoint_path: Path):
    """Inspect a single checkpoint and return its metadata."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': str(checkpoint_path),
            'exists': True,
            'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        if isinstance(checkpoint, dict):
            info['epoch'] = checkpoint.get('epoch', 'N/A')
            info['val_acc'] = checkpoint.get('val_acc', 'N/A')
            info['has_optimizer'] = 'optimizer_state' in checkpoint
            info['has_args'] = 'args' in checkpoint
            
            if 'args' in checkpoint:
                args = checkpoint['args']
                info['num_classes'] = args.get('num_classes', 'N/A')
                info['arch'] = args.get('arch', 'N/A')
                info['pretrained'] = args.get('pretrained', 'N/A')
                info['resize'] = args.get('resize', 'N/A')
                info['epochs'] = args.get('epochs', 'N/A')
                info['lr'] = args.get('lr', 'N/A')
        else:
            info['format'] = 'state_dict_only'
            
        return info
    except Exception as e:
        return {
            'path': str(checkpoint_path),
            'exists': False,
            'error': str(e)
        }


def list_all_checkpoints(runs_dir: Path = Path("runs/train")):
    """List all available checkpoints with their metadata."""
    checkpoints = list(runs_dir.glob("exp*/weights/best.pt"))
    
    if not checkpoints:
        print("No checkpoints found!")
        return []
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):\n")
    print("=" * 100)
    
    all_info = []
    for ckpt_path in sorted(checkpoints):
        info = inspect_checkpoint(ckpt_path)
        all_info.append(info)
        
        if info.get('exists', False):
            print(f"\n📁 {Path(info['path']).parent.parent.name}")
            print(f"   Path: {info['path']}")
            print(f"   Size: {info['size_mb']:.1f} MB")
            print(f"   Modified: {info['modified']}")
            
            if 'epoch' in info:
                print(f"   Epoch: {info['epoch']}")
            if 'val_acc' in info and info['val_acc'] != 'N/A':
                print(f"   Val Accuracy: {info['val_acc']:.4f}")
            if 'num_classes' in info:
                print(f"   Classes: {info['num_classes']}")
            if 'arch' in info:
                print(f"   Architecture: {info['arch']}")
            if 'pretrained' in info:
                print(f"   Pretrained: {info['pretrained']}")
            if 'resize' in info:
                print(f"   Resize: {info['resize']}")
            if 'epochs' in info:
                print(f"   Max Epochs: {info['epochs']}")
        else:
            print(f"\n❌ {info['path']}")
            print(f"   Error: {info.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 100)
    
    # Find best by validation accuracy
    valid_checkpoints = [info for info in all_info if info.get('exists') and info.get('val_acc') != 'N/A' and isinstance(info.get('val_acc'), (int, float))]
    if valid_checkpoints:
        best = max(valid_checkpoints, key=lambda x: x['val_acc'])
        print(f"\n🏆 Best by validation accuracy: {Path(best['path']).parent.parent.name} (val_acc={best['val_acc']:.4f})")
    
    # Find most recent
    if all_info:
        recent = max([info for info in all_info if info.get('exists')], key=lambda x: x.get('modified', ''))
        print(f"🕐 Most recent: {Path(recent['path']).parent.parent.name} (modified: {recent['modified']})")
    
    return all_info


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Inspect specific checkpoint
        ckpt_path = Path(sys.argv[1])
        info = inspect_checkpoint(ckpt_path)
        print(json.dumps(info, indent=2))
    else:
        # List all checkpoints
        list_all_checkpoints()

