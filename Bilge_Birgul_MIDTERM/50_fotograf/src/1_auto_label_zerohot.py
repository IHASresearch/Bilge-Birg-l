# -*- coding: utf-8 -*-
"""
1_auto_label_zerohot.py
Open-Vocabulary YOLO-World Zero-Shot Pseudo-Labeling Pipeline
- Loads yolov8s-worldv2.pt.
- Targets open-vocabulary classes: ["fire", "flame"].
- Calibrates zero-shot contrastive head bias for unit-norm text features.
- Runs inference on unlabeled images (conf=0.35).
- Generates YOLO format labels (0: fire) and splits 80% train / 20% val.
- Generates portable data.yaml.
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import torch
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def safe_read_image(image_path: Path):
    """Safely reads images across Windows platforms with Unicode paths."""
    try:
        bytes_data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to read image: {image_path.name} -> {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Pseudo-Labeling with YOLO-World")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="",
        help="Path to folder containing unlabeled images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Path to dataset output directory"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for zero-shot pseudo-labeling"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio (default: 0.80)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split"
    )
    return parser.parse_args()

def resolve_source_dir(arg_dir: str, repo_root: Path) -> Path:
    if arg_dir and Path(arg_dir).exists():
        return Path(arg_dir).resolve()
    
    # Check common locations
    candidates = [
        repo_root.parent / "fire 50 fotoğraf",
        repo_root / "data" / "unlabeled",
        repo_root / "data",
        repo_root / "fire 50 fotoğraf"
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c.resolve()
            
    raise FileNotFoundError(
        "Could not find source image directory. Please specify with --source-dir <path>"
    )

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    source_img_dir = resolve_source_dir(args.source_dir, repo_root)
    dataset_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / "dataset_pseudo")

    train_img_dir = dataset_dir / "images" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_lbl_dir = dataset_dir / "labels" / "val"
    data_yaml_path = dataset_dir / "data.yaml"

    print("=" * 65)
    print(" 🔥 STEP 1: ZERO-SHOT PSEUDO-LABELING (YOLO-WORLD)")
    print("=" * 65)
    print(f"[+] Source Directory: {source_img_dir}")
    print(f"[+] Output Dataset Directory: {dataset_dir}")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[+] Hardware Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"    [GPU] {torch.cuda.get_device_name(0)}")

    # 1. Load YOLO-World
    print("\n[+] Loading YOLO-World model (yolov8s-worldv2.pt)...")
    model = YOLO("yolov8s-worldv2.pt")
    target_classes = ["fire", "flame"]
    print(f"[+] Configuring Open-Vocabulary Classes: {target_classes}")
    model.set_classes(target_classes)

    # 2. Calibrate BNContrastiveHead bias for zero-shot text features
    head = model.model.model[-1]
    for i in range(len(head.cv4)):
        head.cv4[i].bias.data += 8.0
    print("[+] Zero-shot logit calibration applied.")

    # 3. Collect source images
    all_images = sorted([
        p for p in source_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    total_imgs = len(all_images)
    print(f"[+] Found {total_imgs} valid images in source directory.")
    if total_imgs == 0:
        raise ValueError("No images found in source folder!")

    # 4. Zero-shot inference
    print(f"\n[+] Running Zero-Shot Inference (conf={args.conf})...")
    pseudo_data = []
    total_detections = 0
    images_with_det = 0

    for img_path in tqdm(all_images, desc="Zero-Shot Labeling"):
        img = safe_read_image(img_path)
        if img is None:
            continue

        results = model.predict(source=img, conf=args.conf, device=device, verbose=False)
        labels = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                images_with_det += 1
                total_detections += len(boxes)
                xywhn = boxes.xywhn.cpu().numpy()

                for cx, cy, w, h in xywhn:
                    cls_id = 0  # Map all fire/flame detections to single class (0: fire)
                    labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        pseudo_data.append((img_path, labels))

    print(f"\n[*] Pseudo-Labeling Statistics:")
    print(f"    - Total Images Processed: {len(pseudo_data)}")
    print(f"    - Images with Fire Detection: {images_with_det} / {len(pseudo_data)} ({images_with_det/len(pseudo_data)*100:.1f}%)")
    print(f"    - Total Generated Fire Bounding Boxes: {total_detections}")

    # 5. Split train / val
    random.seed(args.seed)
    random.shuffle(pseudo_data)

    split_idx = int(len(pseudo_data) * args.train_ratio)
    train_set = pseudo_data[:split_idx]
    val_set = pseudo_data[split_idx:]

    print(f"\n[+] Dataset Partition: {len(train_set)} Train ({args.train_ratio*100:.0f}%), {len(val_set)} Val ({(1-args.train_ratio)*100:.0f}%)")

    def save_split(items, dest_img_dir, dest_lbl_dir):
        for src_path, lbls in items:
            shutil.copy2(src_path, dest_img_dir / src_path.name)
            dst_lbl = dest_lbl_dir / f"{src_path.stem}.txt"
            with open(dst_lbl, "w", encoding="utf-8") as f:
                if lbls:
                    f.write("\n".join(lbls) + "\n")

    save_split(train_set, train_img_dir, train_lbl_dir)
    save_split(val_set, val_img_dir, val_lbl_dir)

    # 6. Generate data.yaml (portable relative/absolute paths)
    yaml_data = {
        "path": dataset_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "fire"
        }
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"[+] data.yaml created: {data_yaml_path}")
    print("\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
