# -*- coding: utf-8 -*-
"""
2_train_from_scratch.py
YOLOv8s Self-Supervised Training and Detection Efficiency (eta_det) Benchmark
- Trains standard yolov8s.pt architecture using generated dataset_pseudo/data.yaml.
- Evaluates the fine-tuned model against test images with conf=0.25.
- Computes Detection Efficiency eta_det = (F_det / F_top) * 100.
- Exports annotated predictions and experiment summary markdown report.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def safe_read_image(image_path: Path):
    try:
        bytes_data = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Failed to read image: {image_path.name} -> {e}")
        return None

def safe_write_image(image_path: Path, img: np.ndarray):
    try:
        ext = image_path.suffix.lower() if image_path.suffix else ".jpg"
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(str(image_path))
            return True
        return False
    except Exception as e:
        print(f"[ERROR] Failed to save image: {image_path.name} -> {e}")
        return False

def draw_predictions(img, boxes, conf_thresh):
    canvas = img.copy()
    box_color = (0, 69, 255)  # Fire orange/red
    text_bg = (0, 40, 180)

    det_count = 0
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            if conf < conf_thresh:
                continue
            det_count += 1
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            label = f"Fire: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(y1, th + 6)), text_bg, -1)
            cv2.putText(canvas, label, (x1 + 3, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    info_text = f"Detections: {det_count} (conf>={conf_thresh})"
    cv2.putText(canvas, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas, det_count

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8s and evaluate eta_det")
    parser.add_argument("--data", type=str, default="", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--conf-test", type=float, default=0.25, help="Inference test confidence threshold")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0 or cpu)")
    parser.add_argument("--source-dir", type=str, default="", help="Source test image directory")
    return parser.parse_args()

def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    data_yaml_path = Path(args.data).resolve() if args.data else (repo_root / "dataset_pseudo" / "data.yaml")
    results_dir = repo_root / "results"
    predictions_dir = results_dir / "predictions"
    summary_md_path = results_dir / "experiment_summary.md"

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml_path}. Run step 1 first!")

    # Resolve source images for evaluation
    source_img_dir = None
    if args.source_dir and Path(args.source_dir).exists():
        source_img_dir = Path(args.source_dir).resolve()
    else:
        for c in [repo_root.parent / "fire 50 fotoğraf", repo_root / "data" / "unlabeled", repo_root / "data", repo_root / "dataset_pseudo" / "images" / "train"]:
            if c.exists() and any(c.iterdir()):
                source_img_dir = c.resolve()
                break

    print("=" * 65)
    print(" 🚀 STEP 2: YOLOV8s TRAINING & DETECTION EFFICIENCY BENCHMARK")
    print("=" * 65)
    print(f"[+] Device: {args.device} (CUDA: {torch.cuda.is_available()})")
    print(f"[+] Dataset Config: {data_yaml_path}")
    print(f"[+] Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")

    # 1. Train YOLOv8s
    model = YOLO("yolov8s.pt")
    train_project_dir = repo_root / "runs" / "detect"
    
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        project=str(train_project_dir),
        name="train",
        exist_ok=True,
        verbose=True
    )

    best_model_path = train_project_dir / "train" / "weights" / "best.pt"
    if not best_model_path.exists():
        best_model_path = train_project_dir / "train" / "weights" / "last.pt"

    print(f"\n[+] Training Complete. Best model weights: {best_model_path}")

    # 2. Validation Metrics
    print("\n[+] Evaluating Validation Metrics (Val Set)...")
    best_model = YOLO(str(best_model_path))
    val_results = best_model.val(
        data=str(data_yaml_path),
        split="val",
        device=args.device,
        workers=0,
        verbose=False
    )

    precision = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    # 3. Test Evaluation and eta_det Calculation
    all_images = sorted([
        p for p in source_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    f_top = len(all_images)
    f_det = 0
    total_detected_boxes = 0

    print(f"\n[+] Testing on {f_top} images (conf={args.conf_test})...")
    for img_path in tqdm(all_images, desc="Predicting & Annotating"):
        img = safe_read_image(img_path)
        if img is None:
            continue

        res = best_model.predict(source=img, conf=args.conf_test, device=args.device, verbose=False)
        boxes = res[0].boxes if res and len(res) > 0 else None
        annotated_img, count = draw_predictions(img, boxes, args.conf_test)
        total_detected_boxes += count

        if count > 0:
            f_det += 1

        out_path = predictions_dir / f"pred_{img_path.name}"
        safe_write_image(out_path, annotated_img)

    eta_det = (f_det / f_top) * 100 if f_top > 0 else 0.0

    print("\n" + "=" * 65)
    print(" 🎯 DETECTION EFFICIENCY (eta_det) RESULTS")
    print("=" * 65)
    print(f"Total Images (F_top): {f_top}")
    print(f"Images with Detection (F_det): {f_det}")
    print(f"Detection Efficiency (eta_det): %{eta_det:.2f}")
    print(f"Total Bounding Boxes Detected: {total_detected_boxes}")

    # 4. Summary Report
    summary_md = f"""# 🧪 YOLOv8s Zero-Shot Pseudo-Labeling Benchmark Report

This project implements an autonomous fire detection training pipeline without human annotation, utilizing open-vocabulary **YOLO-World (yolov8s-worldv2.pt)** for zero-shot pseudo-labeling, followed by supervised training of standard **YOLOv8s** architecture.

---

## 📌 1. Experiment Setup
- **Source Dataset**: `{source_img_dir}` ($F_{{top}} = {f_top}$)
- **Pseudo-Labeling Model**: `yolov8s-worldv2.pt` (Zero-shot open-vocabulary classes: `['fire', 'flame']`)
- **Target Architecture**: Standard `yolov8s.pt`
- **Training Hyperparameters**:
  - `Epochs`: {args.epochs}
  - `Batch Size`: {args.batch}
  - `Image Size`: {args.imgsz}
  - `Device`: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
- **Evaluation Confidence**: `{args.conf_test}`

---

## 🎯 2. Detection Efficiency ($\\eta_{{det}}$)

Detection Efficiency Formula:
$$\\eta_{{det}} = \\left( \\frac{{F_{{det}}}}{{F_{{top}}}} \\right) \\times 100$$

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Total Images ($F_{{top}}$)** | **{f_top}** | Total unlabeled image pool |
| **Images Detected ($F_{{det}}$)** | **{f_det}** | Successfully detected fire images |
| **Detection Efficiency ($\\eta_{{det}}$)** | **%{eta_det:.2f}** | Dataset fire coverage ratio |
| **Total Bounding Boxes** | **{total_detected_boxes}** | Aggregate detected flame/fire regions |

---

## 📈 3. Model Validation Metrics

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Precision** | {precision:.4f} | %{precision*100:.2f} |
| **Recall** | {recall:.4f} | %{recall*100:.2f} |
| **mAP@50** | {map50:.4f} | %{map50*100:.2f} |
| **mAP@50-95** | {map50_95:.4f} | %{map50_95*100:.2f} |

---

## 🖼️ 4. Visual Predictions
Annotated predictions with bounding boxes and confidence scores are located in:
📁 `results/predictions/` (Total: {f_top} images)
"""

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"\n[+] Summary report generated: {summary_md_path}")
    print("\n✅ PIPELINE EXECUTION COMPLETED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
