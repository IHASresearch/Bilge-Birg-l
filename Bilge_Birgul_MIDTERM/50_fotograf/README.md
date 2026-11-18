# YOLOv8 Zero-Shot Pseudo-Labeling Fire Detection

An autonomous end-to-end computer vision project that trains a YOLOv8s fire detection model from completely unlabeled photos using open-vocabulary zero-shot pseudo-labeling (YOLO-World).

---

## 📌 Project Architecture

1. **Zero-Shot Pseudo-Labeling (`src/1_auto_label_zerohot.py`)**:
   - Employs open-vocabulary `yolov8s-worldv2.pt` with classes `['fire', 'flame']`.
   - Calibrates contrastive prediction logits for zero-shot text features.
   - Detects flame regions on unlabeled images with a confidence threshold of `0.35`.
   - Automatically maps bounding boxes to YOLO format (`0: fire`) and splits into 80% train / 20% validation.

2. **Self-Supervised Fine-Tuning (`src/2_train_from_scratch.py`)**:
   - Trains standard `yolov8s.pt` on the automatically generated pseudo-dataset for 50 epochs.
   - Evaluates fire detection efficiency ($\eta_{det}$) across the image pool with `conf=0.25`.
   - Generates visual predictions and metric summaries.

---

## 🎯 Detection Efficiency ($\eta_{det}$) & Benchmark Results

Detection Efficiency is defined as:
$$\eta_{det} = \left( \frac{F_{det}}{F_{top}} \right) \times 100$$

Where $F_{top}$ is the total number of images (50) and $F_{det}$ is the number of images where the model successfully localized fire.

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Total Images ($F_{top}$)** | **50** | Unlabeled input image pool |
| **Detected Images ($F_{det}$)** | **42** | Images with confirmed fire detection |
| **Detection Efficiency ($\eta_{det}$)** | **84.00%** | Flame coverage efficiency |
| **Total Flame Boxes** | **222** | Detected fire bounding boxes |
| **Precision** | **42.73%** | Validation metric |
| **Recall** | **42.06%** | Validation metric |
| **mAP@50** | **39.42%** | Validation metric |
| **mAP@50-95** | **28.23%** | Validation metric |

---

## 📁 Repository Structure

```text
├── dataset_pseudo/       # Generated training and validation splits
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── results/              # Performance reports & visual predictions
│   ├── experiment_summary.md
│   └── sample_predictions/
├── src/                  # Source scripts
│   ├── 1_auto_label_zerohot.py
│   └── 2_train_from_scratch.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd yolov8-fire-pseudo-labeling
pip install -r requirements.txt
```

### 2. Step 1: Run Zero-Shot Pseudo-Labeling

```bash
python src/1_auto_label_zerohot.py --source-dir "path/to/unlabeled_images" --conf 0.35
```

### 3. Step 2: Train and Benchmark YOLOv8s

```bash
python src/2_train_from_scratch.py --epochs 50 --batch 8 --conf-test 0.25
```
