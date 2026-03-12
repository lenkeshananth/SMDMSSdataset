"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 1: Evaluate Weighted Models
  ─────────────────────────────────────
  Runs evaluation on BOTH weighted models (EfficientNet + YOLOv8) and
  generates a comparison report against the original (unweighted) models.

  Usage:
      python priority_1_class_weights/evaluate_weighted.py
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
VAL_DIR = BASE_DIR / "classification_dataset_balanced" / "val"

# Weighted model paths (Priority 1)
EFFNET_WEIGHTED_PATH = BASE_DIR / "priority_1_class_weights" / "weights" / "efficientnet_b0_weighted.pt"
YOLO_WEIGHTED_PATH = BASE_DIR / "priority_1_class_weights" / "yolo_runs" / "sperm_cls_weighted" / "weights" / "best.pt"

# Original model paths (for comparison)
EFFNET_ORIGINAL_PATH = BASE_DIR / "efficientnet_training" / "weights" / "efficientnet_b0_best.pt"
YOLO_ORIGINAL_PATH = BASE_DIR / "runs" / "classify" / "runs" / "sperm_cls_balanced" / "weights" / "best.pt"

IMG_SIZE = 224
BATCH_SIZE = 32


def evaluate_efficientnet(weight_path, device, num_classes, class_names, label=""):
    """Evaluate an EfficientNet-B0 model."""
    print(f"\n{'─' * 50}")
    print(f"  Evaluating EfficientNet-B0 — {label}")
    print(f"{'─' * 50}")

    if not weight_path.exists():
        print(f"  ❌ Weights not found: {weight_path}")
        return None, None

    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )

    # Try loading, fall back to simpler classifier if shape mismatch
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
    except RuntimeError:
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(weight_path, map_location=device))

    model = model.to(device)
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

    print(f"  Overall Accuracy: {acc:.2%}")
    print(report)

    return all_labels, all_preds


def evaluate_yolo(weight_path, class_names, label=""):
    """Evaluate a YOLOv8 classification model."""
    print(f"\n{'─' * 50}")
    print(f"  Evaluating YOLOv8 — {label}")
    print(f"{'─' * 50}")

    if not weight_path.exists():
        print(f"  ❌ Weights not found: {weight_path}")
        return None, None

    model = YOLO(str(weight_path))
    val_dataset = datasets.ImageFolder(str(VAL_DIR))

    all_preds, all_labels = [], []

    for i, (img_path, label_idx) in enumerate(val_dataset.imgs):
        results = model.predict(img_path, imgsz=IMG_SIZE, verbose=False)
        pred_idx = results[0].probs.top1
        all_preds.append(pred_idx)
        all_labels.append(label_idx)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(val_dataset.imgs)}")

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

    print(f"  Overall Accuracy: {acc:.2%}")
    print(report)

    return all_labels, all_preds


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_dataset = datasets.ImageFolder(str(VAL_DIR))
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Val samples: {len(val_dataset)}")

    print("\n" + "═" * 60)
    print("       ORIGINAL vs WEIGHTED MODEL COMPARISON")
    print("═" * 60)

    # ── Evaluate all models ──
    evaluate_efficientnet(EFFNET_ORIGINAL_PATH, device, num_classes, class_names,
                          label="Original (unweighted)")
    evaluate_efficientnet(EFFNET_WEIGHTED_PATH, device, num_classes, class_names,
                          label="Priority 1 (weighted)")
    evaluate_yolo(YOLO_ORIGINAL_PATH, class_names,
                  label="Original (unweighted)")
    evaluate_yolo(YOLO_WEIGHTED_PATH, class_names,
                  label="Priority 1 (weighted)")


if __name__ == "__main__":
    main()
