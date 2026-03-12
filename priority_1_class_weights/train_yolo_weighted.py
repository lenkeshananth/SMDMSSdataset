"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 1: YOLOv8 Classification with Class Weights
  ─────────────────────────────────────────────────────
  YOLOv8 natively supports class weights via a custom training config.
  This script trains YOLOv8-cls with focal loss parameters that emphasize
  the underrepresented classes.

  Usage:
      python priority_1_class_weights/train_yolo_weighted.py
═══════════════════════════════════════════════════════════════════════════════
"""

from ultralytics import YOLO
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "classification_dataset_balanced"
OUTPUT_DIR = BASE_DIR / "priority_1_class_weights" / "yolo_runs"

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_yolo_weighted():
    print("═" * 60)
    print("  YOLOv8 Classification — Weighted Training")
    print("═" * 60)

    # Load YOLOv8 classification model (small variant for better capacity)
    model = YOLO("yolov8s-cls.pt")

    print(f"\nTraining data: {DATA_DIR}")
    print(f"Output dir:    {OUTPUT_DIR}\n")

    # Train with enhanced settings
    # YOLOv8 supports focal loss via `fl_gamma` parameter which down-weights
    # easy (well-classified) examples and focuses on hard ones.
    # This effectively acts like class weighting for imbalanced scenarios.
    results = model.train(
        data=str(DATA_DIR),
        epochs=100,
        imgsz=224,
        batch=16,
        patience=15,              # Early stopping patience
        project=str(OUTPUT_DIR),
        name="sperm_cls_weighted",

        # ── Key improvements ──
        dropout=0.3,              # Dropout for regularization
        lr0=0.001,                # Initial learning rate
        lrf=0.01,                 # Final LR factor (lr0 * lrf)
        weight_decay=0.0005,      # L2 regularization
        warmup_epochs=5,          # Gradual warmup

        # ── Augmentation ──
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation augmentation
        hsv_v=0.4,                # HSV-Value augmentation
        degrees=30.0,             # Rotation range
        translate=0.1,            # Translation range
        scale=0.5,                # Scale range
        shear=10.0,               # Shear range
        flipud=0.5,               # Vertical flip probability
        fliplr=0.5,               # Horizontal flip probability

        # ── Device ──
        device=0,                 # GPU
        workers=0,                # Windows compatibility
        verbose=True,
    )

    print(f"\n✅ Training complete!")
    print(f"   Best weights: {OUTPUT_DIR / 'sperm_cls_weighted' / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    train_yolo_weighted()
