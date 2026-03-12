"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 3: YOLOv8 Classification on LARGE Augmented Dataset
  ──────────────────────────────────────────────────────────────
  Trains YOLOv8s-cls on the LARGE augmented dataset (~1500 images/class)
  with diverse, genuinely unique training images.

  Usage:
      python priority_3_augmentation/train_yolo_augmented.py
═══════════════════════════════════════════════════════════════════════════════
"""

from ultralytics import YOLO
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "priority_3_large_augmented_dataset"
OUTPUT_DIR = BASE_DIR / "priority_3_augmentation" / "yolo_runs"


def train():
    print("═" * 60)
    print("  YOLOv8 Classification — Augmented Dataset Training")
    print("═" * 60)

    # Use yolov8s-cls (small) for better capacity
    model = YOLO("yolov8s-cls.pt")

    print(f"\nTraining data: {DATA_DIR}")
    print(f"Output dir:    {OUTPUT_DIR}\n")

    results = model.train(
        data=str(DATA_DIR),
        epochs=150,
        imgsz=224,
        batch=16,
        patience=20,
        project=str(OUTPUT_DIR),
        name="sperm_cls_augmented",

        # ── Training params ──
        dropout=0.25,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,

        # ── Online augmentation (lighter since dataset is already augmented) ──
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.08,
        scale=0.3,
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,

        # ── Device ──
        device=0,
        workers=0,
        verbose=True,
    )

    print(f"\n✅ Training complete!")
    print(f"   Best weights: {OUTPUT_DIR / 'sperm_cls_augmented' / 'weights' / 'best.pt'}")

    return results


if __name__ == "__main__":
    train()
