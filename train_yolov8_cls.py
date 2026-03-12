"""
YOLOv8 Classification Training Script for SMDSS Sperm Dataset
==============================================================
This script:
1. Parses the expert annotation labels from the SMDSS dataset
2. Determines the morphological class for each sperm image based on
   majority vote from 3 experts
3. Organizes images into a YOLOv8-compatible classification folder structure
4. Splits data into train/val sets (80/20)
5. Trains a YOLOv8 classification model

Classes (based on anomaly categories):
  0: Normal       - No anomalies detected (NR prefix)
  1: Head         - Head anomalies (b, c, e, f, g prefixes)
  2: Midpiece     - Midpiece anomalies (h, j prefixes)
  3: Tail         - Tail anomalies (l, n, o prefixes)
  4: Combined     - Multiple anomaly regions (CN prefix / mixed)
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import Counter

# ──────────────────────────────── CONFIGURATION ────────────────────────────────
# Paths
DATASET_DIR   = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
IMAGES_DIR    = DATASET_DIR / "images"
LABELS_DIR    = DATASET_DIR / "labels"
OUTPUT_DIR    = DATASET_DIR / "classification_dataset"

# Training hyperparameters
MODEL_NAME    = "yolov8n-cls"        # YOLOv8 Nano classification (options: yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls)
EPOCHS        = 100                  # Number of training epochs
IMG_SIZE      = 224                  # Input image size
BATCH_SIZE    = 32                   # Batch size
TRAIN_SPLIT   = 0.8                  # 80% train, 20% val
SEED          = 42                   # Random seed for reproducibility
DEVICE        = "0"                  # GPU device ("0" for first GPU, "cpu" for CPU)
PROJECT_NAME  = "smdss_sperm_cls"    # Project name for saving results

# Class names
CLASS_NAMES = ["Normal", "Head_Anomaly", "Midpiece_Anomaly", "Tail_Anomaly", "Combined_Anomaly"]
# ───────────────────────────────────────────────────────────────────────────────


def parse_label_file(label_path):
    """
    Parse a SMDSS annotation file and extract anomaly information
    from all 3 experts using majority voting.
    
    Returns:
        dict with keys: head_anomalies, midpiece_anomalies, tail_anomalies
              (each is a list of anomaly codes from majority vote)
    """
    experts = {1: {"head": [], "mid": [], "tail": []},
               2: {"head": [], "mid": [], "tail": []},
               3: {"head": [], "mid": [], "tail": []}}
    
    current_expert = None
    
    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Expert1:"):
                current_expert = 1
            elif line.startswith("Expert2:"):
                current_expert = 2
            elif line.startswith("Expert3:"):
                current_expert = 3
            elif current_expert is not None:
                if line.startswith("head_anomalies:"):
                    value = line.split(":", 1)[1].strip().strip('"')
                    if value:
                        experts[current_expert]["head"] = [a.strip() for a in value.split(",")]
                elif line.startswith("midpiece_anomalies:"):
                    value = line.split(":", 1)[1].strip().strip('"')
                    if value:
                        experts[current_expert]["mid"] = [a.strip() for a in value.split(",")]
                elif line.startswith("tail_anomalies:"):
                    value = line.split(":", 1)[1].strip().strip('"')
                    if value:
                        experts[current_expert]["tail"] = [a.strip() for a in value.split(",")]
    
    # Determine presence of anomalies by majority vote (at least 2 out of 3 experts)
    has_head = sum(1 for e in experts.values() if e["head"]) >= 2
    has_mid  = sum(1 for e in experts.values() if e["mid"]) >= 2
    has_tail = sum(1 for e in experts.values() if e["tail"]) >= 2
    
    return {"has_head": has_head, "has_mid": has_mid, "has_tail": has_tail}


def determine_class(anomalies):
    """
    Determine the classification class based on anomaly presence.
    
    Returns:
        str: class name
    """
    regions_affected = sum([anomalies["has_head"], anomalies["has_mid"], anomalies["has_tail"]])
    
    if regions_affected == 0:
        return "Normal"
    elif regions_affected > 1:
        return "Combined_Anomaly"
    elif anomalies["has_head"]:
        return "Head_Anomaly"
    elif anomalies["has_mid"]:
        return "Midpiece_Anomaly"
    elif anomalies["has_tail"]:
        return "Tail_Anomaly"
    
    return "Normal"


def prepare_dataset():
    """
    Parse all labels, classify images, and organize into
    YOLOv8 classification folder structure:
    
    classification_dataset/
    ├── train/
    │   ├── Normal/
    │   ├── Head_Anomaly/
    │   ├── Midpiece_Anomaly/
    │   ├── Tail_Anomaly/
    │   └── Combined_Anomaly/
    └── val/
        ├── Normal/
        ├── Head_Anomaly/
        ├── Midpiece_Anomaly/
        ├── Tail_Anomaly/
        └── Combined_Anomaly/
    """
    print("=" * 60)
    print("  SMDSS Sperm Dataset - Data Preparation")
    print("=" * 60)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create directory structure
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Collect all image files (case-insensitive extension matching)
    image_files = sorted([f for f in IMAGES_DIR.iterdir() 
                          if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]])
    
    print(f"\nFound {len(image_files)} images in {IMAGES_DIR}")
    
    # Parse labels and classify each image
    class_assignments = {}  # {class_name: [image_paths]}
    for cls in CLASS_NAMES:
        class_assignments[cls] = []
    
    skipped = 0
    for img_path in image_files:
        label_name = img_path.stem + ".txt"
        label_path = LABELS_DIR / label_name
        
        if not label_path.exists():
            # Try case-insensitive search
            found = False
            for lf in LABELS_DIR.iterdir():
                if lf.name.lower() == label_name.lower():
                    label_path = lf
                    found = True
                    break
            if not found:
                print(f"  Warning: No label found for {img_path.name}, skipping.")
                skipped += 1
                continue
        
        anomalies = parse_label_file(label_path)
        cls = determine_class(anomalies)
        class_assignments[cls].append(img_path)
    
    # Print class distribution
    print(f"\nSkipped (no label): {skipped}")
    print(f"\n{'Class':<25} {'Count':>6}")
    print("-" * 35)
    total = 0
    for cls in CLASS_NAMES:
        count = len(class_assignments[cls])
        total += count
        print(f"  {cls:<23} {count:>6}")
    print("-" * 35)
    print(f"  {'TOTAL':<23} {total:>6}")
    
    # Split into train/val and copy images
    random.seed(SEED)
    train_count = 0
    val_count = 0
    
    print(f"\nSplitting data ({TRAIN_SPLIT*100:.0f}% train / {(1-TRAIN_SPLIT)*100:.0f}% val)...")
    
    for cls in CLASS_NAMES:
        images = class_assignments[cls]
        random.shuffle(images)
        
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_idx]
        val_images   = images[split_idx:]
        
        for img_path in train_images:
            dst = OUTPUT_DIR / "train" / cls / img_path.name
            shutil.copy2(img_path, dst)
            train_count += 1
        
        for img_path in val_images:
            dst = OUTPUT_DIR / "val" / cls / img_path.name
            shutil.copy2(img_path, dst)
            val_count += 1
        
        print(f"  {cls:<23} train: {len(train_images):>4}  |  val: {len(val_images):>4}")
    
    print(f"\n  Total train: {train_count}  |  Total val: {val_count}")
    print(f"\nDataset prepared at: {OUTPUT_DIR}")
    
    return OUTPUT_DIR


def train_model(data_path):
    """
    Train a YOLOv8 classification model on the prepared dataset.
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("  YOLOv8 Classification Training")
    print("=" * 60)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Epochs:     {EPOCHS}")
    print(f"  Image Size: {IMG_SIZE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Device:     {DEVICE}")
    print(f"  Data Path:  {data_path}")
    print("=" * 60)
    
    # Load pretrained YOLOv8 classification model
    model = YOLO(f"{MODEL_NAME}.pt")
    
    # Train the model
    results = model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(DATASET_DIR / "runs"),
        name=PROJECT_NAME,
        patience=0,               # Disable early stopping
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every 10 epochs
        pretrained=True,          # Use pretrained weights
        optimizer="Adam",         # Optimizer
        lr0=0.001,                # Initial learning rate
        lrf=0.01,                 # Final learning rate (fraction of lr0)
        weight_decay=0.0005,      # Weight decay
        warmup_epochs=5,          # Warmup epochs
        seed=SEED,                # Random seed
        workers=4,                # Number of data loader workers
        verbose=True,             # Verbose output
    )
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    
    # Validate on the validation set
    print("\nRunning validation...")
    metrics = model.val()
    
    print(f"\n  Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f}")
    
    # Print best model path
    best_model = DATASET_DIR / "runs" / PROJECT_NAME / "weights" / "best.pt"
    print(f"\n  Best model saved at: {best_model}")
    
    return model, results


def predict_sample(model, image_path):
    """
    Run prediction on a single image using the trained model.
    """
    results = model.predict(source=str(image_path), imgsz=IMG_SIZE, verbose=False)
    
    for result in results:
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = CLASS_NAMES[top1_idx]
        
        print(f"  Image: {Path(image_path).name}")
        print(f"  Predicted: {class_name} ({top1_conf*100:.1f}%)")
        print(f"  All probabilities:")
        for i, prob in enumerate(probs.data.tolist()):
            print(f"    {CLASS_NAMES[i]:<23} {prob*100:.1f}%")
    
    return results


# ──────────────────────────────── MAIN ────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Prepare the dataset
    data_path = prepare_dataset()
    
    # Step 2: Train the model
    model, results = train_model(data_path)
    
    # Step 3: Test prediction on a sample image
    print("\n" + "=" * 60)
    print("  Sample Prediction")
    print("=" * 60)
    
    sample_images = list((IMAGES_DIR).iterdir())[:3]
    for img in sample_images:
        predict_sample(model, img)
        print()
    
    print("Done! Training pipeline complete.")
