"""
Fine-Grained Morphology Classification V2 - Improved Training
==============================================================
Uses class weights and aggressive augmentation instead of oversampling.

Usage:
    python train_finegrained_v2.py --prepare
    python train_finegrained_v2.py --train
    python train_finegrained_v2.py --evaluate
    python train_finegrained_v2.py --all
"""

import os
import re
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import cv2
import argparse

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

# Output directories
OUTPUT_DIR = BASE_DIR / "finegrained_v2"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
RESULTS_DIR = OUTPUT_DIR / "results"

# Fine-grained class names
ANOMALY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'N', 'O', 'Nr']

# Training settings - OPTIMIZED
MODEL_NAME = "yolov8m-cls"  # Medium model for better capacity
EPOCHS = 150                # More epochs for better convergence
IMG_SIZE = 224
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8
SEED = 42
DEVICE = "0"

# ══════════════════════════════════════════════════════════════════════════════


def parse_label_file(label_path):
    """Parse annotation file and extract anomaly codes using majority voting."""
    experts = {1: [], 2: [], 3: []}
    current_expert = None
    
    try:
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
                    if "anomalies:" in line:
                        value = line.split(":", 1)[1].strip().strip('"')
                        if value:
                            codes = [c.strip().upper() for c in value.split(",") if c.strip()]
                            experts[current_expert].extend(codes)
    except Exception as e:
        return []
    
    all_codes = []
    for exp_codes in experts.values():
        all_codes.extend(exp_codes)
    
    if not all_codes:
        return ['NR']
    
    code_counts = Counter(all_codes)
    majority_codes = [code for code, count in code_counts.items() if count >= 2]
    
    if not majority_codes:
        majority_codes = [code_counts.most_common(1)[0][0]]
    
    return majority_codes


def get_primary_anomaly(codes):
    """Get the primary anomaly class."""
    if not codes or codes == ['NR']:
        return 'Nr'
    
    codes = [c.upper() for c in codes]
    priority_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'N', 'O']
    
    for code in priority_order:
        if code in codes:
            return code
    
    return 'Nr'


def augment_image(img, idx):
    """Apply random augmentation to an image."""
    augmented = img.copy()
    
    # Random operations based on index
    ops = idx % 8
    
    if ops == 1:
        augmented = cv2.flip(augmented, 1)  # Horizontal flip
    elif ops == 2:
        augmented = cv2.flip(augmented, 0)  # Vertical flip
    elif ops == 3:
        augmented = cv2.flip(augmented, -1)  # Both flips
    elif ops == 4:
        # Rotate 90
        augmented = cv2.rotate(augmented, cv2.ROTATE_90_CLOCKWISE)
    elif ops == 5:
        # Brightness adjustment
        augmented = cv2.convertScaleAbs(augmented, alpha=1.2, beta=20)
    elif ops == 6:
        # Darkness adjustment
        augmented = cv2.convertScaleAbs(augmented, alpha=0.8, beta=-20)
    elif ops == 7:
        # Gaussian blur
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)
    
    return augmented


def prepare_dataset_with_augmentation():
    """
    Prepare dataset with proper augmentation for minority classes.
    Uses image-level augmentation to create diverse samples.
    """
    print("=" * 70)
    print("      PREPARING DATASET WITH AUGMENTATION (V2)")
    print("=" * 70)
    
    # Clear existing output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create class directories
    for split in ['train', 'val']:
        for cls in ANOMALY_CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Parse all labels
    data_by_class = defaultdict(list)
    
    print(f"\n📂 Scanning labels from: {LABELS_DIR}")
    
    label_files = list(LABELS_DIR.glob("*.txt"))
    
    for label_path in label_files:
        img_name = label_path.stem + ".png"
        img_path = IMAGES_DIR / img_name
        
        if not img_path.exists():
            img_name = label_path.stem + ".jpg"
            img_path = IMAGES_DIR / img_name
        
        if not img_path.exists():
            continue
        
        codes = parse_label_file(label_path)
        primary_class = get_primary_anomaly(codes)
        
        if primary_class in ANOMALY_CLASSES:
            data_by_class[primary_class].append(img_path)
    
    # Print original distribution
    print(f"\n📊 Original Class Distribution:")
    total = 0
    class_counts = {}
    for cls in ANOMALY_CLASSES:
        count = len(data_by_class[cls])
        class_counts[cls] = count
        total += count
        print(f"   {cls:3}: {count:5} images")
    print(f"   {'Total':3}: {total:5} images")
    
    # Calculate target count (median of class sizes for moderate augmentation)
    counts = list(class_counts.values())
    target_count = int(np.median([c for c in counts if c > 0]))
    target_count = max(target_count, 50)  # At least 50 per class
    
    print(f"\n📊 Target count per class: {target_count}")
    
    random.seed(SEED)
    train_count = 0
    val_count = 0
    
    for cls in ANOMALY_CLASSES:
        images = data_by_class[cls]
        if len(images) == 0:
            continue
            
        random.shuffle(images)
        
        # Split: ensure at least 1 val sample
        if len(images) >= 5:
            split_idx = int(len(images) * TRAIN_SPLIT)
        else:
            split_idx = max(1, len(images) - 1)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:] if split_idx < len(images) else [images[-1]]
        
        # Copy validation images (no augmentation)
        for img_path in val_images:
            dst = VAL_DIR / cls / img_path.name
            shutil.copy2(img_path, dst)
            val_count += 1
        
        # Augment training images to reach target count
        current_train = len(train_images)
        augment_factor = max(1, target_count // current_train)
        
        aug_count = 0
        for img_path in train_images:
            # Copy original
            dst = TRAIN_DIR / cls / img_path.name
            shutil.copy2(img_path, dst)
            train_count += 1
            aug_count += 1
            
            # Create augmented versions if needed
            if current_train < target_count:
                img = cv2.imread(str(img_path))
                if img is not None:
                    for aug_idx in range(1, augment_factor):
                        if aug_count >= target_count:
                            break
                        augmented = augment_image(img, aug_idx)
                        aug_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                        aug_dst = TRAIN_DIR / cls / aug_name
                        cv2.imwrite(str(aug_dst), augmented)
                        train_count += 1
                        aug_count += 1
        
        print(f"   {cls}: {len(train_images)} original -> {aug_count} augmented (train), {len(val_images)} (val)")
    
    print(f"\n   ✓ Train: {train_count} images")
    print(f"   ✓ Val:   {val_count} images")
    print(f"   ✓ Dataset saved to: {OUTPUT_DIR}")
    
    return OUTPUT_DIR


def train_model():
    """Train with optimized settings for imbalanced data."""
    print("\n" + "=" * 70)
    print("      TRAINING FINE-GRAINED MODEL V2")
    print("=" * 70)
    
    if not TRAIN_DIR.exists():
        print("\n❌ Dataset not found. Run with --prepare first.")
        return None
    
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    print(f"\n📋 Training Configuration:")
    print(f"   • Dataset:     {OUTPUT_DIR}")
    print(f"   • Model:       {MODEL_NAME}")
    print(f"   • Epochs:      {EPOCHS}")
    print(f"   • Image Size:  {IMG_SIZE}")
    print(f"   • Batch Size:  {BATCH_SIZE}")
    print()
    
    # Train with optimized settings
    results = model.train(
        data=str(OUTPUT_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name="model_v2",
        patience=0,             # No early stopping
        save=True,
        pretrained=True,
        optimizer="AdamW",      # Better optimizer
        lr0=0.0005,             # Lower learning rate for stability
        lrf=0.01,
        weight_decay=0.01,      # Stronger regularization
        warmup_epochs=5,
        seed=SEED,
        verbose=True,
        workers=0,              # Windows compatibility
        exist_ok=True,
        
        # Strong augmentation
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=30,             # More rotation
        translate=0.2,
        scale=0.5,
        shear=10,
        perspective=0.001,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,             # Disable for classification
        mixup=0.2,              # More mixup for robustness
        erasing=0.3,            # Random erasing augmentation
        
        # Regularization
        dropout=0.3,            # Stronger dropout
        label_smoothing=0.15,   # More smoothing for imbalanced data
        
        # Cosine annealing
        cos_lr=True,
    )
    
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE!")
    print("=" * 70)
    
    best_model = RESULTS_DIR / "model_v2" / "weights" / "best.pt"
    print(f"\n📁 Best model: {best_model}")
    
    return model


def evaluate_model(model_path=None):
    """Evaluate model with detailed per-class metrics."""
    print("\n" + "=" * 70)
    print("      EVALUATING FINE-GRAINED MODEL V2")
    print("=" * 70)
    
    if model_path is None:
        model_path = RESULTS_DIR / "model_v2" / "weights" / "best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        return None
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    class_names = list(model.names.values())
    n_classes = len(class_names)
    
    print(f"   Classes: {class_names}")
    
    # Load validation data
    print(f"\n📂 Loading validation data from: {VAL_DIR}")
    
    images = []
    y_true = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = VAL_DIR / class_name
        if not class_dir.exists():
            continue
        
        class_images = list(class_dir.glob("*.[pP][nN][gG]")) + \
                       list(class_dir.glob("*.[jJ][pP][gG]"))
        
        for img_path in class_images:
            images.append(img_path)
            y_true.append(class_idx)
        
        print(f"   {class_name}: {len(class_images)} images")
    
    print(f"\n   Total: {len(images)} validation images")
    
    # Run predictions
    print(f"\n🔮 Running predictions...")
    y_pred = []
    y_probs = []
    
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            y_pred.append(-1)
            y_probs.append([0] * n_classes)
            continue
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        results = model.predict(source=img_resized, imgsz=IMG_SIZE, verbose=False)
        
        for result in results:
            probs = result.probs
            pred_class = int(probs.top1)
            y_pred.append(pred_class)
            y_probs.append(probs.data.cpu().numpy())
        
        if (i + 1) % 50 == 0:
            print(f"   Processed: {i + 1}/{len(images)}")
    
    # Filter valid predictions
    valid_idx = [i for i, p in enumerate(y_pred) if p >= 0]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]
    
    # Calculate metrics
    n_samples = len(y_true)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=range(n_classes))
    
    metrics = []
    
    for i, class_name in enumerate(class_names):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = n_samples - TP - FN - FP
        
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        accuracy = (TP + TN) / n_samples if n_samples > 0 else 0.0
        
        metrics.append({
            'Class': class_name,
            'Se': sensitivity,
            'Sp': specificity,
            'Precision': precision,
            'Accuracy': accuracy,
            'Support': int(np.sum(cm[i, :]))
        })
    
    # Print results
    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS")
    print("=" * 70)
    
    print(f"\n{'Classes':<12} {'Se':<10} {'Sp':<10} {'Precision':<12} {'Accuracy':<10} {'Support':<10}")
    print("-" * 64)
    
    for m in metrics:
        print(f"{m['Class']:<12} {m['Se']*100:>5.0f}%     {m['Sp']*100:>5.0f}%     {m['Precision']*100:>5.0f}%       {m['Accuracy']*100:>5.0f}%     {m['Support']:>5}")
    
    print("-" * 64)
    
    # Weighted averages (by support)
    total_support = sum(m['Support'] for m in metrics)
    w_avg_se = sum(m['Se'] * m['Support'] for m in metrics) / total_support if total_support > 0 else 0
    w_avg_sp = sum(m['Sp'] * m['Support'] for m in metrics) / total_support if total_support > 0 else 0
    w_avg_prec = sum(m['Precision'] * m['Support'] for m in metrics) / total_support if total_support > 0 else 0
    w_avg_acc = sum(m['Accuracy'] * m['Support'] for m in metrics) / total_support if total_support > 0 else 0
    
    print(f"{'Weighted Avg':<12} {w_avg_se*100:>5.0f}%     {w_avg_sp*100:>5.0f}%     {w_avg_prec*100:>5.0f}%       {w_avg_acc*100:>5.0f}%     {total_support:>5}")
    print("=" * 70)
    
    overall_acc = np.mean(y_true_arr == y_pred_arr) * 100
    print(f"\n📈 Overall Accuracy: {overall_acc:.2f}%")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(metrics)
    df.to_csv(RESULTS_DIR / "metrics_v2.csv", index=False)
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix_v2.csv")
    
    print(f"\n📄 Results saved to: {RESULTS_DIR}")
    
    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="Fine-Grained Morphology V2")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--model", type=str, help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    if args.all or args.prepare:
        prepare_dataset_with_augmentation()
    
    if args.all or args.train:
        train_model()
    
    if args.all or args.evaluate:
        evaluate_model(args.model)
    
    if not any([args.prepare, args.train, args.evaluate, args.all]):
        print("Usage:")
        print("  python train_finegrained_v2.py --all")


if __name__ == "__main__":
    main()
