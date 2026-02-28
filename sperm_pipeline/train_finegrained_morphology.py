"""
Fine-Grained Morphology Classification - Training & Evaluation
===============================================================
This script trains and evaluates on individual anomaly codes:
  Head anomalies: A, B, C, D, E, F, G
  Midpiece anomalies: H, J
  Tail anomalies: L, N, O
  Normal: Nr

Output metrics: Sensitivity (Se), Specificity (Sp), Precision, Accuracy per class.

Usage:
    python train_finegrained_morphology.py --prepare      # Prepare dataset only
    python train_finegrained_morphology.py --train        # Train model
    python train_finegrained_morphology.py --evaluate     # Evaluate model
    python train_finegrained_morphology.py --all          # Do all steps
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
import cv2
import argparse

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"

# Output directories
OUTPUT_DIR = BASE_DIR / "finegrained_classification"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
RESULTS_DIR = OUTPUT_DIR / "results"

# Fine-grained class names (matching the image)
# Note: Class names sorted alphabetically for consistency
ANOMALY_CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'N', 'O', 'Nr']

# Mapping of anomaly codes to their regions (for reference)
ANOMALY_REGIONS = {
    'A': 'Head',      # Acrosome anomaly
    'B': 'Head',      # Large head
    'C': 'Head',      # Small head
    'D': 'Head',      # Tapered head
    'E': 'Head',      # Pyriform head
    'F': 'Head',      # Amorphous head
    'G': 'Head',      # Vacuolated head
    'H': 'Midpiece',  # Bent midpiece
    'J': 'Midpiece',  # Thick/irregular midpiece
    'L': 'Tail',      # Bent tail
    'N': 'Tail',      # Coiled tail
    'O': 'Tail',      # Short tail
    'Nr': 'Normal',   # Normal sperm
}

# Training settings
MODEL_NAME = "yolov8s-cls"  # Use small model for better accuracy (was nano)
EPOCHS = 100
IMG_SIZE = 224
BATCH_SIZE = 16             # Reduced for small imbalanced dataset
TRAIN_SPLIT = 0.8
SEED = 42
DEVICE = "0"

# ══════════════════════════════════════════════════════════════════════════════


def parse_label_file(label_path):
    """
    Parse a SMDSS annotation file and extract all anomaly codes.
    Uses majority voting from 3 experts.
    
    Returns:
        List of anomaly codes (e.g., ['B', 'G']) or ['Nr'] for normal
    """
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
        print(f"Error parsing {label_path}: {e}")
        return []
    
    # Majority voting: count occurrences across experts
    all_codes = []
    for exp_codes in experts.values():
        all_codes.extend(exp_codes)
    
    if not all_codes:
        return ['NR']  # Normal if no anomalies
    
    # Count each code
    code_counts = Counter(all_codes)
    
    # Keep codes that appear in at least 2/3 experts
    majority_codes = [code for code, count in code_counts.items() if count >= 2]
    
    if not majority_codes:
        # If no majority, take the most common
        majority_codes = [code_counts.most_common(1)[0][0]]
    
    return majority_codes


def get_primary_anomaly(codes):
    """
    Get the primary anomaly class for multi-label cases.
    Priority: Head > Midpiece > Tail > Normal
    
    For the fine-grained classification, we assign each image to 
    its PRIMARY anomaly (first detected).
    """
    if not codes or codes == ['NR']:
        return 'Nr'
    
    # Normalize codes
    codes = [c.upper() for c in codes]
    
    # Priority order based on clinical significance
    priority_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L', 'N', 'O']
    
    for code in priority_order:
        if code in codes:
            return code
    
    return 'Nr'


def prepare_dataset():
    """
    Prepare the fine-grained classification dataset.
    Creates train/val folders with individual anomaly classes.
    """
    print("=" * 70)
    print("      PREPARING FINE-GRAINED CLASSIFICATION DATASET")
    print("=" * 70)
    
    # Clear existing output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    # Create class directories
    for split in ['train', 'val']:
        for cls in ANOMALY_CLASSES:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Parse all labels and organize images
    data_by_class = defaultdict(list)
    
    print(f"\n📂 Scanning labels from: {LABELS_DIR}")
    
    label_files = list(LABELS_DIR.glob("*.txt"))
    print(f"   Found {len(label_files)} label files")
    
    for label_path in label_files:
        # Get corresponding image
        img_name = label_path.stem + ".png"
        img_path = IMAGES_DIR / img_name
        
        if not img_path.exists():
            # Try jpg
            img_name = label_path.stem + ".jpg"
            img_path = IMAGES_DIR / img_name
        
        if not img_path.exists():
            continue
        
        # Parse anomaly codes
        codes = parse_label_file(label_path)
        primary_class = get_primary_anomaly(codes)
        
        if primary_class in ANOMALY_CLASSES:
            data_by_class[primary_class].append(img_path)
    
    # Print distribution
    print(f"\n📊 Class Distribution:")
    total = 0
    for cls in ANOMALY_CLASSES:
        count = len(data_by_class[cls])
        total += count
        print(f"   {cls:3}: {count:5} images")
    print(f"   {'Total':3}: {total:5} images")
    
    # Split and copy images
    print(f"\n📁 Splitting data ({TRAIN_SPLIT*100:.0f}% train, {(1-TRAIN_SPLIT)*100:.0f}% val)...")
    
    random.seed(SEED)
    train_count = 0
    val_count = 0
    
    # First pass: split data
    train_by_class = {}
    val_by_class = {}
    
    for cls in ANOMALY_CLASSES:
        images = data_by_class[cls]
        if len(images) == 0:
            train_by_class[cls] = []
            val_by_class[cls] = []
            continue
            
        random.shuffle(images)
        
        # Ensure at least 1 validation sample if possible
        if len(images) >= 2:
            split_idx = max(1, int(len(images) * TRAIN_SPLIT))
            train_by_class[cls] = images[:split_idx]
            val_by_class[cls] = images[split_idx:]
        else:
            # Only 1 image: use for both train and val
            train_by_class[cls] = images
            val_by_class[cls] = images
    
    # Find max class size for oversampling
    max_train_size = max(len(imgs) for imgs in train_by_class.values())
    print(f"\n📊 Oversampling minority classes to {max_train_size} images each...")
    
    for cls in ANOMALY_CLASSES:
        train_images = train_by_class[cls]
        val_images = val_by_class[cls]
        
        if len(train_images) == 0:
            continue
        
        # Oversample training images to balance classes
        oversampled_train = []
        while len(oversampled_train) < max_train_size:
            for img_path in train_images:
                if len(oversampled_train) >= max_train_size:
                    break
                # Create unique filename for oversampled copies
                copy_idx = len(oversampled_train)
                dst = TRAIN_DIR / cls / f"{img_path.stem}_copy{copy_idx}{img_path.suffix}"
                shutil.copy2(img_path, dst)
                oversampled_train.append(dst)
                train_count += 1
        
        # Copy validation images (no oversampling for val)
        for img_path in val_images:
            dst = VAL_DIR / cls / img_path.name
            shutil.copy2(img_path, dst)
            val_count += 1
        
        print(f"   {cls}: {len(train_images)} -> {len(oversampled_train)} (train), {len(val_images)} (val)")
    
    print(f"\n   ✓ Train: {train_count} images (balanced)")
    print(f"   ✓ Val:   {val_count} images (original)")
    print(f"   ✓ Dataset saved to: {OUTPUT_DIR}")
    
    return OUTPUT_DIR


def train_model():
    """
    Train YOLOv8 classification model on fine-grained classes.
    """
    print("\n" + "=" * 70)
    print("      TRAINING FINE-GRAINED CLASSIFICATION MODEL")
    print("=" * 70)
    
    # Check dataset exists
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
    print(f"   • Classes:     {len(ANOMALY_CLASSES)}")
    print()
    
    # Train
    results = model.train(
        data=str(OUTPUT_DIR),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name="finegrained_model",
        patience=0,
        save=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        seed=SEED,
        verbose=True,
        workers=0,              # Fix Windows shared memory error
        exist_ok=True,          # Overwrite previous run
        
        # Data augmentation to help with class imbalance
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,             # Rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,             # Disable mosaic for small dataset
        mixup=0.1,              # Light mixup augmentation
        
        # Learning rate schedule
        lrf=0.01,
        warmup_epochs=3,
        
        # Regularization
        weight_decay=0.0005,
        dropout=0.2,            # Add dropout for regularization
        label_smoothing=0.1,    # Helps with class imbalance
    )
    
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE!")
    print("=" * 70)
    
    best_model = RESULTS_DIR / "finegrained_model" / "weights" / "best.pt"
    print(f"\n📁 Best model: {best_model}")
    
    return model


def evaluate_model(model_path=None):
    """
    Evaluate the model and compute per-class metrics.
    """
    print("\n" + "=" * 70)
    print("      EVALUATING FINE-GRAINED CLASSIFICATION MODEL")
    print("=" * 70)
    
    # Get model path
    if model_path is None:
        model_path = RESULTS_DIR / "finegrained_model" / "weights" / "best.pt"
    
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Run with --train first.")
        return None
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Get class names from model
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
    
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            y_pred.append(-1)
            continue
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        results = model.predict(source=img_resized, imgsz=IMG_SIZE, verbose=False)
        
        for result in results:
            probs = result.probs
            pred_class = int(probs.top1)
            y_pred.append(pred_class)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed: {i + 1}/{len(images)}")
    
    # Filter valid predictions
    valid_idx = [i for i, p in enumerate(y_pred) if p >= 0]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]
    
    # Calculate per-class metrics
    print(f"\n📊 Calculating metrics...")
    
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
            'Accuracy': accuracy
        })
    
    # Print results table (matching the image format)
    print("\n" + "=" * 70)
    print("                    EVALUATION METRICS")
    print("=" * 70)
    
    print(f"\n{'Classes':<12} {'Se':<12} {'Sp':<12} {'Precision':<12} {'Accuracy':<12}")
    print("-" * 60)
    
    for m in metrics:
        print(f"{m['Class']:<12} {m['Se']*100:.0f}%{'':<8} {m['Sp']*100:.0f}%{'':<8} {m['Precision']*100:.0f}%{'':<8} {m['Accuracy']*100:.0f}%")
    
    print("-" * 60)
    
    # Averages
    avg_se = np.mean([m['Se'] for m in metrics]) * 100
    avg_sp = np.mean([m['Sp'] for m in metrics]) * 100
    avg_prec = np.mean([m['Precision'] for m in metrics]) * 100
    avg_acc = np.mean([m['Accuracy'] for m in metrics]) * 100
    
    print(f"{'Average':<12} {avg_se:.0f}%{'':<8} {avg_sp:.0f}%{'':<8} {avg_prec:.0f}%{'':<8} {avg_acc:.0f}%")
    print("=" * 70)
    
    # Overall accuracy
    overall_acc = np.mean(y_true_arr == y_pred_arr) * 100
    print(f"\n📈 Overall Accuracy: {overall_acc:.2f}%")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    df = pd.DataFrame(metrics)
    df['Se'] = df['Se'].apply(lambda x: f"{x*100:.0f}%")
    df['Sp'] = df['Sp'].apply(lambda x: f"{x*100:.0f}%")
    df['Precision'] = df['Precision'].apply(lambda x: f"{x*100:.0f}%")
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f"{x*100:.0f}%")
    
    csv_path = RESULTS_DIR / "finegrained_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n📄 Metrics saved to: {csv_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = RESULTS_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"📄 Confusion matrix saved to: {cm_path}")
    
    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="Fine-Grained Morphology Classification")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--model", type=str, help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    if args.all or args.prepare:
        prepare_dataset()
    
    if args.all or args.train:
        train_model()
    
    if args.all or args.evaluate:
        evaluate_model(args.model)
    
    if not any([args.prepare, args.train, args.evaluate, args.all]):
        print("Usage:")
        print("  python train_finegrained_morphology.py --prepare   # Prepare dataset")
        print("  python train_finegrained_morphology.py --train     # Train model")
        print("  python train_finegrained_morphology.py --evaluate  # Evaluate model")
        print("  python train_finegrained_morphology.py --all       # Run all steps")


if __name__ == "__main__":
    main()
