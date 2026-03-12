"""
Improved YOLOv8 Classification Training Script for SMDSS Sperm Dataset
========================================================================
Improvements over the original:
  1. Heavy data augmentation (flips, rotations, color jitter, Gaussian blur)
  2. Minority class oversampling to balance all 5 classes
  3. Upgraded model: YOLOv8s-cls (Small) instead of Nano
  4. Lower learning rate with cosine annealing
  5. Dropout regularization (0.3)
  6. Early stopping with patience=30
  7. Extended training (200 epochs)

Classes:
  0: Normal       - No anomalies detected
  1: Head_Anomaly - Head anomalies
  2: Midpiece_Anomaly - Midpiece anomalies
  3: Tail_Anomaly - Tail anomalies
  4: Combined_Anomaly - Multiple anomaly regions
"""

import os
import sys
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# Fix Windows console encoding for emoji/unicode support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Dataset Paths
DATASET_DIR = Path(r"d:\SMDMSSdataset")
ORIGINAL_DATASET = DATASET_DIR / "classification_dataset"
AUGMENTED_DATASET = DATASET_DIR / "classification_dataset_augmented"

# Model — Upgraded from nano to small for better capacity
MODEL_NAME = "yolov8s-cls"

# Training Hyperparameters — Tuned for improvement
EPOCHS = 200
IMG_SIZE = 224
BATCH_SIZE = 16          # Smaller batch for better generalization
PATIENCE = 30            # Early stopping patience
DEVICE = "0"
SEED = 42
PROJECT_NAME = "smdss_sperm_cls_improved"

# Augmentation target — oversample minority classes to match this count
TARGET_SAMPLES_PER_CLASS = 350

# Class names
CLASS_NAMES = ["Normal", "Head_Anomaly", "Midpiece_Anomaly", "Tail_Anomaly", "Combined_Anomaly"]

# ══════════════════════════════════════════════════════════════════════════════


def augment_image(img, aug_type):
    """
    Apply a single augmentation to an image.
    
    Args:
        img: Input image (numpy array)
        aug_type: Type of augmentation to apply
    
    Returns:
        Augmented image
    """
    h, w = img.shape[:2]
    
    if aug_type == "hflip":
        return cv2.flip(img, 1)
    
    elif aug_type == "vflip":
        return cv2.flip(img, 0)
    
    elif aug_type == "hvflip":
        return cv2.flip(img, -1)
    
    elif aug_type == "rotate90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    elif aug_type == "rotate180":
        return cv2.rotate(img, cv2.ROTATE_180)
    
    elif aug_type == "rotate270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    elif aug_type == "rotate_random":
        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == "brightness":
        factor = random.uniform(0.6, 1.4)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif aug_type == "contrast":
        factor = random.uniform(0.7, 1.5)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    elif aug_type == "color_jitter":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-15, 15)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif aug_type == "gaussian_blur":
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    elif aug_type == "gaussian_noise":
        noise = np.random.normal(0, random.uniform(5, 25), img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    elif aug_type == "scale_crop":
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h))
        # Crop or pad back to original size
        if scale > 1:
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y:start_y+h, start_x:start_x+w]
        else:
            result = np.zeros_like(img)
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            return result
    
    elif aug_type == "elastic":
        # Simple elastic-like distortion
        dx = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1) * 8, (7, 7), 0)
        dy = cv2.GaussianBlur((np.random.rand(h, w).astype(np.float32) * 2 - 1) * 8, (7, 7), 0)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    elif aug_type == "compound":
        # Apply 2-3 random augmentations together
        simple_augs = ["hflip", "vflip", "brightness", "contrast", "color_jitter", "gaussian_blur", "rotate_random"]
        chosen = random.sample(simple_augs, k=random.randint(2, 3))
        result = img.copy()
        for a in chosen:
            result = augment_image(result, a)
        return result
    
    return img


AUGMENTATION_TYPES = [
    "hflip", "vflip", "hvflip", 
    "rotate90", "rotate180", "rotate270", "rotate_random",
    "brightness", "contrast", "color_jitter",
    "gaussian_blur", "gaussian_noise",
    "scale_crop", "elastic", "compound"
]


def prepare_augmented_dataset():
    """
    Create a balanced, augmented dataset:
    1. Copy all original images
    2. Oversample minority classes using data augmentation
    3. Target: ~350 images per class for training
    """
    print("=" * 70)
    print("  IMPROVED DATASET PREPARATION")
    print("  Strategy: Oversample minority classes via augmentation")
    print("=" * 70)
    
    # Clean output
    if AUGMENTED_DATASET.exists():
        shutil.rmtree(AUGMENTED_DATASET)
    
    # Create directory structure
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (AUGMENTED_DATASET / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "val"]:
        print(f"\n📁 Processing {split} split:")
        src_split = ORIGINAL_DATASET / split
        dst_split = AUGMENTED_DATASET / split
        
        class_counts = {}
        
        for cls in CLASS_NAMES:
            src_cls_dir = src_split / cls
            dst_cls_dir = dst_split / cls
            
            if not src_cls_dir.exists():
                print(f"   ⚠️ {cls}: directory not found, skipping")
                class_counts[cls] = 0
                continue
            
            # Get all original images
            images = sorted([f for f in src_cls_dir.iterdir() 
                           if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']])
            
            original_count = len(images)
            
            # Step 1: Copy all originals
            for img_path in images:
                shutil.copy2(img_path, dst_cls_dir / img_path.name)
            
            # Step 2: Augment for training split only
            if split == "train" and original_count < TARGET_SAMPLES_PER_CLASS:
                needed = TARGET_SAMPLES_PER_CLASS - original_count
                aug_count = 0
                
                while aug_count < needed:
                    # Pick a random source image
                    src_img_path = random.choice(images)
                    img = cv2.imread(str(src_img_path))
                    
                    if img is None:
                        continue
                    
                    # Pick a random augmentation
                    aug_type = random.choice(AUGMENTATION_TYPES)
                    
                    try:
                        aug_img = augment_image(img, aug_type)
                        
                        if aug_img is not None and aug_img.size > 0:
                            aug_name = f"{src_img_path.stem}_aug{aug_count:04d}_{aug_type}.png"
                            cv2.imwrite(str(dst_cls_dir / aug_name), aug_img)
                            aug_count += 1
                    except Exception as e:
                        continue
                
                total = original_count + aug_count
                print(f"   {cls:25} original: {original_count:4}  + augmented: {aug_count:4}  = total: {total:4}")
                class_counts[cls] = total
            else:
                # For val split: also augment to have at least some samples
                if split == "val" and original_count < 10:
                    val_needed = min(10, max(5, original_count * 3)) - original_count
                    aug_count = 0
                    
                    while aug_count < val_needed:
                        src_img_path = random.choice(images)
                        img = cv2.imread(str(src_img_path))
                        if img is None:
                            continue
                        
                        aug_type = random.choice(["hflip", "vflip", "brightness", "contrast"])
                        try:
                            aug_img = augment_image(img, aug_type)
                            if aug_img is not None and aug_img.size > 0:
                                aug_name = f"{src_img_path.stem}_vaug{aug_count:03d}_{aug_type}.png"
                                cv2.imwrite(str(dst_cls_dir / aug_name), aug_img)
                                aug_count += 1
                        except:
                            continue
                    
                    total = original_count + aug_count
                    print(f"   {cls:25} original: {original_count:4}  + augmented: {aug_count:4}  = total: {total:4}")
                    class_counts[cls] = total
                else:
                    print(f"   {cls:25} original: {original_count:4}  (no augmentation needed)")
                    class_counts[cls] = original_count
        
        # Print summary
        total_all = sum(class_counts.values())
        print(f"\n   {'TOTAL':25} {total_all:4} images")
    
    print(f"\n✅ Augmented dataset saved to: {AUGMENTED_DATASET}")
    return AUGMENTED_DATASET


def train_improved_model(data_path):
    """
    Train an improved YOLOv8 classification model with:
    - Larger model (yolov8s-cls)
    - Lower learning rate
    - Dropout regularization
    - Cosine LR schedule
    - Early stopping with patience=30
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 70)
    print("  IMPROVED YOLOv8 Classification Training")
    print("=" * 70)
    print(f"  Model:          {MODEL_NAME} (upgraded from nano)")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Image Size:     {IMG_SIZE}")
    print(f"  Batch Size:     {BATCH_SIZE}")
    print(f"  Patience:       {PATIENCE}")
    print(f"  Device:         {DEVICE}")
    print(f"  Data Path:      {data_path}")
    print(f"  Dropout:        0.3")
    print(f"  Learning Rate:  0.0005 (reduced from 0.001)")
    print(f"  LR Schedule:    Cosine annealing")
    print("=" * 70)
    
    # Load pretrained YOLOv8-Small classification model
    model = YOLO(f"{MODEL_NAME}.pt")
    
    # Train with improved hyperparameters
    results = model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(DATASET_DIR / "runs"),
        name=PROJECT_NAME,
        exist_ok=True,
        patience=PATIENCE,           # Early stopping
        save=True,
        save_period=10,
        pretrained=True,
        
        # Optimizer settings — reduced LR + cosine annealing
        optimizer="AdamW",           # AdamW with weight decay
        lr0=0.0005,                  # Lower initial LR (was 0.001)
        lrf=0.01,                    # Final LR = 0.01 * lr0
        cos_lr=True,                 # Cosine LR schedule
        weight_decay=0.001,          # Increased weight decay (was 0.0005)
        warmup_epochs=10,            # Longer warmup (was 5)
        
        # Regularization
        dropout=0.3,                 # Dropout for regularization
        
        # Built-in augmentations
        hsv_h=0.02,                  # HSV-Hue (slightly increased)
        hsv_s=0.7,                   # HSV-Saturation
        hsv_v=0.4,                   # HSV-Value
        degrees=15.0,                # Rotation (was 0.0)
        translate=0.15,              # Translation (was 0.1)
        scale=0.5,                   # Scale
        shear=5.0,                   # Shear (was 0.0)
        perspective=0.0001,          # Slight perspective
        flipud=0.5,                  # Vertical flip (was 0.0)
        fliplr=0.5,                  # Horizontal flip
        erasing=0.3,                 # Random erasing (was 0.4)
        auto_augment="randaugment",  # Auto augmentation
        
        # Other
        seed=SEED,
        workers=4,
        verbose=True,
    )
    
    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    
    # Validate
    print("\n🔍 Running validation...")
    metrics = model.val()
    
    print(f"\n  📊 Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"  📊 Top-5 Accuracy: {metrics.top5:.4f}")
    
    best_model = DATASET_DIR / "runs" / PROJECT_NAME / "weights" / "best.pt"
    print(f"\n  📁 Best model saved at: {best_model}")
    
    return model, results


def evaluate_improved(model):
    """
    Run detailed per-class evaluation on the improved model.
    """
    import pandas as pd
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("\n" + "=" * 70)
    print("  DETAILED EVALUATION")
    print("=" * 70)
    
    val_dir = AUGMENTED_DATASET / "val"
    
    # Load validation data
    images = []
    y_true = []
    class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = val_dir / class_name
        class_images = list(class_dir.glob("*.[pP][nN][gG]")) + \
                       list(class_dir.glob("*.[jJ][pP][gG]")) + \
                       list(class_dir.glob("*.[jJ][pP][eE][gG]"))
        
        for img_path in class_images:
            images.append(img_path)
            y_true.append(class_idx)
        
        print(f"   {class_name}: {len(class_images)} images")
    
    print(f"\n   Total: {len(images)} images")
    
    # Run predictions
    y_pred = []
    confidences = []
    
    print(f"\n🔮 Running predictions...")
    
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            y_pred.append(-1)
            confidences.append(0.0)
            continue
        
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        results = model.predict(source=img_resized, imgsz=IMG_SIZE, verbose=False)
        
        for result in results:
            probs = result.probs
            y_pred.append(int(probs.top1))
            confidences.append(float(probs.top1conf))
        
        if (i + 1) % 50 == 0:
            print(f"   Processed: {i + 1}/{len(images)}")
    
    # Filter valid
    valid = [(t, p) for t, p in zip(y_true, y_pred) if p >= 0]
    y_true_v = [v[0] for v in valid]
    y_pred_v = [v[1] for v in valid]
    
    # Per-class metrics
    n_classes = len(class_names)
    n_samples = len(y_true_v)
    cm = confusion_matrix(y_true_v, y_pred_v, labels=range(n_classes))
    
    print("\n" + "=" * 80)
    print("            IMPROVED MODEL - EVALUATION METRICS")
    print("=" * 80)
    print(f"\n{'Class':<25} {'Se':>8} {'Sp':>8} {'Prec':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 73)
    
    metrics_data = []
    
    for i, cls_name in enumerate(class_names):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = n_samples - TP - FN - FP
        
        se = TP / (TP + FN) if (TP + FN) > 0 else 0
        sp = TN / (TN + FP) if (TN + FP) > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        acc = (TP + TN) / n_samples if n_samples > 0 else 0
        f1 = 2 * (prec * se) / (prec + se) if (prec + se) > 0 else 0
        
        print(f"   {cls_name:<23} {se*100:>6.1f}% {sp*100:>6.1f}% {prec*100:>6.1f}% {acc*100:>6.1f}% {f1*100:>6.1f}%")
        
        metrics_data.append({
            'Class': cls_name,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
            'Sensitivity': f"{se*100:.0f}%",
            'Specificity': f"{sp*100:.0f}%",
            'Precision': f"{prec*100:.0f}%",
            'Accuracy': f"{acc*100:.0f}%",
            'F1-Score': f"{f1*100:.0f}%",
            'Se_raw': se, 'Sp_raw': sp, 'Precision_raw': prec, 'Accuracy_raw': acc, 'F1_raw': f1
        })
    
    print("-" * 73)
    
    # Averages
    avg_se = np.mean([m['Se_raw'] for m in metrics_data]) * 100
    avg_sp = np.mean([m['Sp_raw'] for m in metrics_data]) * 100
    avg_prec = np.mean([m['Precision_raw'] for m in metrics_data]) * 100
    avg_acc = np.mean([m['Accuracy_raw'] for m in metrics_data]) * 100
    avg_f1 = np.mean([m['F1_raw'] for m in metrics_data]) * 100
    
    print(f"   {'Average':<23} {avg_se:>6.1f}% {avg_sp:>6.1f}% {avg_prec:>6.1f}% {avg_acc:>6.1f}% {avg_f1:>6.1f}%")
    print("=" * 80)
    
    overall_acc = np.mean(np.array(y_true_v) == np.array(y_pred_v))
    print(f"\n📈 Overall Accuracy: {overall_acc * 100:.2f}%")
    
    # Confusion Matrix
    print("\n" + "=" * 80)
    print("                    CONFUSION MATRIX")
    print("=" * 80)
    
    actual_pred = "Actual \\ Predicted"
    header = f"{actual_pred:<20}"
    for name in class_names:
        short = name[:10]
        header += f"{short:>12}"
    print(header)
    print("-" * (20 + 12 * len(class_names)))
    
    for i, name in enumerate(class_names):
        row = f"   {name[:18]:<18}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>12}"
        print(row)
    
    # Save results
    output_dir = DATASET_DIR / "sperm_pipeline" / "evaluation_results_improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "metrics_improved.csv", index=False)
    
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_dir / "confusion_matrix_improved.csv")
    
    # Save comparison report
    with open(output_dir / "comparison_report.txt", 'w') as f:
        f.write("IMPROVED vs ORIGINAL MODEL COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Metric':<25} {'Original':>12} {'Improved':>12}\n")
        f.write("-" * 49 + "\n")
        
        # Original results
        orig_se = [0.6395, 0.6386, 0.0, 0.0, 0.0]
        orig_sp = [0.5948, 0.6050, 1.0, 1.0, 1.0]
        orig_prec = [0.5392, 0.53, 0.0, 0.0, 0.0]
        orig_acc = [0.6139, 0.6188, 0.9752, 0.8812, 0.9802]
        orig_f1 = [0.5851, 0.5792, 0.0, 0.0, 0.0]
        
        f.write(f"{'Avg Sensitivity':<25} {np.mean(orig_se)*100:>10.1f}% {avg_se:>10.1f}%\n")
        f.write(f"{'Avg Specificity':<25} {np.mean(orig_sp)*100:>10.1f}% {avg_sp:>10.1f}%\n")
        f.write(f"{'Avg Precision':<25} {np.mean(orig_prec)*100:>10.1f}% {avg_prec:>10.1f}%\n")
        f.write(f"{'Avg Accuracy':<25} {np.mean(orig_acc)*100:>10.1f}% {avg_acc:>10.1f}%\n")
        f.write(f"{'Avg F1-Score':<25} {np.mean(orig_f1)*100:>10.1f}% {avg_f1:>10.1f}%\n")
        f.write(f"{'Overall Accuracy':<25} {'~53%':>12} {overall_acc*100:>10.1f}%\n")
        
        f.write("\n\nPer-class:\n")
        for i, cls_name in enumerate(class_names):
            f.write(f"\n{cls_name}:\n")
            f.write(f"  Sensitivity:  {orig_se[i]*100:>6.1f}%  ->  {metrics_data[i]['Se_raw']*100:>6.1f}%\n")
            f.write(f"  Specificity:  {orig_sp[i]*100:>6.1f}%  ->  {metrics_data[i]['Sp_raw']*100:>6.1f}%\n")
            f.write(f"  Precision:    {orig_prec[i]*100:>6.1f}%  ->  {metrics_data[i]['Precision_raw']*100:>6.1f}%\n")
            f.write(f"  F1-Score:     {orig_f1[i]*100:>6.1f}%  ->  {metrics_data[i]['F1_raw']*100:>6.1f}%\n")
    
    print(f"\n📄 Improved metrics saved to: {output_dir}")
    print(f"📄 Comparison report saved to: {output_dir / 'comparison_report.txt'}")
    
    return df, cm


# ══════════════════════════════════════════════════════════════════════════════
#                                  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    
    print("🚀 IMPROVED SPERM MORPHOLOGY CLASSIFICATION TRAINING")
    print("=" * 70)
    print("  Improvements Applied:")
    print("  ✅ 1. Heavy data augmentation (15 augmentation types)")
    print("  ✅ 2. Minority class oversampling (target: 350/class)")
    print("  ✅ 3. Larger model: YOLOv8s-cls (Small)")
    print("  ✅ 4. Lower learning rate: 0.0005 with cosine annealing")
    print("  ✅ 5. Dropout regularization: 0.3")
    print("  ✅ 6. Early stopping: patience=30")
    print("  ✅ 7. Extended training: 200 epochs")
    print("  ✅ 8. AdamW optimizer with increased weight decay")
    print("  ✅ 9. Rotation, shear, perspective augmentation enabled")
    print("  ✅ 10. Vertical flip enabled (sperm orientation varies)")
    print("=" * 70)
    
    # Step 1: Prepare augmented dataset
    data_path = prepare_augmented_dataset()
    
    # Step 2: Train improved model
    model, results = train_improved_model(data_path)
    
    # Step 3: Evaluate and compare
    evaluate_improved(model)
    
    print("\n" + "=" * 70)
    print("  ✅ IMPROVED TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
