"""
Morphology Defects Evaluation Metrics Script
=============================================
Computes per-class metrics for sperm morphology classification:
- Sensitivity (Se) = TP / (TP + FN) - Recall
- Specificity (Sp) = TN / (TN + FP)
- Precision = TP / (TP + FP)
- Accuracy = (TP + TN) / Total

Usage:
    python evaluate_morphology.py
    python evaluate_morphology.py --model path/to/best.pt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import os
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")

# Model path
MODEL_PATH = BASE_DIR / "runs" / "smdss_sperm_cls" / "weights" / "best.pt"

# Dataset paths
VAL_DIR = BASE_DIR / "classification_dataset" / "val"

# Image size for classification
IMG_SIZE = 224

# Device
DEVICE = "0"

# Output directory
OUTPUT_DIR = BASE_DIR / "sperm_pipeline" / "evaluation_results"

# ══════════════════════════════════════════════════════════════════════════════


def load_validation_data(val_dir):
    """
    Load validation images and their ground truth labels.
    
    Returns:
        images: List of image paths
        labels: List of class indices
        class_names: List of class names
    """
    val_dir = Path(val_dir)
    images = []
    labels = []
    class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"\n📂 Loading validation data from: {val_dir}")
    print(f"   Classes found: {class_names}")
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = val_dir / class_name
        class_images = list(class_dir.glob("*.[jJ][pP][gG]")) + \
                       list(class_dir.glob("*.[pP][nN][gG]")) + \
                       list(class_dir.glob("*.[jJ][pP][eE][gG]"))
        
        for img_path in class_images:
            images.append(img_path)
            labels.append(class_idx)
        
        print(f"   {class_name}: {len(class_images)} images")
    
    print(f"\n   Total: {len(images)} images")
    return images, labels, class_names


def predict_batch(model, images, img_size=224):
    """
    Run predictions on a list of images.
    
    Returns:
        predictions: List of predicted class indices
        confidences: List of confidence scores
    """
    predictions = []
    confidences = []
    
    print(f"\n🔮 Running predictions on {len(images)} images...")
    
    for i, img_path in enumerate(images):
        # Load and resize image
        img = cv2.imread(str(img_path))
        if img is None:
            predictions.append(-1)
            confidences.append(0.0)
            continue
        
        img_resized = cv2.resize(img, (img_size, img_size))
        
        # Predict
        results = model.predict(source=img_resized, imgsz=img_size, verbose=False)
        
        for result in results:
            probs = result.probs
            pred_class = int(probs.top1)
            pred_conf = float(probs.top1conf)
            predictions.append(pred_class)
            confidences.append(pred_conf)
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"   Processed: {i + 1}/{len(images)}")
    
    return predictions, confidences


def calculate_per_class_metrics(y_true, y_pred, class_names):
    """
    Calculate Sensitivity, Specificity, Precision, Accuracy for each class.
    
    Returns:
        DataFrame with metrics for each class
    """
    n_classes = len(class_names)
    n_samples = len(y_true)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    metrics = []
    
    for i, class_name in enumerate(class_names):
        # For class i:
        # TP = correctly predicted as class i
        # FN = actually class i but predicted as other
        # FP = predicted as class i but actually other
        # TN = not class i and not predicted as class i
        
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP  # Row sum minus TP
        FP = np.sum(cm[:, i]) - TP  # Column sum minus TP
        TN = n_samples - TP - FN - FP
        
        # Calculate metrics
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Recall
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        accuracy = (TP + TN) / n_samples if n_samples > 0 else 0.0
        
        # F1 Score (bonus metric)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        metrics.append({
            'Class': class_name,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'Se (Sensitivity)': f"{sensitivity * 100:.0f}%",
            'Sp (Specificity)': f"{specificity * 100:.0f}%",
            'Precision': f"{precision * 100:.0f}%",
            'Accuracy': f"{accuracy * 100:.0f}%",
            'F1-Score': f"{f1 * 100:.0f}%",
            # Raw values for sorting/analysis
            'Se_raw': sensitivity,
            'Sp_raw': specificity,
            'Precision_raw': precision,
            'Accuracy_raw': accuracy,
            'F1_raw': f1
        })
    
    return pd.DataFrame(metrics), cm


def print_metrics_table(df, class_names):
    """
    Print metrics in a formatted table similar to the image.
    """
    print("\n" + "=" * 80)
    print("                    MORPHOLOGY DEFECTS - EVALUATION METRICS")
    print("=" * 80)
    
    # Print header
    print(f"\n{'Classes':<20} {'Se':<12} {'Sp':<12} {'Precision':<12} {'Accuracy':<12}")
    print("-" * 68)
    
    # Print each class
    for _, row in df.iterrows():
        print(f"{row['Class']:<20} {row['Se (Sensitivity)']:<12} {row['Sp (Specificity)']:<12} {row['Precision']:<12} {row['Accuracy']:<12}")
    
    print("-" * 68)
    
    # Calculate and print averages
    avg_se = df['Se_raw'].mean() * 100
    avg_sp = df['Sp_raw'].mean() * 100
    avg_prec = df['Precision_raw'].mean() * 100
    avg_acc = df['Accuracy_raw'].mean() * 100
    
    print(f"{'Average':<20} {avg_se:.0f}%{'':<8} {avg_sp:.0f}%{'':<8} {avg_prec:.0f}%{'':<8} {avg_acc:.0f}%")
    print("=" * 80)


def print_confusion_matrix(cm, class_names):
    """
    Print the confusion matrix.
    """
    print("\n" + "=" * 80)
    print("                         CONFUSION MATRIX")
    print("=" * 80)
    
    # Header
    header = "Actual \\ Predicted"
    print(f"\n{header:<20}", end="")
    for name in class_names:
        short_name = name[:8] if len(name) > 8 else name
        print(f"{short_name:>10}", end="")
    print()
    print("-" * (20 + 10 * len(class_names)))
    
    # Matrix rows
    for i, name in enumerate(class_names):
        short_name = name[:18] if len(name) > 18 else name
        print(f"{short_name:<20}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    print("=" * 80)


def save_results(df, cm, class_names, output_dir):
    """
    Save evaluation results to files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    metrics_path = output_dir / "metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(f"\n📄 Metrics saved to: {metrics_path}")
    
    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"📄 Confusion matrix saved to: {cm_path}")
    
    # Save formatted report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("MORPHOLOGY DEFECTS - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Classes':<20} {'Se':<12} {'Sp':<12} {'Precision':<12} {'Accuracy':<12}\n")
        f.write("-" * 68 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['Class']:<20} {row['Se (Sensitivity)']:<12} {row['Sp (Specificity)']:<12} {row['Precision']:<12} {row['Accuracy']:<12}\n")
        
        f.write("-" * 68 + "\n")
        avg_se = df['Se_raw'].mean() * 100
        avg_sp = df['Sp_raw'].mean() * 100
        avg_prec = df['Precision_raw'].mean() * 100
        avg_acc = df['Accuracy_raw'].mean() * 100
        f.write(f"{'Average':<20} {avg_se:.0f}%{'':<8} {avg_sp:.0f}%{'':<8} {avg_prec:.0f}%{'':<8} {avg_acc:.0f}%\n")
        
        f.write("\n\nCONFUSION MATRIX\n")
        f.write("=" * 70 + "\n")
        f.write(cm_df.to_string())
    
    print(f"📄 Report saved to: {report_path}")


def evaluate(model_path=None, val_dir=None):
    """
    Main evaluation function.
    """
    if model_path is None:
        model_path = MODEL_PATH
    if val_dir is None:
        val_dir = VAL_DIR
    
    print("=" * 80)
    print("          MORPHOLOGY CLASSIFICATION - MODEL EVALUATION")
    print("=" * 80)
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the classification model first!")
        return
    
    # Load model
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Load validation data
    images, y_true, class_names = load_validation_data(val_dir)
    
    if len(images) == 0:
        print("\n❌ No validation images found!")
        return
    
    # Run predictions
    y_pred, confidences = predict_batch(model, images, IMG_SIZE)
    
    # Filter out failed predictions
    valid_indices = [i for i, p in enumerate(y_pred) if p >= 0]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    df, cm = calculate_per_class_metrics(y_true_valid, y_pred_valid, class_names)
    
    # Print results
    print_metrics_table(df, class_names)
    print_confusion_matrix(cm, class_names)
    
    # Overall metrics
    overall_accuracy = np.mean(np.array(y_true_valid) == np.array(y_pred_valid))
    print(f"\n📈 Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"📈 Total Samples: {len(y_true_valid)}")
    
    # Save results
    save_results(df, cm, class_names, OUTPUT_DIR)
    
    return df, cm


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Morphology Classification Model")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--val-dir", type=str, default=None, help="Path to validation directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    
    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = Path(args.output)
    
    evaluate(
        model_path=args.model,
        val_dir=args.val_dir
    )


if __name__ == "__main__":
    main()
