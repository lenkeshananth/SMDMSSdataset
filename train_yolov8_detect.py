"""
YOLOv8 Object Detection Training Script for Sperm Dataset
==========================================================
This script trains a YOLOv8 object detection model to detect sperm cells
using the Roboflow-exported dataset in YOLO format.

Usage:
    python train_yolov8_detect.py
"""

from ultralytics import YOLO
from pathlib import Path
import os

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Dataset paths
DATASET_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
DATA_YAML = DATASET_DIR / "sperm.v1i.yolov8" / "data.yaml"

# Model configuration
MODEL_NAME = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                           # n=nano (fastest), s=small, m=medium, l=large, x=extra-large (most accurate)

# Training hyperparameters
EPOCHS = 100               # Number of training epochs
IMG_SIZE = 640             # Input image size (standard for YOLOv8 detection)
BATCH_SIZE = 16            # Batch size (reduce if out of memory)
PATIENCE = 0              # Early stopping patience (epochs without improvement)
DEVICE = "0"               # GPU device ID ("0" for first GPU, "cpu" for CPU training)

# Output configuration
PROJECT = DATASET_DIR / "sperm.v1i.yolov8" / "runs" / "detect"
NAME = "sperm_detection"   # Experiment name

# ══════════════════════════════════════════════════════════════════════════════


def fix_data_yaml():
    """
    Fix the data.yaml to use absolute paths for reliable training.
    Creates a modified version with correct paths.
    """
    import yaml
    
    yaml_path = DATA_YAML
    fixed_yaml_path = DATASET_DIR / "sperm.v1i.yolov8" / "data_fixed.yaml"
    
    # Read original yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Fix paths to be absolute
    base_dir = DATASET_DIR / "sperm.v1i.yolov8"
    data['train'] = str(base_dir / "train" / "images")
    data['val'] = str(base_dir / "valid" / "images")
    
    # Remove test if it doesn't exist
    if 'test' in data:
        test_path = base_dir / "test" / "images"
        if not test_path.exists():
            del data['test']
        else:
            data['test'] = str(test_path)
    
    # Write fixed yaml
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✓ Created fixed data.yaml at: {fixed_yaml_path}")
    return fixed_yaml_path


def train():
    """
    Train YOLOv8 object detection model.
    """
    print("=" * 70)
    print("           YOLOv8 SPERM DETECTION TRAINING")
    print("=" * 70)
    
    # Fix data.yaml paths
    data_yaml = fix_data_yaml()
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)
    
    # Display training configuration
    print("\n📋 Training Configuration:")
    print(f"   • Dataset:     {data_yaml}")
    print(f"   • Model:       {MODEL_NAME}")
    print(f"   • Epochs:      {EPOCHS}")
    print(f"   • Image Size:  {IMG_SIZE}")
    print(f"   • Batch Size:  {BATCH_SIZE}")
    print(f"   • Device:      {DEVICE}")
    print(f"   • Output:      {PROJECT / NAME}")
    print()
    
    # Train the model
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=DEVICE,
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,        # Overwrite existing experiment
        pretrained=True,      # Use pretrained weights
        optimizer="auto",     # Automatic optimizer selection (AdamW)
        verbose=True,         # Verbose output
        seed=42,              # Reproducibility
        deterministic=True,   # Deterministic training
        
        # Data augmentation
        hsv_h=0.015,          # HSV-Hue augmentation
        hsv_s=0.7,            # HSV-Saturation augmentation
        hsv_v=0.4,            # HSV-Value augmentation
        degrees=0.0,          # Rotation degrees
        translate=0.1,        # Translation
        scale=0.5,            # Scale
        shear=0.0,            # Shear
        perspective=0.0,      # Perspective
        flipud=0.5,           # Flip up-down probability
        fliplr=0.5,           # Flip left-right probability
        mosaic=1.0,           # Mosaic augmentation
        mixup=0.0,            # Mixup augmentation
        copy_paste=0.0,       # Copy-paste augmentation
    )
    
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results saved to: {PROJECT / NAME}")
    print(f"📊 Best model: {PROJECT / NAME / 'weights' / 'best.pt'}")
    print(f"📊 Last model: {PROJECT / NAME / 'weights' / 'last.pt'}")
    
    return results


def validate(model_path=None):
    """
    Validate the trained model on the validation set.
    """
    if model_path is None:
        model_path = PROJECT / NAME / "weights" / "best.pt"
    
    print(f"\n🔍 Validating model: {model_path}")
    
    model = YOLO(str(model_path))
    data_yaml = DATASET_DIR / "sperm.v1i.yolov8" / "data_fixed.yaml"
    
    results = model.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        verbose=True,
    )
    
    return results


def predict(image_path, model_path=None, conf=0.25):
    """
    Run inference on an image.
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model (default: best.pt)
        conf: Confidence threshold
    """
    if model_path is None:
        model_path = PROJECT / NAME / "weights" / "best.pt"
    
    print(f"\n🔮 Running inference on: {image_path}")
    
    model = YOLO(str(model_path))
    
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=True,
        project=str(PROJECT),
        name="predictions",
        exist_ok=True,
    )
    
    return results


def export_model(model_path=None, format="onnx"):
    """
    Export trained model to different formats.
    
    Args:
        model_path: Path to trained model
        format: Export format (onnx, torchscript, tflite, etc.)
    """
    if model_path is None:
        model_path = PROJECT / NAME / "weights" / "best.pt"
    
    print(f"\n📤 Exporting model to {format} format...")
    
    model = YOLO(str(model_path))
    model.export(format=format, imgsz=IMG_SIZE)
    
    print(f"✓ Model exported successfully!")


if __name__ == "__main__":
    # Train the model
    train()
    
    # Optionally validate after training
    # validate()
    
    # Optionally run prediction on a test image
    # predict("path/to/test/image.jpg")
    
    # Optionally export to ONNX
    # export_model(format="onnx")
