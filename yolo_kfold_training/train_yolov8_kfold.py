import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATASET_PATH = BASE_DIR / "classification_dataset_balanced"
KFOLD_DIR = BASE_DIR / "yolo_kfold_training" / "kfold_datasets"
EPOCHS = 30
IMGSZ = 224
K_FOLDS = 5

def create_kfold_datasets():
    """Reads the balanced dataset and creates K distinct train/val splits."""
    print(f"Reading dataset from {DATASET_PATH}")
    
    # The dataset has 'train' and 'val' subdirectories, with class folders inside those.
    # We want to combine them all for StratifiedKFold.
    train_dir = DATASET_PATH / "train"
    
    if not train_dir.exists():
        print(f"Error: Could not find train dir at {train_dir}")
        return []
        
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    images = []
    labels = []
    
    # Crawl both train and val directories to gather all images
    for split_dir in [DATASET_PATH / "train", DATASET_PATH / "val"]:
        if not split_dir.exists():
            continue
            
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.rglob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append(img_path)
                    labels.append(cls_idx)
                
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Total images: {len(images)}")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    
    fold_paths = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        fold += 1  # 1-indexed
        fold_dir = KFOLD_DIR / f"fold_{fold}"
        fold_paths.append(fold_dir)
        
        if fold_dir.exists():
            print(f"Fold {fold} dataset already exists.")
            continue
            
        print(f"Creating Split for Fold {fold}...")
        for cls_name in classes:
            (fold_dir / "train" / cls_name).mkdir(parents=True, exist_ok=True)
            (fold_dir / "val" / cls_name).mkdir(parents=True, exist_ok=True)
            
        # Copy train images
        for idx in train_idx:
            img_path = images[idx]
            cls_name = img_path.parent.name
            shutil.copy(img_path, fold_dir / "train" / cls_name / img_path.name)
            
        # Copy val images
        for idx in val_idx:
            img_path = images[idx]
            cls_name = img_path.parent.name
            shutil.copy(img_path, fold_dir / "val" / cls_name / img_path.name)
            
    return fold_paths

def train_kfold():
    import numpy as np # import here to avoid global scope issues if ran incorrectly
    fold_paths = create_kfold_datasets()
    
    print("\n" + "="*50)
    print(f"STARTING {K_FOLDS}-FOLD CROSS VALIDATION")
    print("="*50)
    
    for fold, fold_dir in enumerate(fold_paths, start=1):
        print(f"\n▶ Training Fold {fold}/{K_FOLDS}")
        print(f"  Dataset: {fold_dir}")
        
        # Initialize a fresh YOLO classification model
        model = YOLO("yolov8s-cls.pt")
        
        # Train on this fold
        model.train(
            data=str(fold_dir),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            project=str(BASE_DIR / "yolo_kfold_training" / "runs"),
            name=f"kfold_{fold}",
            device="0",  # Change to "cpu" if no GPU
            verbose=False
        )
        
    print("\n✅ K-Fold Training Complete!")
    print(f"Check yolo_kfold_training/runs/kfold_1..{K_FOLDS} for weights and metrics.")

if __name__ == "__main__":
    import numpy as np
    train_kfold()
