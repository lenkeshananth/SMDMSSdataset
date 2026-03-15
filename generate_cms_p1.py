import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "classification_dataset_balanced" / "val"
OUTPUT_DIR = BASE_DIR / "confusion_matrices"

# P1 Model paths
YOLO_P1_PATH = BASE_DIR / "priority_1_class_weights" / "yolo_runs" / "sperm_cls_weighted" / "weights" / "best.pt"
EFFNET_P1_PATH = BASE_DIR / "priority_1_class_weights" / "weights" / "efficientnet_b0_weighted.pt"

IMG_SIZE = 224
BATCH_SIZE = 32

def plot_cm(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()
    print(f"✅ Saved visually plotted confusion matrix to {OUTPUT_DIR / filename}")

def evaluate_yolo(class_names):
    print("\n--- Evaluating YOLOv8 Priority 1 (Class Weights) ---")
    if not YOLO_P1_PATH.exists():
        print("❌ YOLO P1 weights not found.")
        return None, None
        
    model = YOLO(str(YOLO_P1_PATH))
    all_preds = []
    all_labels = []
    
    val_dataset = datasets.ImageFolder(str(DATA_DIR))
    print(f"Validating over {len(val_dataset.imgs)} images...")
    
    for i, (img_path, label_idx) in enumerate(val_dataset.imgs):
        results = model.predict(img_path, imgsz=IMG_SIZE, verbose=False)
        pred_idx = results[0].probs.top1
        
        all_preds.append(pred_idx)
        all_labels.append(label_idx)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_dataset.imgs)}")
            
    return all_labels, all_preds

def evaluate_efficientnet(device, num_classes, class_names):
    print("\n--- Evaluating EfficientNet-B0 Priority 1 (Class Weights) ---")
    if not EFFNET_P1_PATH.exists():
        print("❌ EfficientNet P1 weights not found.")
        return None, None
        
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    # Loading logic used for the multi-model pipeline since P1 has dropout layers
    try:
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(num_ftrs, num_classes))
        model.load_state_dict(torch.load(EFFNET_P1_PATH, map_location=device))
    except RuntimeError:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.35, inplace=True), nn.Linear(model.classifier[1].in_features, num_classes))
        try:
            model.load_state_dict(torch.load(EFFNET_P1_PATH, map_location=device))
        except RuntimeError:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            model.load_state_dict(torch.load(EFFNET_P1_PATH, map_location=device))

    model = model.to(device)
    model.eval()
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    print(f"Validating over {len(val_dataset.imgs)} images...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            print(f"  Processed batch {i+1}/{len(val_loader)}")
            
    return all_labels, all_preds

def generate_cms():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    val_dataset_dummy = datasets.ImageFolder(str(DATA_DIR))
    class_names = val_dataset_dummy.classes
    print(f"Classes found: {class_names}")
    
    # Evaluate YOLO P1
    y_true_yolo, y_pred_yolo = evaluate_yolo(class_names)
    if y_true_yolo:
        plot_cm(y_true_yolo, y_pred_yolo, class_names, "YOLOv8 Priority 1 - Confusion Matrix", "cm_yolo_p1.png")
        
    # Evaluate EfficientNet P1
    y_true_eff, y_pred_eff = evaluate_efficientnet(device, len(class_names), class_names)
    if y_true_eff:
        plot_cm(y_true_eff, y_pred_eff, class_names, "EfficientNet-B0 Priority 1 - Confusion Matrix", "cm_effnet_p1.png")
        
    print(f"\n✅ All P1 matrices saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_cms()
