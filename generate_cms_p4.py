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
from PIL import Image

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "classification_dataset_balanced" / "val"
OUTPUT_DIR = BASE_DIR / "confusion_matrices"

# P1 Model paths (P4 is the ensemble of P1 models)
YOLO_P1_PATH = BASE_DIR / "priority_1_class_weights" / "yolo_runs" / "sperm_cls_weighted" / "weights" / "best.pt"
EFFNET_P1_PATH = BASE_DIR / "priority_1_class_weights" / "weights" / "efficientnet_b0_weighted.pt"

IMG_SIZE = 224

# P4 Ensemble Weights (as empirically determined during testing)
YOLO_WEIGHT = 0.4
EFFNET_WEIGHT = 0.6

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

def load_effnet(path, device, num_classes):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    try:
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(num_ftrs, num_classes))
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.35, inplace=True), nn.Linear(model.classifier[1].in_features, num_classes))
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            model.load_state_dict(torch.load(path, map_location=device))
            
    model.to(device).eval()
    return model

def load_yolo(path):
    return YOLO(str(path))

def evaluate_ensemble(device):
    print("\n--- Evaluating P4 Ensemble (YOLO_P1 + EffNet_P1) ---")
    if not YOLO_P1_PATH.exists() or not EFFNET_P1_PATH.exists():
        print("❌ Cannot run Ensemble P4: Missing P1 base models.")
        return None, None
        
    val_dataset = datasets.ImageFolder(str(DATA_DIR))
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")
    
    yolo_model = load_yolo(YOLO_P1_PATH)
    effnet_model = load_effnet(EFFNET_P1_PATH, device, num_classes)
    
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_preds = []
    all_labels = []
    
    print(f"Validating over {len(val_dataset.imgs)} images...")
    
    for i, (img_path, label_idx) in enumerate(val_dataset.imgs):
        # Read the raw image and convert to RGB (since Ultralytics uses BGR by default, 
        # but ImageFolder uses PIL RGB. We need consistent input for both)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Get YOLO probabilities
        res = yolo_model.predict(img_bgr, imgsz=IMG_SIZE, verbose=False)
        y_probs_tensor = res[0].probs.data.cpu().numpy()
        yolo_names = res[0].names
        yolo_probs = np.zeros(num_classes)
        for yolo_idx, name in yolo_names.items():
            if name in class_names:
                yolo_probs[class_names.index(name)] = y_probs_tensor[yolo_idx]
                
        # 2. Get EffNet probabilities
        pil_img = Image.fromarray(img_rgb)
        with torch.no_grad():
            out = effnet_model(tfm(pil_img).unsqueeze(0).to(device))
            effnet_probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
            
        # 3. Soft-Voting Ensemble
        combined_probs = YOLO_WEIGHT * yolo_probs + EFFNET_WEIGHT * effnet_probs
        combined_probs = combined_probs / combined_probs.sum()  # Normalize
        
        pred_idx = np.argmax(combined_probs)
        
        all_preds.append(pred_idx)
        all_labels.append(label_idx)
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_dataset.imgs)}")
            
    return all_labels, all_preds, class_names

def generate_cms():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluate Ensemble P4
    y_true, y_pred, class_names = evaluate_ensemble(device)
    if y_true:
        plot_cm(y_true, y_pred, class_names, "Ensemble Priority 4 (P1 YOLO + P1 EffNet) - Confusion Matrix", "cm_ensemble_p4.png")
        
    print(f"\n✅ All P4 matrices saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_cms()
