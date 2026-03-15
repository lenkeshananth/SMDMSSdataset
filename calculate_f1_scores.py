import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, f1_score
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "priority_3_large_augmented_dataset" / "val"

# All Model Paths
MODELS = {
    "YOLO_Orig": BASE_DIR / "runs" / "classify" / "runs" / "sperm_cls_balanced" / "weights" / "best.pt",
    "EffNet_Orig": BASE_DIR / "efficientnet_training" / "weights" / "efficientnet_b0_best.pt",
    "YOLO_P1": BASE_DIR / "priority_1_class_weights" / "yolo_runs" / "sperm_cls_weighted" / "weights" / "best.pt",
    "EffNet_P1": BASE_DIR / "priority_1_class_weights" / "weights" / "efficientnet_b0_weighted.pt",
    "YOLO_P3": BASE_DIR / "priority_3_augmentation" / "yolo_runs" / "sperm_cls_augmented" / "weights" / "best.pt",
    "EffNet_P3": BASE_DIR / "priority_3_augmentation" / "weights" / "efficientnet_b0_augmented.pt",
}

IMG_SIZE = 224
BATCH_SIZE = 32

# P4 Ensemble Weights 
YOLO_WEIGHT = 0.4
EFFNET_WEIGHT = 0.6

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

def evaluate_models():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    val_dataset = datasets.ImageFolder(str(DATA_DIR))
    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}\n")
    
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load all models
    loaded_models = {}
    for name, path in MODELS.items():
        if "YOLO" in name:
            loaded_models[name] = load_yolo(path)
        else:
            loaded_models[name] = load_effnet(path, device, num_classes)
            
    results_true = []
    results_preds = {name: [] for name in loaded_models.keys()}
    results_preds["Ensemble_P4"] = []
    
    print(f"Evaluating {len(val_dataset.imgs)} images across all 7 models...")
    
    for i, (img_path, label_idx) in enumerate(val_dataset.imgs):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        effnet_input = tfm(pil_img).unsqueeze(0).to(device)
        
        results_true.append(label_idx)
        
        # Track predictions for ensemble
        yolo_p1_probs = np.zeros(num_classes)
        effnet_p1_probs = np.zeros(num_classes)
        
        for name, model in loaded_models.items():
            if "YOLO" in name:
                res = model.predict(img_bgr, imgsz=IMG_SIZE, verbose=False)
                pred_idx = res[0].probs.top1
                results_preds[name].append(pred_idx)
                
                # Save P1 probs for ensemble
                if name == "YOLO_P1":
                    y_probs_tensor = res[0].probs.data.cpu().numpy()
                    yolo_names = res[0].names
                    for yolo_idx, cname in yolo_names.items():
                        if cname in class_names:
                            yolo_p1_probs[class_names.index(cname)] = y_probs_tensor[yolo_idx]
            else:
                with torch.no_grad():
                    out = model(effnet_input)
                    probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy()
                    pred_idx = np.argmax(probs)
                    results_preds[name].append(pred_idx)
                    
                    if name == "EffNet_P1":
                        effnet_p1_probs = probs
                        
        # Calculate Ensemble P4
        combined_probs = YOLO_WEIGHT * yolo_p1_probs + EFFNET_WEIGHT * effnet_p1_probs
        combined_probs = combined_probs / combined_probs.sum()
        results_preds["Ensemble_P4"].append(np.argmax(combined_probs))
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(val_dataset.imgs)}")
            
    print("\n" + "="*100)
    print("🎯 F1-SCORE SUMMARY (Macro Avg & Per-Class)")
    print("="*100)
    
    header = f"{'Model':15s} | {'Macro F1':>8s} | {'Accuracy':>8s} | " + " | ".join([f"{c[:8]:>8s}" for c in class_names])
    print(header)
    print("-" * len(header))
    
    for name in ["YOLO_Orig", "EffNet_Orig", "YOLO_P1", "EffNet_P1", "YOLO_P3", "EffNet_P3", "Ensemble_P4"]:
        f1_macro = f1_score(results_true, results_preds[name], average='macro')
        acc = np.mean(np.array(results_true) == np.array(results_preds[name]))
        f1_classes = f1_score(results_true, results_preds[name], average=None, labels=range(num_classes))
        
        class_scores = " | ".join([f"{score:>8.4f}" for score in f1_classes])
        print(f"{name:15s} | {f1_macro:>8.4f} | {acc:>8.4f} | {class_scores}")
        
        # Plot and save confusion matrix
        cm = confusion_matrix(results_true, results_preds[name], labels=range(num_classes))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(BASE_DIR / f'confusion_matrix_{name}.png')
        plt.close()
        
    print("\nDone!")

if __name__ == "__main__":
    evaluate_models()
