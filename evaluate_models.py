import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from ultralytics import YOLO
from pathlib import Path
import os
from sklearn.metrics import classification_report, accuracy_score

# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DATA_DIR = BASE_DIR / "classification_dataset_balanced" / "val"

# Model paths
EFFICIENTNET_PATH = BASE_DIR / "efficientnet_training" / "weights" / "efficientnet_b0_best.pt"
YOLO_BALANCED_PATH = BASE_DIR / "runs" / "classify" / "runs" / "sperm_cls_balanced" / "weights" / "best.pt"

IMG_SIZE = 224
BATCH_SIZE = 32

def evaluate_efficientnet(device, num_classes, class_names):
    print("\n--- Evaluating EfficientNet-B0 ---")
    
    if not EFFICIENTNET_PATH.exists():
        print("❌ EfficientNet weights not found. (Is it still training?)")
        return None, None
        
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
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
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return all_labels, all_preds

def evaluate_yolo(class_names):
    print("\n--- Evaluating YOLOv8 Balanced ---")
    
    if not YOLO_BALANCED_PATH.exists():
        print("❌ YOLO weights not found.")
        return None, None
        
    model = YOLO(str(YOLO_BALANCED_PATH))
    
    all_preds = []
    all_labels = []
    
    # We run it on the specifically named 'val' split using ultralytics natively 
    # since it handles the dataloading and metrics perfectly.
    val_dataset = datasets.ImageFolder(str(DATA_DIR))
    
    print(f"Validating over {len(val_dataset.imgs)} images...")
    
    for i, (img_path, label_idx) in enumerate(val_dataset.imgs):
        results = model.predict(img_path, imgsz=IMG_SIZE, verbose=False)
        pred_idx = results[0].probs.top1
        
        all_preds.append(pred_idx)
        all_labels.append(label_idx)
        
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(val_dataset.imgs)}")
            
    return all_labels, all_preds

def generate_report():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get class names from the validation folder structure
    val_dataset_dummy = datasets.ImageFolder(str(DATA_DIR))
    class_names = val_dataset_dummy.classes
    num_classes = len(class_names)
    
    print(f"Classes found: {class_names}")
    
    report_content = "# Model Evaluation Report\n\n"
    report_content += "This document compares the classification performance of EfficientNet-B0 and the balanced YOLOv8 model on the validation dataset.\n\n"
    
    # ── Evaluate YOLOv8 ──
    y_true_yolo, y_pred_yolo = evaluate_yolo(class_names)
    
    if y_true_yolo:
        acc_yolo = accuracy_score(y_true_yolo, y_pred_yolo)
        report_yolo = classification_report(y_true_yolo, y_pred_yolo, target_names=class_names, digits=3)
        
        report_content += f"## 1. YOLOv8 (Balanced)\n"
        report_content += f"**Overall Accuracy:** {acc_yolo:.2%}\n\n"
        report_content += "```text\n"
        report_content += report_yolo
        report_content += "\n```\n\n"
    
    # ── Evaluate EfficientNet ──
    try:
        y_true_eff, y_pred_eff = evaluate_efficientnet(device, num_classes, class_names)
        
        if y_true_eff:
            acc_eff = accuracy_score(y_true_eff, y_pred_eff)
            report_eff = classification_report(y_true_eff, y_pred_eff, target_names=class_names, digits=3)
            
            report_content += f"## 2. EfficientNet-B0\n"
            report_content += f"**Overall Accuracy:** {acc_eff:.2%}\n\n"
            report_content += "```text\n"
            report_content += report_eff
            report_content += "\n```\n"
    except Exception as e:
        report_content += f"## 2. EfficientNet-B0\n"
        report_content += "_Note: EfficientNet evaluation failed or may still be training in the terminal._\n"
        report_content += f"Error: `{e}`\n"
        
    # Write to Markdown
    output_path = BASE_DIR / "model_metrics_comparison.md"
    with open(output_path, "w") as f:
        f.write(report_content)
        
    print(f"\n✅ Report generated successfully and saved to: {output_path}")

if __name__ == "__main__":
    generate_report()
