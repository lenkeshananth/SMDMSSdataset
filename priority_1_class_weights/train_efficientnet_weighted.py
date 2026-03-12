"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 1: Class-Weighted Loss Function
  ─────────────────────────────────────────
  Problem:  Both YOLOv8 and EfficientNet score 0% F1 on Midpiece_Anomaly
            and Tail_Anomaly despite balanced training data. The training
            set was oversampled to 342/class, but the originals for Midpiece
            and Tail are very few (~5-6 unique images). The model memorizes
            duplicates instead of learning generalizable features.

  Solution: Apply class weights inversely proportional to the ORIGINAL
            (unbalanced) class distribution, plus stronger augmentation
            specifically for rare classes.

  Usage:
      python priority_1_class_weights/train_efficientnet_weighted.py
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import time
import copy

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
TRAIN_DIR = BASE_DIR / "classification_dataset_balanced" / "train"
VAL_DIR = BASE_DIR / "classification_dataset_balanced" / "val"
OUTPUT_DIR = BASE_DIR / "priority_1_class_weights" / "weights"

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224
PATIENCE = 10          # Early stopping patience

# ═════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ═════════════════════════════════════════════════════════════════════════════
# These weights are based on the VALIDATION (natural) distribution, which
# reflects how rare each class truly is in the real world.
#
# Val distribution:
#   Combined_Anomaly: 87  (42%)
#   Head_Anomaly:     84  (41%)
#   Normal:           25  (12%)
#   Midpiece_Anomaly:  6  ( 3%)
#   Tail_Anomaly:      5  ( 2%)
#
# Weight = max_count / class_count (inverse frequency)
# Then normalized so the mean weight ≈ 1.0
#
# Classes are alphabetically ordered by ImageFolder:
#   [0] Combined_Anomaly, [1] Head_Anomaly, [2] Midpiece_Anomaly,
#   [3] Normal, [4] Tail_Anomaly

CLASS_WEIGHTS = torch.tensor([
    1.0,    # Combined_Anomaly  (87 val samples — well represented)
    1.0,    # Head_Anomaly      (84 val samples — well represented)
    14.5,   # Midpiece_Anomaly  (6 val samples — 87/6 ≈ 14.5x boost)
    3.5,    # Normal            (25 val samples — 87/25 ≈ 3.5x boost)
    17.4,   # Tail_Anomaly      (5 val samples — 87/5 ≈ 17.4x boost)
])


def train_efficientnet_weighted():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Data Loading with Stronger Augmentation ──
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            shear=10
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    # ── 2. Initialize EfficientNet-B0 ──
    print(f"\nLoading EfficientNet-B0 (pretrained)...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze early layers to prevent overfitting on small dataset
    for param in model.features[:6].parameters():
        param.requires_grad = False

    # Replace classification head
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),   # Increased dropout
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    # ── 3. Weighted Loss Function ──
    weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"\n✅ Class weights applied:")
    for i, name in enumerate(class_names):
        print(f"   {name:25s} → weight = {CLASS_WEIGHTS[i]:.1f}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4    # L2 regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ── 4. Training Loop with Early Stopping ──
    print("\nStarting Training...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            class_correct = [0] * num_classes
            class_total = [0] * num_classes

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Per-class accuracy tracking
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize():6s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            # Print per-class accuracy during validation
            if phase == 'val':
                print(f'  Per-class accuracy:')
                for i, name in enumerate(class_names):
                    if class_total[i] > 0:
                        cls_acc = class_correct[i] / class_total[i]
                        print(f'    {name:25s} {class_correct[i]:3d}/{class_total[i]:3d} ({cls_acc:.1%})')

                # Learning rate scheduler
                scheduler.step(epoch_loss)

                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), OUTPUT_DIR / 'efficientnet_b0_weighted.pt')
                    epochs_no_improve = 0
                    print(f'  ✅ New best! Saved weights.')
                else:
                    epochs_no_improve += 1
                    print(f'  ⏳ No improvement for {epochs_no_improve}/{PATIENCE} epochs')

                if epochs_no_improve >= PATIENCE:
                    print(f'\n⛔ Early stopping triggered after {epoch+1} epochs.')
                    break

        if epochs_no_improve >= PATIENCE:
            break

    print(f'\n{"═" * 50}')
    print(f'Training complete.')
    print(f'Best val Acc: {best_acc:.4f}')
    print(f'Weights saved: {OUTPUT_DIR / "efficientnet_b0_weighted.pt"}')
    print(f'{"═" * 50}')


if __name__ == '__main__':
    train_efficientnet_weighted()
