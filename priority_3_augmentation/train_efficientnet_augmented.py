"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 3: EfficientNet-B0 Training on LARGE Augmented Dataset
  ──────────────────────────────────────────────────────────────────
  Trains on the priority_3_large_augmented_dataset (~1500 images/class)
  augmented images (not just oversampled duplicates).

  Combines:
    - Priority 1: Class-weighted loss
    - Priority 3: Genuinely augmented dataset + online augmentation

  Usage:
      python priority_3_augmentation/train_efficientnet_augmented.py
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
TRAIN_DIR = BASE_DIR / "priority_3_large_augmented_dataset" / "train"
VAL_DIR = BASE_DIR / "priority_3_large_augmented_dataset" / "val"
OUTPUT_DIR = BASE_DIR / "priority_3_augmentation" / "weights"

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.0003     # Lower LR for large augmented data
IMG_SIZE = 224
PATIENCE = 12

# Class weights (mild boost — large augmented dataset already balanced at ~1500/class)
CLASS_WEIGHTS = torch.tensor([
    1.0,    # Combined_Anomaly
    1.0,    # Head_Anomaly
    2.0,    # Midpiece_Anomaly  (slight boost for the rarest original class)
    1.5,    # Normal
    2.5,    # Tail_Anomaly      (slight boost for the rarest original class)
])


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Online augmentation (applied on top of the offline augmented dataset) ──
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(
            degrees=0, translate=(0.08, 0.08),
            scale=(0.9, 1.1), shear=5
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout-style aug
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

    # ── Model ──
    print(f"\nLoading EfficientNet-B0 (pretrained)...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze fewer layers — more augmented data means less overfitting risk
    for param in model.features[:5].parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    # ── Loss + optimizer ──
    weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"\n✅ Class weights: {dict(zip(class_names, CLASS_WEIGHTS.tolist()))}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-3
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # ── Training loop ──
    print("\nStarting Training...\n")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    torch.cuda.empty_cache()

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
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
                inputs, labels = inputs.to(device), labels.to(device)
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

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'  {phase.capitalize():6s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val':
                for i, name in enumerate(class_names):
                    if class_total[i] > 0:
                        cls_acc = class_correct[i] / class_total[i]
                        print(f'    {name:25s} {class_correct[i]:3d}/{class_total[i]:3d} ({cls_acc:.1%})')

                scheduler.step(epoch)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), OUTPUT_DIR / 'efficientnet_b0_augmented.pt')
                    epochs_no_improve = 0
                    print(f'  ✅ New best! Saved.')
                else:
                    epochs_no_improve += 1
                    print(f'  ⏳ No improve {epochs_no_improve}/{PATIENCE}')

                if epochs_no_improve >= PATIENCE:
                    print(f'\n⛔ Early stopping at epoch {epoch+1}.')
                    break

        if epochs_no_improve >= PATIENCE:
            break

    print(f'\n{"═" * 50}')
    print(f'Best val Acc: {best_acc:.4f}')
    print(f'Weights: {OUTPUT_DIR / "efficientnet_b0_augmented.pt"}')
    print(f'{"═" * 50}')


if __name__ == '__main__':
    train()
