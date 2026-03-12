# Priority 1: Class-Weighted Loss Function

## Problem
Both YOLOv8 and EfficientNet-B0 score **0% F1** on `Midpiece_Anomaly` and `Tail_Anomaly`.
Despite the training set being balanced (342 samples/class via oversampling), the original
images for these classes are very few (~5–6), so the oversampled copies are near-duplicates.

## Solution
Apply **inverse-frequency class weights** to the loss function so the model pays heavier
penalties for misclassifying rare classes.

| Class | Val Samples | Weight |
|---|---|---|
| Combined_Anomaly | 87 | 1.0 |
| Head_Anomaly | 84 | 1.0 |
| Midpiece_Anomaly | 6 | **14.5** |
| Normal | 25 | **3.5** |
| Tail_Anomaly | 5 | **17.4** |

## Additional Improvements in These Scripts
- **Stronger augmentation**: vertical flip, affine transforms, color jitter, gaussian blur
- **Early stopping** with patience=10 (EfficientNet) / 15 (YOLOv8)
- **Layer freezing**: First 6 EfficientNet feature blocks frozen to prevent overfitting
- **Increased dropout**: 0.4 (EfficientNet) / 0.3 (YOLOv8)
- **ReduceLROnPlateau**: Halves LR after 5 epochs without improvement
- **Per-class accuracy tracking**: See exactly how each class performs per epoch
- **Focal loss** (YOLOv8): gamma=2.0 to focus on hard examples

## Scripts

| File | Description |
|---|---|
| `train_efficientnet_weighted.py` | EfficientNet-B0 with weighted CrossEntropyLoss |
| `train_yolo_weighted.py` | YOLOv8s-cls with focal loss (gamma=2.0) |
| `evaluate_weighted.py` | Compare original vs weighted models |

## Usage

```bash
# Train EfficientNet (weighted)
python priority_1_class_weights/train_efficientnet_weighted.py

# Train YOLOv8 (weighted)
python priority_1_class_weights/train_yolo_weighted.py

# Evaluate & compare all models
python priority_1_class_weights/evaluate_weighted.py
```
