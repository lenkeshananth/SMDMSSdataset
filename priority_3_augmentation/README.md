# Priority 3: Stronger Data Augmentation

## Problem
The "balanced" training dataset has 342 images per class, but Midpiece and Tail classes
are just duplicated copies of very few originals:

| Class | Total Files | Unique (by content) | Duplication Factor |
|---|---|---|---|
| Combined_Anomaly | 342 | ~341 | 1x (almost all unique) |
| Head_Anomaly | 342 | ~330 | 1x |
| Normal | 342 | ~92 | ~4x |
| **Midpiece_Anomaly** | 342 | **~20** | **17x** |
| **Tail_Anomaly** | 342 | **~14** | **24x** |

The model memorizes duplicates instead of learning generalizable features.

## Solution
1. **Deduplicate** — find truly unique images using MD5 file hashing
2. **Aggressively augment rare classes** — generate genuinely diverse variations:
   - Midpiece: 20 augmented copies per original → ~420 total
   - Tail: 25 augmented copies per original → ~364 total
   - Normal: 4 copies per original → ~460 total
   - Combined/Head: keep originals only (~340 each)
3. **Trim to ~400/class** for balance

### Augmentation Techniques Applied
- Random rotation (±45°)
- Horizontal & vertical flips
- Random scale/crop (70-100%)
- Brightness, contrast, sharpness, color jitter
- Gaussian blur
- Noise injection (Gaussian noise σ=3-15)
- Perspective distortion

## Scripts

| File | Purpose |
|---|---|
| `create_augmented_dataset.py` | Generate the augmented dataset |
| `train_efficientnet_augmented.py` | EfficientNet-B0 on augmented data |
| `train_yolo_augmented.py` | YOLOv8s-cls on augmented data |
| `compare_pipeline_augmented.py` | Compare all model versions on video |

## Usage

```bash
# Step 1: Generate augmented dataset
python priority_3_augmentation/create_augmented_dataset.py

# Step 2: Train models
python priority_3_augmentation/train_efficientnet_augmented.py
python priority_3_augmentation/train_yolo_augmented.py

# Step 3: Compare all models
python priority_3_augmentation/compare_pipeline_augmented.py --video path/to/video.mp4
```
