# Sperm Morphology Detection & Classification Pipeline

This repository contains an end-to-end pipeline for analyzing sperm morphology in wet-mount video microscopy. The pipeline performs **detection** (locating sperm in video frames) followed by **multi-class classification** (Normal vs. 4 types of anomalies).

We have systematically improved the initial classification models through a series of "Priority" improvements to counter class imbalance and dataset duplication.

---

## 📁 Repository Structure

### 1. Datasets
*   **`sperm.v1i.yolov8/`**: The original detection dataset (`train/val/test` images mapped with bound boxes). Used to train the initial YOLOv8 detection model.
*   **`classification_dataset_balanced/`**: The original classification dataset. **Issue:** It appeared "balanced" (342 images per class), but our script discovered the rare classes (`Tail_Anomaly`, `Midpiece_Anomaly`) were simply the same 14-20 images duplicated dozens of times.
*   **`priority_3_augmented_dataset/`**: The first augmented dataset (~400 unique variations per class) to fix the duplication issue.
*   **`priority_3_large_augmented_dataset/`**: The massive augmented dataset (~1,500 highly diverse images per class) containing advanced augmentations like CLAHE, Cutout, and Perspective transformations.

### 2. Core Pipeline
*   **`sperm_pipeline/`**
    *   `pipeline.py`: The core `SpermAnalysisPipeline` class. It manages loading both a YOLO detection model and a Classification model (YOLO or EfficientNet), extracting crops from video frames, running inference, and mapping back to bounding boxes.
*   **`compare_pipeline.py` & `evaluate_models.py`**: Base scripts for running side-by-side comparisons of the original models on full videos or the validation dataset.

### 3. Model Improvements (The Priority Folders)

#### 🔸 `efficientnet_training/` (Original Baseline)
*   The first attempt at training an EfficientNet-B0 model. Suffered from the dataset duplication issues, resulting in poor real-world generalizability.

#### 🔸 `priority_1_class_weights/`
*   **Goal:** Address class imbalance mathematically.
*   **What it does:** Modifies the loss function (CrossEntropyLoss in Torch, class weights in YOLO) to heavily penalize misses on the minority classes (Midpiece, Tail).
*   **Result:** Improved detection of rare classes, but still fundamentally limited by the lack of unique training images.

#### 🔸 `priority_3_augmentation/`
*   **Goal:** Generate genuinely un-duplicated, massive training data.
*   **What it does:** 
    *   `create_large_augmented_dataset.py`: Strips duplicates via MD5 hash, then generates 1,500 unique images per class using 12 heavy microscopic-focused augmentation techniques.
    *   `train_efficientnet_augmented.py`: Trains EfficientNet-B0 on the massive dataset using AdamW + Cosine Annealing.
    *   `train_yolo_augmented.py`: Trains YOLOv8s-cls on the new data.
    *   `compare_pipeline_augmented.py`: Runs a side-by-side comparison on video.
*   **Result:** **EffNet P3** is our best performing model, achieving an 85% anomaly detection rate (closest to the 98% clinical ground truth) and the lowest false-Normal rate (15%).

#### 🔸 `priority_4_ensemble/`
*   **Goal:** Combine model biases for better overall accuracy.
*   **What it does:** Extracts the raw probability vectors from both YOLOv8 and EfficientNet for the *same sperm crop*. It computes a weighted average (40% YOLO + 60% EffNet) to output a final `Ensemble_P4` prediction.

### 4. Analysis & Evaluation
*   **`full_video_analysis.py`**: The definitive evaluation script.
    *   It loops through a clinical video (e.g., ID 30).
    *   Extracts sperm using YOLOv8 detection.
    *   Classifies each sperm instance using **all 7 model variations** simultaneously (YOLO Orig, YOLO P1, YOLO P3, EffNet Orig, EffNet P1, EffNet P3, and Ensemble P4).
    *   Calculates the **Mean Absolute Error (MAE)** between each model's predicted class distribution and the actual clinical ground-truth data (from `semen_analysis_data.csv`).
    *   Generates a detailed `.docx` report.

---

## 🚀 How to Run the Best Analysis

To run the ultimate full-video evaluation that compares all 7 models against clinical data and calculates MAE, use:

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the analysis (evaluates EVERY frame by default)
python full_video_analysis.py --video D:\Krishna\SMDMSSdataset\sperm.v1i.yolov8\testing\30_09.03.25_SSW.avi

# For a quicker test (samples every 50th frame)
python full_video_analysis.py --video D:\Krishna\SMDMSSdataset\sperm.v1i.yolov8\testing\30_09.03.25_SSW.avi --every 50
```

The script will produce terminal output with MAE rankings and a Microsoft Word document (`full_video_analysis_report.docx`) containing the final distributions.
