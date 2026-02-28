# Sperm Analysis Pipeline

Two-stage cascade pipeline for sperm detection and morphological defect classification.

## Architecture

```
Video/Image Input
       │
       ▼
┌─────────────────────────────────┐
│  Stage 1: Sperm Detection       │
│  (YOLOv8 Object Detection)      │
│  Output: Bounding boxes         │
└───────────────┬─────────────────┘
                │
                ▼
       Crop Bounding Boxes
                │
                ▼
┌─────────────────────────────────┐
│  Stage 2: Morphology            │
│  Classification                 │
│  (YOLOv8 Classification)        │
│  Output: Defect category        │
└───────────────┬─────────────────┘
                │
                ▼
         Combined Results
   (Location + Defect Type)
```

## Usage

### Process a video:
```powershell
cd d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset\sperm_pipeline
python pipeline.py --video path/to/video.mp4
```

### Process a single image:
```powershell
python pipeline.py --image path/to/image.jpg
```

### Process a folder of images:
```powershell
python pipeline.py --folder path/to/images/
```

### With custom models:
```powershell
python pipeline.py --video video.mp4 --detection-model path/to/detect.pt --classification-model path/to/classify.pt
```

## Output

Results are saved to `sperm_pipeline/results/`:

```
results/
├── video_name/
│   ├── video_name_analyzed.mp4    # Annotated video
│   ├── results.json               # JSON results
│   └── crops/                     # Cropped sperm images
│       ├── frame1_sperm1.jpg
│       ├── frame1_sperm2.jpg
│       └── ...
```

## Morphology Classes

| Class | Description |
|-------|-------------|
| Normal | No morphological defects |
| Head_Anomaly | Defects in sperm head |
| Midpiece_Anomaly | Defects in midpiece |
| Tail_Anomaly | Defects in tail |
| Combined_Anomaly | Multiple defect regions |

## Configuration

Edit `pipeline.py` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTION_CONF` | 0.25 | Detection confidence threshold |
| `CLASSIFICATION_CONF` | 0.5 | Classification confidence |
| `FRAME_SKIP` | 1 | Process every Nth frame |
| `SAVE_CROPS` | True | Save cropped sperm images |
| `DEVICE` | "0" | GPU device (or "cpu") |

## Requirements

```
ultralytics
opencv-python
numpy
```

## Training the Models

Before using the pipeline, train both models:

1. **Detection Model:**
   ```powershell
   cd d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset
   python train_yolov8_detect.py
   ```

2. **Classification Model:**
   ```powershell
   python train_yolov8_cls.py
   ```
