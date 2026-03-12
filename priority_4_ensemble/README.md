# Priority 4: Model Ensemble

## Concept
Instead of picking one model, **combine both models' probability distributions** to produce a more robust prediction. This exploits complementary biases:

- **YOLOv8** tends to over-predict `Normal` → biased toward false negatives
- **EfficientNet** tends to over-predict `Combined_Anomaly` → biased toward false anomalies
- **Ensemble** averages them out → more balanced predictions

## How It Works

```
YOLOv8 probs:      [0.15, 0.35, 0.05, 0.40, 0.05]  → "Normal" (40%)
EfficientNet probs: [0.60, 0.20, 0.05, 0.10, 0.05]  → "Combined_Anomaly" (60%)

Ensemble (40/60):   [0.42, 0.26, 0.05, 0.22, 0.05]  → "Combined_Anomaly" (42%)
```

The ensemble resolves disagreements by weighing each model's **full probability distribution**, not just the top prediction.

## Ensemble Weights
| Model | Weight | Rationale |
|---|---|---|
| YOLOv8 | 40% | Lower weight because it over-predicts Normal |
| EfficientNet | 60% | Higher weight because it better detects anomalies |

Adjustable via CLI: `--yolo-weight 0.5 --effnet-weight 0.5`

## Usage

```bash
# Run ensemble on a video (uses Priority 1 weighted models by default)
python priority_4_ensemble/ensemble_classifier.py --video path/to/video.mp4

# More frames
python priority_4_ensemble/ensemble_classifier.py --video path/to/video.mp4 --frames 5

# Custom weights (equal weighting)
python priority_4_ensemble/ensemble_classifier.py --video path/to/video.mp4 --yolo-weight 0.5 --effnet-weight 0.5
```

## Output
Generates `ensemble_comparison_report.docx` containing:
- Class distribution: individual models vs ensemble
- Per-crop comparison table
- **Tiebreaker analysis**: where models disagreed and how the ensemble resolved it
- Agreement statistics
