# Model Evaluation Report

This document compares the classification performance of EfficientNet-B0 and the balanced YOLOv8 model on the validation dataset.

## 1. YOLOv8 (Balanced)
**Overall Accuracy:** 48.02%

```text
                  precision    recall  f1-score   support

Combined_Anomaly      0.565     0.558     0.561        86
    Head_Anomaly      0.518     0.518     0.518        83
Midpiece_Anomaly      0.000     0.000     0.000         5
          Normal      0.316     0.250     0.279        24
    Tail_Anomaly      0.000     0.000     0.000         4

        accuracy                          0.480       202
       macro avg      0.280     0.265     0.272       202
    weighted avg      0.491     0.480     0.485       202

```

## 2. EfficientNet-B0
**Overall Accuracy:** 54.46%

```text
                  precision    recall  f1-score   support

Combined_Anomaly      0.573     0.686     0.624        86
    Head_Anomaly      0.539     0.578     0.558        83
Midpiece_Anomaly      0.000     0.000     0.000         5
          Normal      0.333     0.125     0.182        24
    Tail_Anomaly      0.000     0.000     0.000         4

        accuracy                          0.545       202
       macro avg      0.289     0.278     0.273       202
    weighted avg      0.505     0.545     0.517       202

```
