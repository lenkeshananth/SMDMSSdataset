"""
Two-Stage Sperm Analysis Pipeline
==================================
Stage 1: Sperm Detection (YOLOv8 Object Detection)
Stage 2: Morphological Defect Classification (YOLOv8 Classification)

This pipeline:
1. Takes a video or image as input
2. Detects sperm cells using object detection model
3. Crops each detected sperm from the frame
4. Classifies each cropped sperm for morphological defects
5. Outputs combined results with location + defect type

Usage:
    python pipeline.py --video path/to/video.mp4
    python pipeline.py --image path/to/image.jpg
    python pipeline.py --folder path/to/images/
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
import os

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Base paths
BASE_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
PIPELINE_DIR = BASE_DIR / "sperm_pipeline"

# Model paths (update these after training)
DETECTION_MODEL = BASE_DIR / "sperm.v1i.yolov8" / "runs" / "detect" / "sperm_detection" / "weights" / "best.pt"
CLASSIFICATION_MODEL = BASE_DIR / "runs" / "smdss_sperm_cls" / "weights" / "best.pt"

# Detection settings
DETECTION_CONF = 0.25          # Confidence threshold for detection
DETECTION_IOU = 0.45           # IoU threshold for NMS
DETECTION_IMG_SIZE = 640       # Detection image size

# Classification settings
CLASSIFICATION_IMG_SIZE = 224  # Classification image size
CLASSIFICATION_CONF = 0.5      # Minimum confidence for classification

# Morphological defect classes
DEFECT_CLASSES = {
    0: "Normal",
    1: "Head_Anomaly",
    2: "Midpiece_Anomaly", 
    3: "Tail_Anomaly",
    4: "Combined_Anomaly"
}

# Output settings
OUTPUT_DIR = PIPELINE_DIR / "results"
SAVE_CROPS = True              # Save cropped sperm images
SAVE_ANNOTATED = True          # Save annotated frames
SAVE_JSON = True               # Save JSON results

# Video processing
FRAME_SKIP = 1                 # Process every Nth frame (1 = all frames)
MAX_FRAMES = None              # Maximum frames to process (None = all)

# Device
DEVICE = "0"                   # GPU device ("0" for GPU, "cpu" for CPU)

# ══════════════════════════════════════════════════════════════════════════════


class SpermAnalysisPipeline:
    """
    Two-stage pipeline for sperm detection and morphological classification.
    """
    
    def __init__(self, detection_model=None, classification_model=None):
        """
        Initialize the pipeline with detection and classification models.
        """
        self.detection_model_path = detection_model or DETECTION_MODEL
        self.classification_model_path = classification_model or CLASSIFICATION_MODEL
        
        self.detection_model = None
        self.classification_model = None
        
        # Results storage
        self.results = []
        
    def load_models(self):
        """Load both detection and classification models."""
        print("=" * 70)
        print("       LOADING SPERM ANALYSIS PIPELINE MODELS")
        print("=" * 70)
        
        # Load detection model
        print(f"\n📦 Loading Detection Model...")
        if Path(self.detection_model_path).exists():
            self.detection_model = YOLO(str(self.detection_model_path))
            print(f"   ✓ Loaded: {self.detection_model_path}")
        else:
            print(f"   ❌ Not found: {self.detection_model_path}")
            print("   Please train the detection model first!")
            return False
        
        # Load classification model
        print(f"\n📦 Loading Classification Model...")
        if Path(self.classification_model_path).exists():
            self.classification_model = YOLO(str(self.classification_model_path))
            print(f"   ✓ Loaded: {self.classification_model_path}")
        else:
            print(f"   ⚠️ Not found: {self.classification_model_path}")
            print("   Classification will be skipped.")
            self.classification_model = None
        
        print("\n" + "=" * 70)
        return True
    
    def detect_sperm(self, frame):
        """
        Stage 1: Detect sperm cells in a frame.
        
        Args:
            frame: Input image (numpy array)
            
        Returns:
            List of detections with bounding boxes
        """
        results = self.detection_model.predict(
            source=frame,
            conf=DETECTION_CONF,
            iou=DETECTION_IOU,
            imgsz=DETECTION_IMG_SIZE,
            device=DEVICE,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.detection_model.names[cls]
                })
        
        return detections
    
    def crop_detection(self, frame, bbox, padding=10):
        """
        Crop a detected region from the frame with optional padding.
        
        Args:
            frame: Input image
            bbox: [x1, y1, x2, y2] bounding box
            padding: Pixels to add around the bbox
            
        Returns:
            Cropped image
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return frame[y1:y2, x1:x2]
    
    def classify_morphology(self, crop):
        """
        Stage 2: Classify morphological defects in a cropped sperm image.
        
        Args:
            crop: Cropped sperm image
            
        Returns:
            Classification result dict
        """
        if self.classification_model is None:
            return {'class_id': -1, 'class_name': 'Unknown', 'confidence': 0.0}
        
        # Resize crop for classification
        crop_resized = cv2.resize(crop, (CLASSIFICATION_IMG_SIZE, CLASSIFICATION_IMG_SIZE))
        
        results = self.classification_model.predict(
            source=crop_resized,
            imgsz=CLASSIFICATION_IMG_SIZE,
            device=DEVICE,
            verbose=False
        )
        
        for result in results:
            probs = result.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            
            return {
                'class_id': top1_idx,
                'class_name': self.classification_model.names[top1_idx],
                'confidence': top1_conf,
                'all_probs': {self.classification_model.names[i]: float(probs.data[i]) 
                             for i in range(len(probs.data))}
            }
        
        return {'class_id': -1, 'class_name': 'Unknown', 'confidence': 0.0}
    
    def process_frame(self, frame, frame_id=0, save_crops=False, output_dir=None):
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame: Input image
            frame_id: Frame identifier
            save_crops: Save cropped regions
            output_dir: Directory for saving crops
            
        Returns:
            List of analysis results, annotated frame
        """
        frame_results = []
        annotated_frame = frame.copy()
        
        # Stage 1: Detect sperm
        detections = self.detect_sperm(frame)
        
        for i, det in enumerate(detections):
            bbox = det['bbox']
            
            # Crop detection
            crop = self.crop_detection(frame, bbox)
            
            if crop.size == 0:
                continue
            
            # Stage 2: Classify morphology
            morphology = self.classify_morphology(crop)
            
            # Combine results
            sperm_result = {
                'sperm_id': i + 1,
                'frame_id': frame_id,
                'detection': det,
                'morphology': morphology
            }
            frame_results.append(sperm_result)
            
            # Save crop if requested
            if save_crops and output_dir:
                crop_path = output_dir / "crops" / f"frame{frame_id}_sperm{i+1}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_path), crop)
            
            # Annotate frame
            x1, y1, x2, y2 = bbox
            
            # Color based on morphology
            if morphology['class_name'] == 'Normal':
                color = (0, 255, 0)  # Green
            elif morphology['class_name'] == 'Unknown':
                color = (128, 128, 128)  # Gray
            else:
                color = (0, 0, 255)  # Red for anomalies
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{morphology['class_name']} ({morphology['confidence']:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_results, annotated_frame
    
    def process_video(self, video_path, output_dir=None):
        """
        Process a video file through the pipeline.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for results
        """
        video_path = Path(video_path)
        if output_dir is None:
            output_dir = OUTPUT_DIR / video_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        
        # Setup video writer for annotated output
        if SAVE_ANNOTATED:
            output_video_path = output_dir / f"{video_path.stem}_analyzed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        all_results = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if needed
            if frame_count % FRAME_SKIP != 0:
                if SAVE_ANNOTATED:
                    out_video.write(frame)
                continue
            
            if MAX_FRAMES and processed_count >= MAX_FRAMES:
                break
            
            # Process frame
            frame_results, annotated_frame = self.process_frame(
                frame, 
                frame_id=frame_count,
                save_crops=SAVE_CROPS,
                output_dir=output_dir
            )
            
            all_results.extend(frame_results)
            processed_count += 1
            
            # Write annotated frame
            if SAVE_ANNOTATED:
                out_video.write(annotated_frame)
            
            # Progress
            if processed_count % 10 == 0:
                print(f"   Processed: {processed_count} frames, Found: {len(all_results)} sperm")
        
        cap.release()
        if SAVE_ANNOTATED:
            out_video.release()
            print(f"\n📹 Annotated video saved: {output_video_path}")
        
        # Save JSON results
        if SAVE_JSON:
            self._save_results(all_results, output_dir / "results.json")
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def process_image(self, image_path, output_dir=None):
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to image file
            output_dir: Output directory for results
        """
        image_path = Path(image_path)
        if output_dir is None:
            output_dir = OUTPUT_DIR / image_path.stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🖼️ Processing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"❌ Cannot load image: {image_path}")
            return
        
        # Process frame
        frame_results, annotated_frame = self.process_frame(
            frame,
            frame_id=0,
            save_crops=SAVE_CROPS,
            output_dir=output_dir
        )
        
        # Save annotated image
        if SAVE_ANNOTATED:
            output_path = output_dir / f"{image_path.stem}_analyzed.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"📷 Annotated image saved: {output_path}")
        
        # Save JSON results
        if SAVE_JSON:
            self._save_results(frame_results, output_dir / "results.json")
        
        # Print summary
        self._print_summary(frame_results)
        
        return frame_results
    
    def process_folder(self, folder_path, output_dir=None):
        """
        Process all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_dir: Output directory for results
        """
        folder_path = Path(folder_path)
        if output_dir is None:
            output_dir = OUTPUT_DIR / folder_path.name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Processing folder: {folder_path}")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = [f for f in folder_path.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        print(f"   Found {len(images)} images")
        
        all_results = []
        for i, image_path in enumerate(images):
            print(f"\n   [{i+1}/{len(images)}] {image_path.name}")
            
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            
            frame_results, annotated_frame = self.process_frame(
                frame,
                frame_id=i,
                save_crops=SAVE_CROPS,
                output_dir=output_dir
            )
            
            # Add source info
            for r in frame_results:
                r['source_image'] = image_path.name
            
            all_results.extend(frame_results)
            
            # Save annotated image
            if SAVE_ANNOTATED:
                annotated_dir = output_dir / "annotated"
                annotated_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(annotated_dir / image_path.name), annotated_frame)
        
        # Save combined results
        if SAVE_JSON:
            self._save_results(all_results, output_dir / "all_results.json")
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results, output_path):
        """Save results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_sperm': len(results),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"📄 Results saved: {output_path}")
    
    def _print_summary(self, results):
        """Print analysis summary."""
        print("\n" + "=" * 70)
        print("                    ANALYSIS SUMMARY")
        print("=" * 70)
        
        total = len(results)
        print(f"\n📊 Total sperm detected: {total}")
        
        if total == 0:
            return
        
        # Count by morphology
        morphology_counts = {}
        for r in results:
            morph = r['morphology']['class_name']
            morphology_counts[morph] = morphology_counts.get(morph, 0) + 1
        
        print("\n📈 Morphological Distribution:")
        for cls, count in sorted(morphology_counts.items()):
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"   {cls:20} {count:4} ({pct:5.1f}%) {bar}")
        
        # Normal vs Abnormal
        normal = morphology_counts.get('Normal', 0)
        abnormal = total - normal
        print(f"\n✅ Normal: {normal} ({normal/total*100:.1f}%)")
        print(f"❌ Abnormal: {abnormal} ({abnormal/total*100:.1f}%)")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-Stage Sperm Analysis Pipeline")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    parser.add_argument("--detection-model", type=str, help="Path to detection model")
    parser.add_argument("--classification-model", type=str, help="Path to classification model")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpermAnalysisPipeline(
        detection_model=args.detection_model,
        classification_model=args.classification_model
    )
    
    # Load models
    if not pipeline.load_models():
        return
    
    # Process input
    output_dir = Path(args.output) if args.output else None
    
    if args.video:
        pipeline.process_video(args.video, output_dir)
    elif args.image:
        pipeline.process_image(args.image, output_dir)
    elif args.folder:
        pipeline.process_folder(args.folder, output_dir)
    else:
        print("\nUsage:")
        print("  python pipeline.py --video path/to/video.mp4")
        print("  python pipeline.py --image path/to/image.jpg")
        print("  python pipeline.py --folder path/to/images/")


if __name__ == "__main__":
    main()
