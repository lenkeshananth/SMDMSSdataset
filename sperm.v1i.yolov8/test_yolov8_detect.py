"""
YOLOv8 Object Detection Testing Script for Sperm Dataset
=========================================================
This script runs inference on images using a trained YOLOv8 model.

Usage:
    # Test single image
    python test_yolov8_detect.py --image path/to/image.jpg
    
    # Test folder of images
    python test_yolov8_detect.py --image path/to/folder/
    
    # Test with custom model
    python test_yolov8_detect.py --image path/to/image.jpg --model path/to/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2

# ══════════════════════════════════════════════════════════════════════════════
#                              CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Paths
DATASET_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset\sperm.v1i.yolov8")
DEFAULT_MODEL = DATASET_DIR / "runs" / "detect" / "sperm_detection" / "weights" / "best.pt"

# Inference settings
CONF_THRESHOLD = 0.25      # Confidence threshold (0-1)
IOU_THRESHOLD = 0.45       # IoU threshold for NMS
IMG_SIZE = 640             # Image size
DEVICE = "0"               # GPU device ("0" for GPU, "cpu" for CPU)

# Output settings
SAVE_DIR = DATASET_DIR / "runs" / "detect" / "predictions"

# ══════════════════════════════════════════════════════════════════════════════


def test_image(image_path, model_path=None, conf=CONF_THRESHOLD, save=True, show=True):
    """
    Run inference on a single image or folder of images.
    
    Args:
        image_path: Path to image file or folder
        model_path: Path to trained model weights
        conf: Confidence threshold
        save: Save results to disk
        show: Display results
    
    Returns:
        results: Detection results
    """
    if model_path is None:
        model_path = DEFAULT_MODEL
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first using train_yolov8_detect.py")
        return None
    
    print("=" * 70)
    print("           YOLOv8 SPERM DETECTION - INFERENCE")
    print("=" * 70)
    print(f"\n📦 Loading model: {model_path}")
    
    # Load model
    model = YOLO(str(model_path))
    
    print(f"🖼️  Input: {image_path}")
    print(f"📊 Confidence threshold: {conf}")
    print()
    
    # Run inference
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=save,
        save_txt=save,           # Save labels
        save_conf=save,          # Save confidence scores
        project=str(SAVE_DIR.parent),
        name=SAVE_DIR.name,
        exist_ok=True,
        show=show,
        verbose=True,
    )
    
    # Print results summary
    print("\n" + "=" * 70)
    print("                    DETECTION RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results):
        boxes = result.boxes
        print(f"\n📷 Image {i+1}: {Path(result.path).name}")
        print(f"   Detections: {len(boxes)}")
        
        if len(boxes) > 0:
            for j, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                class_name = model.names[cls]
                xyxy = box.xyxy[0].tolist()
                print(f"   [{j+1}] {class_name}: {conf_score:.2%} | bbox: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
    
    if save:
        print(f"\n📁 Results saved to: {SAVE_DIR}")
    
    return results


def test_video(video_path, model_path=None, conf=CONF_THRESHOLD, save=True):
    """
    Run inference on a video file.
    
    Args:
        video_path: Path to video file
        model_path: Path to trained model weights
        conf: Confidence threshold
        save: Save results to disk
    
    Returns:
        results: Detection results
    """
    if model_path is None:
        model_path = DEFAULT_MODEL
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"🎥 Processing video: {video_path}")
    
    results = model.predict(
        source=str(video_path),
        conf=conf,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=save,
        project=str(SAVE_DIR.parent),
        name="video_predictions",
        exist_ok=True,
        stream=True,  # Stream results for videos
    )
    
    # Process stream
    for result in results:
        # Results are yielded frame by frame
        pass
    
    print(f"\n✓ Video processing complete!")
    return results


def test_webcam(model_path=None, conf=CONF_THRESHOLD):
    """
    Run real-time inference using webcam.
    
    Args:
        model_path: Path to trained model weights
        conf: Confidence threshold
    """
    if model_path is None:
        model_path = DEFAULT_MODEL
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print("📹 Starting webcam... Press 'q' to quit")
    
    results = model.predict(
        source=0,  # Webcam
        conf=conf,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=DEVICE,
        show=True,
        stream=True,
    )
    
    for result in results:
        # Press 'q' to quit (handled internally by ultralytics)
        pass


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Sperm Detection Inference")
    parser.add_argument("--image", type=str, help="Path to image or folder")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")
    
    args = parser.parse_args()
    
    if args.webcam:
        test_webcam(args.model, args.conf)
    elif args.video:
        test_video(args.video, args.model, args.conf, save=not args.no_save)
    elif args.image:
        test_image(args.image, args.model, args.conf, save=not args.no_save, show=not args.no_show)
    else:
        # Default: test on validation images
        val_images = DATASET_DIR / "valid" / "images"
        if val_images.exists():
            print("No input specified. Testing on validation images...")
            test_image(val_images, args.model, args.conf, save=not args.no_save, show=not args.no_show)
        else:
            print("Usage:")
            print("  python test_yolov8_detect.py --image path/to/image.jpg")
            print("  python test_yolov8_detect.py --image path/to/folder/")
            print("  python test_yolov8_detect.py --video path/to/video.mp4")
            print("  python test_yolov8_detect.py --webcam")


if __name__ == "__main__":
    main()
