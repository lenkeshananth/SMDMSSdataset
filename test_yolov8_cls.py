"""
YOLOv8 Classification Testing Script for SMDSS Sperm Dataset
=============================================================
This script tests a trained YOLOv8 classification model on images.

Usage:
    python test_yolov8_cls.py                           # Test on sample images
    python test_yolov8_cls.py --image path/to/image.png # Test single image
    python test_yolov8_cls.py --folder path/to/folder   # Test folder of images
    python test_yolov8_cls.py --val                     # Evaluate on validation set
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# ──────────────────────────────── CONFIGURATION ────────────────────────────────
DATASET_DIR = Path(r"d:\paper\SMDMSSdataset-20260226T103306Z-1-001\SMDMSSdataset")
MODEL_PATH = DATASET_DIR / "runs" / "smdss_sperm_cls" / "weights" / "best.pt"
IMAGES_DIR = DATASET_DIR / "images"
VAL_DIR = DATASET_DIR / "classification_dataset" / "val"
IMG_SIZE = 224

CLASS_NAMES = ["Normal", "Head_Anomaly", "Midpiece_Anomaly", "Tail_Anomaly", "Combined_Anomaly"]
# ───────────────────────────────────────────────────────────────────────────────


def load_model(model_path=None):
    """Load the trained YOLOv8 classification model."""
    path = Path(model_path) if model_path else MODEL_PATH
    
    if not path.exists():
        print(f"Error: Model not found at {path}")
        print("Make sure you have trained the model first using train_yolov8_cls.py")
        return None
    
    print(f"Loading model from: {path}")
    model = YOLO(str(path))
    return model


def predict_single_image(model, image_path, verbose=True):
    """
    Run prediction on a single image.
    
    Returns:
        dict with prediction results
    """
    results = model.predict(source=str(image_path), imgsz=IMG_SIZE, verbose=False)
    
    for result in results:
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = CLASS_NAMES[top1_idx]
        
        if verbose:
            print(f"\n  Image: {Path(image_path).name}")
            print(f"  Predicted: {class_name} ({top1_conf*100:.1f}%)")
            print(f"  All probabilities:")
            for i, prob in enumerate(probs.data.tolist()):
                bar = "█" * int(prob * 20)
                print(f"    {CLASS_NAMES[i]:<23} {prob*100:>5.1f}% {bar}")
        
        return {
            "image": Path(image_path).name,
            "predicted_class": class_name,
            "predicted_idx": top1_idx,
            "confidence": top1_conf,
            "all_probs": {CLASS_NAMES[i]: p for i, p in enumerate(probs.data.tolist())}
        }
    
    return None


def predict_folder(model, folder_path, save_results=True):
    """
    Run prediction on all images in a folder.
    
    Returns:
        list of prediction results
    """
    folder = Path(folder_path)
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {folder}")
        return []
    
    print(f"\nTesting {len(images)} images from: {folder}")
    print("=" * 60)
    
    results_list = []
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    
    for img_path in sorted(images):
        result = predict_single_image(model, img_path, verbose=True)
        if result:
            results_list.append(result)
            class_counts[result["predicted_class"]] += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"\n  Total images: {len(results_list)}")
    print(f"\n  {'Class':<25} {'Count':>6} {'Percentage':>12}")
    print("  " + "-" * 45)
    for cls in CLASS_NAMES:
        count = class_counts[cls]
        pct = (count / len(results_list) * 100) if results_list else 0
        print(f"  {cls:<25} {count:>6} {pct:>11.1f}%")
    
    # Save results to CSV
    if save_results and results_list:
        csv_path = folder / "predictions.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("image,predicted_class,confidence\n")
            for r in results_list:
                f.write(f"{r['image']},{r['predicted_class']},{r['confidence']:.4f}\n")
        print(f"\n  Results saved to: {csv_path}")
    
    return results_list


def evaluate_validation_set(model):
    """
    Evaluate the model on the validation set and compute metrics.
    """
    if not VAL_DIR.exists():
        print(f"Error: Validation directory not found at {VAL_DIR}")
        print("Make sure you have prepared the dataset first using train_yolov8_cls.py")
        return None
    
    print("\n" + "=" * 60)
    print("  Evaluating on Validation Set")
    print("=" * 60)
    
    # Use YOLO's built-in validation
    metrics = model.val(data=str(DATASET_DIR / "classification_dataset"), imgsz=IMG_SIZE, verbose=True)
    
    print("\n" + "=" * 60)
    print("  Validation Results")
    print("=" * 60)
    print(f"  Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1*100:.2f}%)")
    print(f"  Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5*100:.2f}%)")
    
    return metrics


def interactive_test(model):
    """
    Interactive testing mode - enter image paths to test.
    """
    print("\n" + "=" * 60)
    print("  Interactive Testing Mode")
    print("=" * 60)
    print("  Enter image path to test (or 'q' to quit)")
    print("  You can also drag and drop images into the terminal")
    
    while True:
        print()
        user_input = input("  Image path: ").strip().strip('"').strip("'")
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("  Exiting...")
            break
        
        if not user_input:
            continue
        
        image_path = Path(user_input)
        if not image_path.exists():
            print(f"  Error: File not found: {image_path}")
            continue
        
        if image_path.is_dir():
            predict_folder(model, image_path)
        else:
            predict_single_image(model, image_path)


# ──────────────────────────────── MAIN ────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8 classification model on sperm images")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (default: best.pt from training)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to test")
    parser.add_argument("--folder", type=str, default=None, help="Path to folder of images to test")
    parser.add_argument("--val", action="store_true", help="Evaluate on validation set")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive testing mode")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample images to test (default: 5)")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        exit(1)
    
    print(f"\nModel loaded successfully!")
    print(f"Classes: {CLASS_NAMES}")
    
    if args.val:
        # Evaluate on validation set
        evaluate_validation_set(model)
    
    elif args.image:
        # Test single image
        predict_single_image(model, args.image)
    
    elif args.folder:
        # Test folder of images
        predict_folder(model, args.folder)
    
    elif args.interactive:
        # Interactive mode
        interactive_test(model)
    
    else:
        # Default: test sample images from the dataset
        print("\n" + "=" * 60)
        print(f"  Testing {args.sample} Sample Images")
        print("=" * 60)
        
        sample_images = list(IMAGES_DIR.iterdir())[:args.sample]
        for img in sample_images:
            predict_single_image(model, img)
        
        print("\n" + "-" * 60)
        print("  Tip: Use --help to see all options")
        print("       Use --interactive for interactive testing")
        print("       Use --val to evaluate on validation set")
    
    print("\nDone!")
