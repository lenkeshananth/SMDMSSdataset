import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import transforms, models
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import argparse
import sys

# Add parent directory to path to import pipeline if needed
sys.path.insert(0, str(Path(__file__).parent.parent))
from sperm_pipeline.pipeline import SpermAnalysisPipeline

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
DETECTION_MODEL = BASE_DIR / "sperm.v1i.yolov8" / "runs" / "detect" / "sperm_detection" / "weights" / "best.pt"
CLASSIFICATION_MODEL_P3 = BASE_DIR / "priority_3_augmentation" / "weights" / "efficientnet_b0_augmented.pt"

IMG_SIZE = 224
CLASS_NAMES = ["Combined_Anomaly", "Head_Anomaly", "Midpiece_Anomaly", "Normal", "Tail_Anomaly"]
OUTPUT_DIR = BASE_DIR / "report_tests" / "output_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_effnet(path, device):
    """Load the EfficientNet-B0 P3 model."""
    model = models.efficientnet_b0(weights=None)
    
    # Try the architecture used in P3
    try:
        model.classifier = nn.Sequential(nn.Dropout(p=0.35, inplace=True), nn.Linear(model.classifier[1].in_features, 5))
        model.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(p=0.4, inplace=True), nn.Linear(model.classifier[1].in_features, 5))
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
            model.load_state_dict(torch.load(path, map_location=device))
            
    model.to(device).eval()
    return model

def classify_effnet(model, crop, device):
    """Run EfficientNet-B0 classification on a cropped sperm image."""
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out = model(tfm(img).unsqueeze(0).to(device))
        probs = torch.nn.functional.softmax(out, dim=1)[0]
    idx = probs.argmax().item()
    return CLASS_NAMES[idx], float(probs[idx])

def get_random_frame(video_path):
    """Extract a random frame from the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_idx = random.randint(0, total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame at index {random_idx}")
        
    return frame, random_idx

def main():
    parser = argparse.ArgumentParser(description="Generate images for thesis report.")
    parser.add_argument("--video", required=True, help="Path to the test video")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("Loading models...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load YOLO Detection Model
    if not DETECTION_MODEL.exists():
        print(f"Error: YOLO detection model not found at {DETECTION_MODEL}")
        return
    # The pipeline class expects to load the model itself
    pipeline = SpermAnalysisPipeline(str(DETECTION_MODEL), "") 
    pipeline.load_models()
    
    # 2. Load EfficientNet P3 Classification Model
    if not CLASSIFICATION_MODEL_P3.exists():
        print(f"Error: EfficientNet P3 model not found at {CLASSIFICATION_MODEL_P3}")
        return
    effnet_model = load_effnet(CLASSIFICATION_MODEL_P3, device)
    print("Models loaded successfully.")

    print(f"Extracting random frame from {args.video}...")
    frame, frame_idx = get_random_frame(args.video)
    
    # Save Image 1: Raw Random Frame
    img1_path = OUTPUT_DIR / f"1_raw_frame_{frame_idx}.jpg"
    cv2.imwrite(str(img1_path), frame)
    print(f"Saved Image 1: {img1_path}")

    print("Running YOLO detection...")
    dets = pipeline.detect_sperm(frame)
    if not dets:
        print("No sperm detected in this frame. Try running again or providing a different seed.")
        return
        
    print(f"Detected {len(dets)} sperm cells.")
    
    # Save Image 2: Frame with Bounding Boxes
    frame_with_bboxes = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame_with_bboxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    img2_path = OUTPUT_DIR / f"2_detected_frame_{frame_idx}.jpg"
    cv2.imwrite(str(img2_path), frame_with_bboxes)
    print(f"Saved Image 2: {img2_path}")

    # Pick a random detected sperm that has a reasonable size (to avoid edge artifacts)
    valid_dets = [d for d in dets if (d["bbox"][2] - d["bbox"][0]) > 20 and (d["bbox"][3] - d["bbox"][1]) > 20]
    if not valid_dets:
        valid_dets = dets # fallback
        
    target_det = random.choice(valid_dets)
    
    print("Cropping target sperm...")
    crop = pipeline.crop_detection(frame, target_det["bbox"])
    
    if crop.size == 0:
        print("Error cropping sperm.")
        return

    # To make the cropped image visible in the report, resize it (it's usually very small)
    crop_display = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_CUBIC)

    # Save Image 3: Single Cropped Sperm
    img3_path = OUTPUT_DIR / f"3_cropped_sperm_{frame_idx}.jpg"
    cv2.imwrite(str(img3_path), crop_display)
    print(f"Saved Image 3: {img3_path}")

    print("Running EfficientNet-B0 P3 classification...")
    pred_class, conf = classify_effnet(effnet_model, crop, device)
    
    print(f"Predicted Class: {pred_class} (Confidence: {conf:.2f})")
    
    # Save Image 4: Cropped Sperm with Morphology Class prediction
    # Create a slightly larger canvas to draw the text if needed, or simply write on the resized crop
    img4_img = crop_display.copy()
    
    # Add a black border at the bottom for text
    border_size = 40
    img4_img = cv2.copyMakeBorder(img4_img, 0, border_size, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    text = f"{pred_class}"
    text2 = f"Conf: {conf:.2f}"
    
    # Make text fit
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    cv2.putText(img4_img, text, (5, 150 + 15), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(img4_img, text2, (5, 150 + 32), font, font_scale, (255, 255, 255), thickness)

    img4_path = OUTPUT_DIR / f"4_classified_sperm_{frame_idx}.jpg"
    cv2.imwrite(str(img4_path), img4_img)
    print(f"Saved Image 4: {img4_path}")
    
    print("\n✅ All 4 images successfully generated for the thesis report!")

if __name__ == "__main__":
    main()
