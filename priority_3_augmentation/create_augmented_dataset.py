"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 3: Data Augmentation Pipeline
  ────────────────────────────────────────
  Creates a NEW augmented dataset from the ORIGINAL (non-oversampled) images.
  Applies aggressive, diverse augmentations especially for rare classes
  (Midpiece ~20 originals, Tail ~14 originals).

  Strategy:
    - Combined/Head (many originals):  generate 2 augmented copies each
    - Normal (92 originals):           generate 4 augmented copies each
    - Midpiece (~20 originals):        generate 20 augmented copies each
    - Tail (~14 originals):            generate 25 augmented copies each
  Target: ~350-400 images per class (all genuinely different)

  Usage:
      python priority_3_augmentation/create_augmented_dataset.py
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil
import hashlib

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
SOURCE_DIR = BASE_DIR / "classification_dataset_balanced" / "train"
VAL_DIR = BASE_DIR / "classification_dataset_balanced" / "val"

# Output
OUTPUT_TRAIN = BASE_DIR / "priority_3_augmented_dataset" / "train"
OUTPUT_VAL = BASE_DIR / "priority_3_augmented_dataset" / "val"

# How many augmented copies per original image, by class
# Chosen so each class ends up with ~350-400 total images
AUGMENT_MULTIPLIER = {
    "Combined_Anomaly": 1,    # ~341 unique → 341 + 341 = ~682, keep originals only
    "Head_Anomaly": 1,        # ~330 unique → keep originals only
    "Midpiece_Anomaly": 20,   # ~20 unique → 20 + 400 = ~420
    "Normal": 4,              # ~92 unique → 92 + 368 = ~460
    "Tail_Anomaly": 25,       # ~14 unique → 14 + 350 = ~364
}

# Target size after balancing (trim larger classes down to this)
TARGET_PER_CLASS = 400

# ═════════════════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_file_hash(filepath):
    """Get MD5 hash of file content to identify true duplicates."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_unique_files(folder):
    """Return only truly unique files (by content hash) from a folder."""
    seen_hashes = set()
    unique_files = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            h = get_file_hash(f)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_files.append(f)
    return unique_files


def augment_image(img_pil, aug_index):
    """
    Apply a diverse combination of augmentations based on the aug_index.
    Each index produces a different augmentation combo for maximum diversity.
    """
    img = img_pil.copy()
    random.seed(aug_index * 7919 + hash(str(img.size)))  # Deterministic but varied

    # ── 1. Geometric transforms ──
    # Rotation (random angle)
    angle = random.uniform(-45, 45)
    img = img.rotate(angle, fillcolor=(0, 0, 0), expand=False)

    # Random flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # ── 2. Scale/crop (simulate different zoom levels) ──
    if random.random() > 0.3:
        w, h = img.size
        scale = random.uniform(0.7, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((w, h), Image.LANCZOS)

    # ── 3. Color/brightness variations ──
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.6, 1.4))

    if random.random() > 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.6, 1.4))

    if random.random() > 0.4:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(random.uniform(0.5, 2.0))

    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    # ── 4. Blur/noise ──
    if random.random() > 0.6:
        radius = random.uniform(0.3, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # ── 5. Add slight noise (simulates video-quality crops) ──
    if random.random() > 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # ── 6. Elastic-like distortion via perspective transform ──
    if random.random() > 0.6:
        w, h = img.size
        magnitude = random.uniform(0.02, 0.08)
        coeffs = [
            1 + random.uniform(-magnitude, magnitude),
            random.uniform(-magnitude, magnitude),
            random.uniform(-5, 5),
            random.uniform(-magnitude, magnitude),
            1 + random.uniform(-magnitude, magnitude),
            random.uniform(-5, 5),
            random.uniform(-0.001, 0.001),
            random.uniform(-0.001, 0.001),
        ]
        img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    return img


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def create_augmented_dataset():
    print("═" * 65)
    print("  Priority 3: Creating Augmented Dataset")
    print("═" * 65)

    # ── Step 1: Deduplicate & count originals ──
    print("\n📊 Analyzing source dataset (deduplicating)...")
    class_stats = {}
    for class_dir in sorted(SOURCE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        unique = get_unique_files(class_dir)
        total = len(list(class_dir.iterdir()))
        class_stats[class_dir.name] = {
            "unique_files": unique,
            "total": total,
            "unique_count": len(unique),
        }
        print(f"  {class_dir.name:25s} {total:4d} total, {len(unique):4d} unique")

    # ── Step 2: Generate augmented dataset ──
    print(f"\n🔧 Generating augmented dataset → {OUTPUT_TRAIN}")

    for class_name, stats in class_stats.items():
        out_dir = OUTPUT_TRAIN / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        unique_files = stats["unique_files"]
        multiplier = AUGMENT_MULTIPLIER.get(class_name, 1)
        count = 0

        print(f"\n  ── {class_name} ({stats['unique_count']} unique) ──")
        print(f"     Multiplier: {multiplier}x augmentation per original")

        # Copy originals first
        for f in unique_files:
            dst = out_dir / f"orig_{f.stem}{f.suffix}"
            shutil.copy2(f, dst)
            count += 1

        # Generate augmented copies
        aug_count = 0
        for f in unique_files:
            img = Image.open(f).convert("RGB")
            for aug_i in range(multiplier):
                aug_img = augment_image(img, aug_i)
                aug_path = out_dir / f"aug_{f.stem}_v{aug_i:03d}.jpg"
                aug_img.save(aug_path, quality=95)
                aug_count += 1
                count += 1

        # Trim to target if too many
        all_files = sorted(out_dir.iterdir())
        if len(all_files) > TARGET_PER_CLASS:
            random.shuffle(all_files)
            for excess in all_files[TARGET_PER_CLASS:]:
                excess.unlink()
            count = TARGET_PER_CLASS

        print(f"     Generated: {aug_count} augmented copies")
        print(f"     Final count: {count}")

    # ── Step 3: Copy validation set as-is ──
    print(f"\n📁 Copying validation set → {OUTPUT_VAL}")
    if OUTPUT_VAL.exists():
        shutil.rmtree(OUTPUT_VAL)
    shutil.copytree(VAL_DIR, OUTPUT_VAL)

    # ── Summary ──
    print(f"\n{'═' * 65}")
    print(f"  Augmented Dataset Summary")
    print(f"{'═' * 65}")
    for class_dir in sorted(OUTPUT_TRAIN.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.iterdir()))
            print(f"  {class_dir.name:25s} {count:4d} images")

    val_total = 0
    for class_dir in sorted(OUTPUT_VAL.iterdir()):
        if class_dir.is_dir():
            val_total += len(list(class_dir.iterdir()))
    print(f"\n  Validation: {val_total} images (unchanged)")
    print(f"  Output: {BASE_DIR / 'priority_3_augmented_dataset'}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    create_augmented_dataset()
