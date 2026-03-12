"""
═══════════════════════════════════════════════════════════════════════════════
  Priority 3 (Large): Create Large Augmented Dataset
  ───────────────────────────────────────────────────
  Creates a MUCH LARGER augmented dataset (~1500 images per class)
  compared to the original P3 dataset (~400/class).

  Uses more diverse augmentation with additional techniques:
    - Rotation, flips, scaling, cropping
    - Brightness, contrast, sharpness, color jitter
    - Gaussian blur, noise injection
    - Perspective warping
    - Channel shuffling
    - Histogram equalization
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Cutout / random erasing

  Usage:
      python priority_3_augmentation/create_large_augmented_dataset.py
═══════════════════════════════════════════════════════════════════════════════
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import shutil
import hashlib

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(r"D:\Krishna\SMDMSSdataset")
SOURCE_DIR = BASE_DIR / "classification_dataset_balanced" / "train"
VAL_DIR = BASE_DIR / "classification_dataset_balanced" / "val"

# Output — new larger dataset
OUTPUT_TRAIN = BASE_DIR / "priority_3_large_augmented_dataset" / "train"
OUTPUT_VAL = BASE_DIR / "priority_3_large_augmented_dataset" / "val"

# Target: ~1500 images per class
TARGET_PER_CLASS = 1500

# Multipliers based on unique counts:
#   Combined ~341 unique → need ~4-5x
#   Head     ~330 unique → need ~4-5x
#   Normal   ~92  unique → need ~16x
#   Midpiece ~20  unique → need ~75x
#   Tail     ~14  unique → need ~107x
AUGMENT_MULTIPLIER = {
    "Combined_Anomaly": 5,
    "Head_Anomaly": 5,
    "Midpiece_Anomaly": 75,
    "Normal": 16,
    "Tail_Anomaly": 107,
}

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═════════════════════════════════════════════════════════════════════════════

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_unique_files(folder):
    seen = set()
    unique = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            h = get_file_hash(f)
            if h not in seen:
                seen.add(h)
                unique.append(f)
    return unique


# ═════════════════════════════════════════════════════════════════════════════
# AUGMENTATION FUNCTIONS (extended set for maximum diversity)
# ═════════════════════════════════════════════════════════════════════════════

def apply_cutout(img, n_holes=2, max_size=20):
    """Randomly erase rectangular patches (simulates occlusion)."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    for _ in range(n_holes):
        sz_h = random.randint(5, max_size)
        sz_w = random.randint(5, max_size)
        y = random.randint(0, h - sz_h)
        x = random.randint(0, w - sz_w)
        arr[y:y+sz_h, x:x+sz_w] = random.randint(0, 50)
    return Image.fromarray(arr)


def apply_clahe(img):
    """CLAHE contrast enhancement (common in microscopy)."""
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 4.0),
                             tileGridSize=(8, 8))
    arr[:, :, 0] = clahe.apply(arr[:, :, 0])
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_LAB2RGB))


def apply_channel_shuffle(img):
    """Randomly swap color channels."""
    arr = np.array(img)
    channels = list(range(3))
    random.shuffle(channels)
    return Image.fromarray(arr[:, :, channels])


def apply_elastic(img, alpha=15, sigma=3):
    """Simplified elastic deformation."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32),
                           (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32),
                           (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    result = np.zeros_like(arr)
    for c in range(3):
        result[:, :, c] = cv2.remap(arr[:, :, c], map_x, map_y,
                                     interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(result.astype(np.uint8))


def augment_image(img_pil, aug_index):
    """
    Apply a diverse, deterministic combination of augmentations.
    Each aug_index produces a unique combination.
    """
    img = img_pil.copy()
    # Use aug_index to seed — gives deterministic but varied results
    rng = random.Random(aug_index * 7919 + hash(str(img.size)) % 100000)

    # ── 1. Geometric transforms ──
    # Rotation
    angle = rng.uniform(-60, 60)
    img = img.rotate(angle, fillcolor=(0, 0, 0), expand=False)

    # Flips
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # ── 2. Scale/crop ──
    if rng.random() > 0.25:
        w, h = img.size
        scale = rng.uniform(0.6, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            left = rng.randint(0, max(w - new_w, 0))
            top = rng.randint(0, max(h - new_h, 0))
            img = img.crop((left, top, left + new_w, top + new_h))
            img = img.resize((w, h), Image.LANCZOS)

    # ── 3. Color augmentations ──
    if rng.random() > 0.25:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.5, 1.5))
    if rng.random() > 0.25:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 1.5))
    if rng.random() > 0.35:
        img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.3, 2.5))
    if rng.random() > 0.4:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.5))

    # ── 4. Blur ──
    if rng.random() > 0.6:
        radius = rng.uniform(0.3, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # ── 5. Noise injection ──
    if rng.random() > 0.4:
        arr = np.array(img).astype(np.float32)
        noise = np.random.RandomState(aug_index).normal(0, rng.uniform(3, 20), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # ── 6. Perspective warp ──
    if rng.random() > 0.5:
        w, h = img.size
        mag = rng.uniform(0.02, 0.1)
        coeffs = [
            1 + rng.uniform(-mag, mag), rng.uniform(-mag, mag), rng.uniform(-5, 5),
            rng.uniform(-mag, mag), 1 + rng.uniform(-mag, mag), rng.uniform(-5, 5),
            rng.uniform(-0.001, 0.001), rng.uniform(-0.001, 0.001),
        ]
        try:
            img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        except Exception:
            pass

    # ── 7. CLAHE (microscopy-specific) ──
    if rng.random() > 0.6:
        img = apply_clahe(img)

    # ── 8. Channel shuffle (rare, for diversity) ──
    if rng.random() > 0.85:
        img = apply_channel_shuffle(img)

    # ── 9. Cutout / random erasing ──
    if rng.random() > 0.6:
        img = apply_cutout(img, n_holes=rng.randint(1, 3), max_size=rng.randint(10, 25))

    # ── 10. Elastic deformation ──
    if rng.random() > 0.7:
        img = apply_elastic(img, alpha=rng.uniform(8, 25), sigma=rng.uniform(2, 5))

    # ── 11. Histogram equalization (occasional) ──
    if rng.random() > 0.8:
        img = ImageOps.equalize(img)

    # ── 12. Invert (very rare, for extreme diversity) ──
    if rng.random() > 0.95:
        img = ImageOps.invert(img)

    return img


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def create_large_augmented_dataset():
    print("═" * 70)
    print("  Priority 3 (LARGE): Creating Large Augmented Dataset")
    print("  Target: ~1500 images per class")
    print("═" * 70)

    # ── Step 1: Deduplicate ──
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
        mult = AUGMENT_MULTIPLIER.get(class_dir.name, 1)
        expected = len(unique) + len(unique) * mult
        print(f"  {class_dir.name:25s} {total:4d} total, {len(unique):4d} unique "
              f"→ {mult}x aug → ~{min(expected, TARGET_PER_CLASS)} target")

    # ── Step 2: Generate ──
    print(f"\n🔧 Generating large augmented dataset → {OUTPUT_TRAIN}\n")

    for class_name, stats in class_stats.items():
        out_dir = OUTPUT_TRAIN / class_name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        unique_files = stats["unique_files"]
        multiplier = AUGMENT_MULTIPLIER.get(class_name, 1)
        count = 0

        print(f"  ── {class_name} ({stats['unique_count']} unique, {multiplier}x) ──")

        # Copy originals
        for f in unique_files:
            dst = out_dir / f"orig_{f.stem}{f.suffix}"
            shutil.copy2(f, dst)
            count += 1

        # Generate augmented copies
        aug_count = 0
        for f in unique_files:
            img = Image.open(f).convert("RGB")
            for aug_i in range(multiplier):
                aug_img = augment_image(img, aug_i + hash(f.stem) % 100000)
                aug_path = out_dir / f"aug_{f.stem}_v{aug_i:04d}.jpg"
                aug_img.save(aug_path, quality=95)
                aug_count += 1
                count += 1

        # Trim to target if too many
        all_files = sorted(out_dir.iterdir())
        if len(all_files) > TARGET_PER_CLASS:
            random.seed(42)
            random.shuffle(all_files)
            for excess in all_files[TARGET_PER_CLASS:]:
                excess.unlink()
            count = TARGET_PER_CLASS

        print(f"     Originals: {stats['unique_count']}, Augmented: {aug_count}, "
              f"Final: {min(count, TARGET_PER_CLASS)}")

    # ── Step 3: Copy validation set as-is ──
    print(f"\n📁 Copying validation set → {OUTPUT_VAL}")
    if OUTPUT_VAL.exists():
        shutil.rmtree(OUTPUT_VAL)
    shutil.copytree(VAL_DIR, OUTPUT_VAL)

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print(f"  Large Augmented Dataset Summary")
    print(f"{'═' * 70}")
    grand_total = 0
    for class_dir in sorted(OUTPUT_TRAIN.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.iterdir()))
            grand_total += count
            print(f"  {class_dir.name:25s} {count:5d} images")

    val_total = 0
    for class_dir in sorted(OUTPUT_VAL.iterdir()):
        if class_dir.is_dir():
            val_total += len(list(class_dir.iterdir()))

    print(f"\n  Training total:   {grand_total:,d} images")
    print(f"  Validation total: {val_total} images (unchanged)")
    print(f"  Output: {BASE_DIR / 'priority_3_large_augmented_dataset'}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    create_large_augmented_dataset()
