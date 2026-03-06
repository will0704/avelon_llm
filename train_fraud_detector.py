"""
Train a fraud detection model for Avelon LLM.

Uses sklearn RandomForestClassifier on image features extracted by
FraudDetectorService.extract_image_features().

Training data is generated synthetically:
- Clean images → label 0 (not fraudulent)
- Manipulated images (compressed, blurred, edited) → label 1 (fraudulent)

Usage:
    python train_fraud_detector.py
    python train_fraud_detector.py --samples 200 --output app/models/fraud_detector.pkl
"""
import os
import sys
import argparse
import random
import logging
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from app.services.fraud_detector_service import FraudDetectorService

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Train fraud detector model")
    parser.add_argument(
        "--samples", type=int, default=200,
        help="Number of training samples per class"
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "app", "models", "fraud_detector.pkl"),
        help="Output path for the trained model"
    )
    parser.add_argument(
        "--real-images", type=str, default=None,
        help="Optional path to a folder of real document images (used as clean samples)"
    )
    return parser.parse_args()


def generate_clean_image() -> bytes:
    """Generate a clean, unmanipulated document-like image."""
    width = random.randint(400, 800)
    height = random.randint(300, 600)
    bg = (random.randint(240, 255), random.randint(240, 255), random.randint(235, 250))
    img = Image.new('RGB', (width, height), color=bg)

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    # Add text-like elements
    text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    try:
        font = ImageFont.truetype("arial.ttf", random.randint(12, 20))
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Simulate document text lines
    y = 20
    for _ in range(random.randint(5, 15)):
        line_len = random.randint(10, 40)
        text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=line_len))
        draw.text((20, y), text, fill=text_color, font=font)
        y += random.randint(18, 30)
        if y > height - 30:
            break

    # Add some lines/borders
    draw.rectangle([(10, 10), (width - 10, height - 10)], outline=(0, 0, 0), width=1)

    # Save with decent quality
    buf = BytesIO()
    fmt = random.choice(['PNG', 'JPEG'])
    if fmt == 'JPEG':
        img.save(buf, format='JPEG', quality=random.randint(85, 98))
    else:
        img.save(buf, format='PNG')
    return buf.getvalue()


def generate_manipulated_image() -> bytes:
    """Generate a manipulated/fraudulent document-like image."""
    width = random.randint(400, 800)
    height = random.randint(300, 600)
    bg = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    img = Image.new('RGB', (width, height), color=bg)

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Add some content
    for _ in range(random.randint(3, 8)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(10, 50)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color)

    # Apply manipulations
    manipulations = random.sample([
        'heavy_blur', 'heavy_compress', 'extreme_brightness',
        'posterize', 'clone_stamp', 'noise'
    ], k=random.randint(2, 4))

    for m in manipulations:
        if m == 'heavy_blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 8)))
        elif m == 'heavy_compress':
            pass  # Applied at save time
        elif m == 'extreme_brightness':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.choice([0.3, 0.4, 1.8, 2.0]))
        elif m == 'posterize':
            from PIL import ImageOps
            img = ImageOps.posterize(img, random.randint(1, 3))
        elif m == 'clone_stamp':
            # Copy a region to another part
            region_size = 50
            sx, sy = random.randint(0, max(width - region_size, 1)), random.randint(0, max(height - region_size, 1))
            region = img.crop((sx, sy, sx + region_size, sy + region_size))
            dx, dy = random.randint(0, max(width - region_size, 1)), random.randint(0, max(height - region_size, 1))
            img.paste(region, (dx, dy))
        elif m == 'noise':
            arr = np.array(img)
            noise = np.random.randint(-30, 30, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

    # Save with heavy compression
    buf = BytesIO()
    if 'heavy_compress' in manipulations:
        img.save(buf, format='JPEG', quality=random.randint(5, 20))
    else:
        img.save(buf, format='JPEG', quality=random.randint(30, 60))

    return buf.getvalue()


def load_real_images(folder: str) -> list:
    """Load real images from a folder as bytes."""
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if not os.path.isdir(folder):
        return images

    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in extensions:
            path = os.path.join(folder, f)
            with open(path, 'rb') as fh:
                images.append(fh.read())
    return images


def main():
    args = get_args()

    # Create a service instance for feature extraction (no model needed)
    service = FraudDetectorService(model_path=None)

    print(f"Generating {args.samples} samples per class...")

    features = []
    labels = []

    # Generate clean samples (label=0)
    real_images = []
    if args.real_images:
        real_images = load_real_images(args.real_images)
        print(f"  Loaded {len(real_images)} real images from {args.real_images}")

    for i in range(args.samples):
        if real_images and i < len(real_images):
            img_bytes = real_images[i]
        else:
            img_bytes = generate_clean_image()

        feat = service.extract_image_features(img_bytes)
        features.append(list(feat.values()))
        labels.append(0)

        if (i + 1) % 50 == 0:
            print(f"  Clean: {i + 1}/{args.samples}")

    # Generate manipulated samples (label=1)
    for i in range(args.samples):
        img_bytes = generate_manipulated_image()
        feat = service.extract_image_features(img_bytes)
        features.append(list(feat.values()))
        labels.append(1)

        if (i + 1) % 50 == 0:
            print(f"  Manipulated: {i + 1}/{args.samples}")

    X = np.array(features)
    y = np.array(labels)

    print(f"\nTotal samples: {len(y)} (clean={sum(y == 0)}, manipulated={sum(y == 1)})")
    print(f"Feature shape: {X.shape}")
    print(f"Feature names: {list(service.extract_image_features(generate_clean_image()).keys())}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train RandomForest
    print("\nTraining RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["clean", "manipulated"]))

    # Feature importance
    feature_names = list(service.extract_image_features(generate_clean_image()).keys())
    importance = clf.feature_importances_
    print("Feature Importances:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # Save model
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Set FRAUD_MODEL_PATH={output_path} in your .env")


if __name__ == "__main__":
    main()
