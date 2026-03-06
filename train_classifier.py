"""
Train the MobileNetV2 Document Classifier for Avelon LLM.

Supports N-class classification — auto-detects classes from subfolder names.
For example, data/ with subfolders: government_id/, not_id/, proof_of_income/, proof_of_address/

Usage:
    python train_classifier.py --data-dir <path_to_data>
    python train_classifier.py --data-dir data/4class --epochs 30
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Add app to path so we can import the model
sys.path.insert(0, os.path.dirname(__file__))
from app.services.classifier_service import DocumentClassifierModel


def get_args():
    parser = argparse.ArgumentParser(description="Train the document classifier")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "data", "classifier"
        ),
        help="Path to data folder with train/ and val/ subdirs (auto-detects classes from subfolder names)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (small due to small dataset)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="Unfreeze backbone for full fine-tuning")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "app", "models", "philid_classifier.pt"),
        help="Output path for the trained model checkpoint",
    )
    return parser.parse_args()


def main():
    args = get_args()

    # Resolve paths
    data_dir = os.path.abspath(args.data_dir)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        print(f"ERROR: Training directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.isdir(val_dir):
        print(f"ERROR: Validation directory not found: {val_dir}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data transforms ---
    # Training: augmentation to help with small dataset
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Validation: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Load data ---
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    classes = train_dataset.classes  # e.g. ['nonValid', 'validID']
    num_classes = len(classes)

    print(f"\nClasses found: {classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Build model ---
    # Use pretrained backbone for transfer learning
    model = DocumentClassifierModel(num_classes=num_classes)

    # Load pretrained MobileNetV2 weights for the backbone only
    pretrained = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Copy backbone weights (features), skip the classifier head
    backbone_state = {k: v for k, v in pretrained.state_dict().items() if k.startswith("features")}
    model.backbone.load_state_dict(backbone_state, strict=False)

    # Freeze backbone features, only train the classifier head
    if not args.unfreeze_backbone:
        for param in model.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen — training classifier head only")
    else:
        print("Backbone unfrozen — full fine-tuning")

    model.to(device)

    # --- Training setup ---
    criterion = nn.CrossEntropyLoss()
    # Only optimize the classifier head parameters
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # --- Training loop ---
    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # --- Save checkpoint ---
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build class mapping for ClassifierService
    # Map IDVerifier folder names to how classifier_service uses them
    class_mapping = {}
    for idx, cls_name in enumerate(classes):
        class_mapping[cls_name] = idx

    checkpoint = {
        "model_state_dict": best_state or model.state_dict(),
        "classes": classes,
        "class_mapping": class_mapping,
        "config": {
            "num_classes": num_classes,
            "img_size": 224,
        },
        "best_val_acc": best_val_acc,
    }

    torch.save(checkpoint, output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Classes: {classes}")
    print(f"\nTo use this model, set PHILID_MODEL_PATH={output_path} in your .env")


if __name__ == "__main__":
    main()
