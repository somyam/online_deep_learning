"""
Training script for the Classifier model (Part 1)
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from homework.datasets import classification_dataset
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metric = AccuracyMetric()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy
        with torch.no_grad():
            preds = model.predict(images)
            metric.add(preds.cpu(), labels.cpu())

    avg_loss = total_loss / len(train_loader)
    metrics = metric.compute()

    return avg_loss, metrics["accuracy"]


@torch.inference_mode()
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    metric = AccuracyMetric()

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        # Track accuracy
        preds = model.predict(images)
        metric.add(preds.cpu(), labels.cpu())

    avg_loss = total_loss / len(val_loader)
    metrics = metric.compute()

    return avg_loss, metrics["accuracy"]


def main(args):
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Creating a model
    model = Classifier(in_channels=3, num_classes=6).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")

    # Loading the data
    print("\nLoading data...")
    train_loader = classification_dataset.load_data(
        args.train_path,
        transform_pipeline=args.transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = classification_dataset.load_data(
        args.val_path,
        transform_pipeline="default",  # No augmentation for validation
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Creating loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Logging - Tensorboard
    log_dir = Path("logs") / f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Running the optimizer for several epochs
    best_val_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_acc)

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Saving the model (save best model based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = save_model(model)
            print(f"  ✓ New best model saved! Val Acc: {val_acc:.4f} -> {model_path}")

        print()

    writer.close()

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Check if we met the requirement
    if best_val_acc >= 0.80:
        print(f"✓ Met the minimum requirement (0.80)")
    else:
        print(f"✗ Did not meet the minimum requirement (0.80)")

    if best_val_acc >= 0.90:
        print(f"✓ Achieved extra credit threshold (0.90)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Classifier model")

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="classification_data/train",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="classification_data/val",
        help="Path to validation data",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # Data augmentation
    parser.add_argument(
        "--transform",
        type=str,
        default="aug",
        choices=["default", "aug"],
        help="Transform pipeline to use",
    )

    # System
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data workers")

    args = parser.parse_args()

    main(args)
