"""
Training script for the Detector model (Part 2)
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from homework.datasets import road_dataset
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def train_epoch(model, train_loader, optimizer, seg_criterion, depth_criterion, device, depth_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_depth_loss = 0
    metric = DetectionMetric(num_classes=3)

    for batch in train_loader:
        images = batch["image"].to(device)
        track_labels = batch["track"].to(device)
        depth_labels = batch["depth"].to(device)

        # Forward pass
        logits, depth_pred = model(images)

        # Compute losses
        seg_loss = seg_criterion(logits, track_labels)
        depth_loss = depth_criterion(depth_pred, depth_labels)

        # Combined loss
        loss = seg_loss + depth_weight * depth_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_depth_loss += depth_loss.item()

        # Track metrics
        with torch.no_grad():
            pred, depth = model.predict(images)
            metric.add(pred.cpu(), track_labels.cpu(), depth.cpu(), depth_labels.cpu())

    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_depth_loss = total_depth_loss / len(train_loader)
    metrics = metric.compute()

    return avg_loss, avg_seg_loss, avg_depth_loss, metrics


@torch.inference_mode()
def validate(model, val_loader, seg_criterion, depth_criterion, device, depth_weight=1.0):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_depth_loss = 0
    metric = DetectionMetric(num_classes=3)

    for batch in val_loader:
        images = batch["image"].to(device)
        track_labels = batch["track"].to(device)
        depth_labels = batch["depth"].to(device)

        # Forward pass
        logits, depth_pred = model(images)

        # Compute losses
        seg_loss = seg_criterion(logits, track_labels)
        depth_loss = depth_criterion(depth_pred, depth_labels)

        # Combined loss
        loss = seg_loss + depth_weight * depth_loss

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_depth_loss += depth_loss.item()

        # Track metrics
        pred, depth = model.predict(images)
        metric.add(pred.cpu(), track_labels.cpu(), depth.cpu(), depth_labels.cpu())

    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)
    avg_depth_loss = total_depth_loss / len(val_loader)
    metrics = metric.compute()

    return avg_loss, avg_seg_loss, avg_depth_loss, metrics


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
    model = Detector(in_channels=3, num_classes=3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")

    # Loading the data
    print("\nLoading data...")
    train_loader = road_dataset.load_data(
        args.train_path,
        transform_pipeline=args.transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = road_dataset.load_data(
        args.val_path,
        transform_pipeline="default",  # No augmentation for validation
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Creating loss functions
    # Use class weights to handle class imbalance (background vs lane boundaries)
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()  # Mean Absolute Error for depth

    # Creating optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Logging - Tensorboard
    log_dir = Path("logs") / f"detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    # Running the optimizer for several epochs
    best_iou = 0.0
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        # Train
        train_loss, train_seg_loss, train_depth_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, seg_criterion, depth_criterion, device, args.depth_weight
        )

        # Validate
        val_loss, val_seg_loss, val_depth_loss, val_metrics = validate(
            model, val_loader, seg_criterion, depth_criterion, device, args.depth_weight
        )

        # Update learning rate based on IoU
        scheduler.step(val_metrics["iou"])

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("SegLoss/train", train_seg_loss, epoch)
        writer.add_scalar("SegLoss/val", val_seg_loss, epoch)
        writer.add_scalar("DepthLoss/train", train_depth_loss, epoch)
        writer.add_scalar("DepthLoss/val", val_depth_loss, epoch)

        writer.add_scalar("IOU/train", train_metrics["iou"], epoch)
        writer.add_scalar("IOU/val", val_metrics["iou"], epoch)
        writer.add_scalar("DepthError/train", train_metrics["abs_depth_error"], epoch)
        writer.add_scalar("DepthError/val", val_metrics["abs_depth_error"], epoch)
        writer.add_scalar("TPDepthError/train", train_metrics["tp_depth_error"], epoch)
        writer.add_scalar("TPDepthError/val", val_metrics["tp_depth_error"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Depth: {train_depth_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Depth: {val_depth_loss:.4f})")
        print(f"  Train IoU: {train_metrics['iou']:.4f}, Depth Err: {train_metrics['abs_depth_error']:.4f}, TP Depth Err: {train_metrics['tp_depth_error']:.4f}")
        print(f"  Val IoU: {val_metrics['iou']:.4f}, Depth Err: {val_metrics['abs_depth_error']:.4f}, TP Depth Err: {val_metrics['tp_depth_error']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Saving the model (save best model based on IoU)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            model_path = save_model(model)
            print(f"  ✓ New best model saved! Val IoU: {val_metrics['iou']:.4f} -> {model_path}")

        print()

    writer.close()

    print(f"\nTraining complete!")
    print(f"Best validation IoU: {best_iou:.4f}")

    # Check if we met the requirements
    if best_iou >= 0.75:
        print(f"✓ Met the IoU requirement (0.75)")
    else:
        print(f"✗ Did not meet the IoU requirement (0.75)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Detector model")

    # Data paths
    parser.add_argument(
        "--train_path",
        type=str,
        default="drive_data/train",
        help="Path to training data",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="drive_data/val",
        help="Path to validation data",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--depth_weight", type=float, default=1.0, help="Weight for depth loss")

    # Data augmentation
    parser.add_argument(
        "--transform",
        type=str,
        default="default",
        choices=["default", "aug"],
        help="Transform pipeline to use",
    )

    # System
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data workers")

    args = parser.parse_args()

    main(args)
