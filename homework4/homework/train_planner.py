"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 100
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model


def train(
    model_name: str = "mlp_planner",
    transform_pipeline: str | None = None,
    num_workers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 128,
    num_epochs: int = 100,
    log_dir: Path | None = None,
):
    """
    Train a planner model.

    Args:
        model_name: name of the model to train
        transform_pipeline: data transformation pipeline
        num_workers: number of data loading workers
        lr: learning rate
        batch_size: batch size
        num_epochs: number of epochs to train
        log_dir: directory to save logs
    """
    if log_dir is None:
        log_dir = Path("logs") / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"

    log_dir = Path(log_dir)

    # Auto-select transform pipeline based on model type
    if transform_pipeline is None:
        if model_name == "cnn_planner":
            transform_pipeline = "default"  # Includes images
        else:
            transform_pipeline = "state_only"  # Only track boundaries

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using transform pipeline: {transform_pipeline}")

    # Validate transform pipeline matches model type
    if model_name == "cnn_planner" and transform_pipeline != "default":
        print(f"WARNING: CNNPlanner requires 'default' transform (includes images)")
        print(f"         Overriding transform_pipeline from '{transform_pipeline}' to 'default'")
        transform_pipeline = "default"

    # Create model
    if model_name == "mlp_planner":
        model = MLPPlanner()
    elif model_name == "transformer_planner":
        model = TransformerPlanner()
    elif model_name == "cnn_planner":
        model = CNNPlanner()
    else:
        model = load_model(model_name)

    model = model.to(device)
    model.train()

    # Create data loaders
    print("Loading training data...")
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    print("Loading validation data...")
    val_loader = load_data(
        "drive_data/val",
        transform_pipeline=transform_pipeline,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Logs will be saved to {log_dir}")

    best_val_error = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metric.reset()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Forward pass - different inputs for different models
            if model_name == "cnn_planner":
                image = batch["image"].to(device)
                pred_waypoints = model(image=image)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                pred_waypoints = model(track_left=track_left, track_right=track_right)

            # Compute loss (only on valid waypoints)
            loss = loss_fn(
                pred_waypoints * waypoints_mask[..., None],
                waypoints * waypoints_mask[..., None]
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            train_metric.add(pred_waypoints, waypoints, waypoints_mask)
            epoch_loss += loss.item()

            # Log to tensorboard
            if batch_idx % 10 == 0:
                writer.add_scalar("train/loss_step", loss.item(), global_step)

            global_step += 1

        # Compute training metrics
        train_metrics = train_metric.compute()
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass - different inputs for different models
                if model_name == "cnn_planner":
                    image = batch["image"].to(device)
                    pred_waypoints = model(image=image)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    pred_waypoints = model(track_left=track_left, track_right=track_right)

                loss = loss_fn(
                    pred_waypoints * waypoints_mask[..., None],
                    waypoints * waypoints_mask[..., None]
                )

                val_metric.add(pred_waypoints, waypoints, waypoints_mask)
                val_loss += loss.item()

        # Compute validation metrics
        val_metrics = val_metric.compute()
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate
        scheduler.step(val_metrics["l1_error"])

        # Log to tensorboard
        writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
        writer.add_scalar("train/longitudinal_error", train_metrics["longitudinal_error"], epoch)
        writer.add_scalar("train/lateral_error", train_metrics["lateral_error"], epoch)
        writer.add_scalar("train/l1_error", train_metrics["l1_error"], epoch)

        writer.add_scalar("val/loss_epoch", avg_val_loss, epoch)
        writer.add_scalar("val/longitudinal_error", val_metrics["longitudinal_error"], epoch)
        writer.add_scalar("val/lateral_error", val_metrics["lateral_error"], epoch)
        writer.add_scalar("val/l1_error", val_metrics["l1_error"], epoch)
        writer.add_scalar("val/learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Long: {train_metrics['longitudinal_error']:.4f}, Lat: {train_metrics['lateral_error']:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Long: {val_metrics['longitudinal_error']:.4f}, Lat: {val_metrics['lateral_error']:.4f}")

        # Save best model
        if val_metrics["l1_error"] < best_val_error:
            best_val_error = val_metrics["l1_error"]
            save_path = save_model(model)
            print(f"  Saved best model to {save_path} (val error: {best_val_error:.4f})")

    writer.close()
    print("\nTraining complete!")
    print(f"Best validation L1 error: {best_val_error:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a planner model")
    parser.add_argument("--model", type=str, default="mlp_planner", help="Model name")
    parser.add_argument("--transform", type=str, default=None, help="Transform pipeline (auto-selects based on model if not specified)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory")

    args = parser.parse_args()

    train(
        model_name=args.model,
        transform_pipeline=args.transform,
        num_workers=args.num_workers,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        log_dir=args.log_dir,
    )
