import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

from model import ConvLSTM
from utils import visualize_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvLSTM on density map data")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training input tensor file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data (if not provided, will split training data)",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default=None,
        help="Path to target data for supervised training (if not provided, will use train_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints and training logs",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="default",
        help='ConvLSTM model configuration: "small", "medium", "large", or "default"',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (if validation data not provided)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5,
        help="Save model checkpoint every N epochs",
    )
    parser.add_argument(
        "--visualize_freq",
        type=int,
        default=10,
        help="Visualize predictions every N epochs",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="autoencoder",
        choices=["autoencoder", "supervised", "binary"],
        help="Training mode: 'autoencoder' learns to reconstruct input sequence, 'supervised' learns to predict target data, 'binary' outputs binary risk maps",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training (faster but may affect accuracy)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint file",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Patience for learning rate scheduler",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def get_model_config(config_name, input_channels=1):
    """
    Get model configuration based on name
    """
    configs = {
        "small": {
            "in_channels": input_channels,
            "hidden_channels": [16, 32, 16],
            "kernel_size": 3,
            "num_layers": 3,
            "dropout_rate": 0.2,
        },
        "medium": {
            "in_channels": input_channels,
            "hidden_channels": [32, 64, 32],
            "kernel_size": 3,
            "num_layers": 3,
            "dropout_rate": 0.2,
        },
        "large": {
            "in_channels": input_channels,
            "hidden_channels": [64, 128, 64],
            "kernel_size": 5,
            "num_layers": 3,
            "dropout_rate": 0.3,
        },
        "default": {
            "in_channels": input_channels,
            "hidden_channels": [32, 64, 32],
            "kernel_size": 3,
            "num_layers": 3,
            "dropout_rate": 0.2,
        },
    }

    if config_name not in configs:
        print(f"Warning: Configuration '{config_name}' not found. Using default.")
        config_name = "default"

    return configs[config_name]


def create_datasets(
    train_data_path,
    val_data_path=None,
    target_data_path=None,
    val_split=0.2,
    train_mode="autoencoder",
):
    """
    Create training and validation datasets
    """
    # Load training data
    print(f"Loading training data from {train_data_path}")
    train_tensor = torch.load(train_data_path)
    print(f"Training data shape: {train_tensor.shape}")

    # Handle target data based on train mode
    if train_mode == "supervised" and target_data_path:
        print(f"Loading target data from {target_data_path}")
        target_tensor = torch.load(target_data_path)
        print(f"Target data shape: {target_tensor.shape}")
    else:
        if train_mode == "supervised" or train_mode == "binary":
            print(
                f"No target data provided for {train_mode} mode, using last frame as target"
            )
        # For autoencoder or supervised without target, use the last frame of each sequence as target
        target_tensor = train_tensor[:, -1, :, :, :]
        print(f"Using last frame as target, shape: {target_tensor.shape}")

        # For binary mode, threshold the density maps
        if train_mode == "binary":
            print("Converting targets to binary risk maps")
            # Use a threshold to convert density maps to binary risk maps
            # This is a simple approach, better to avoid using it for report
            threshold = 0.5  # Adjust as needed
            target_tensor = (target_tensor > threshold).float()

    # Create datasets
    dataset = TensorDataset(train_tensor, target_tensor)

    # Split into train and validation if validation data not provided
    if val_data_path is None:
        # Calculate split sizes
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        # Split the dataset
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        print(
            f"Split dataset into {train_size} training and {val_size} validation samples"
        )
    else:
        # Load validation data
        print(f"Loading validation data from {val_data_path}")
        val_tensor = torch.load(val_data_path)
        print(f"Validation data shape: {val_tensor.shape}")

        # Create validation target data
        if train_mode == "supervised" and target_data_path:
            # Need to handle validation targets separately - this is just a placeholder
            # In a real scenario, you would have separate validation targets
            val_target_tensor = val_tensor[:, -1, :, :, :]
        else:
            val_target_tensor = val_tensor[:, -1, :, :, :]

        # For binary mode, threshold the validation targets too
        if train_mode == "binary":
            threshold = 0.5  # Use the same threshold as for training
            val_target_tensor = (val_target_tensor > threshold).float()

        val_dataset = TensorDataset(val_tensor, val_target_tensor)
        train_dataset = dataset

    return train_dataset, val_dataset


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_mixed_precision=False,
    scaler=None,
):
    """
    Train for one epoch with optional mixed precision
    """
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        if use_mixed_precision:
            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)

    return total_loss / len(dataloader.dataset)


def get_loss_function(train_mode):
    """
    Get appropriate loss function for the training mode
    """
    if train_mode == "binary":
        # For binary risk prediction, use BCE loss
        return nn.BCELoss()
    else:
        # For regression (autoencoder, supervised), use MSE loss
        return nn.MSELoss()


def visualize_training_results(
    model, val_dataloader, epoch, output_dir, device, train_mode
):
    """
    Visualize predictions on validation data
    """
    model.eval()
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Get a batch of data
    inputs, targets = next(iter(val_dataloader))

    # Use only the first sample for visualization
    input_seq = inputs[0:1].to(device)
    target = targets[0:1].to(device)

    # Get prediction
    with torch.no_grad():
        prediction = model(input_seq)

    # Move to CPU and convert to numpy
    input_seq = input_seq.cpu().numpy()
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()

    # Example frame from the sequence (use last frame)
    input_frame = input_seq[0, -1, 0]  # Last frame, first channel

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Plot input frame
    plt.subplot(1, 3, 1)
    plt.imshow(input_frame, cmap="viridis")
    plt.title("Input (Last Frame)")
    plt.axis("off")

    # Plot ground truth
    plt.subplot(1, 3, 2)
    if train_mode == "binary":
        plt.imshow(target[0, 0], cmap="gray")
        plt.title("Ground Truth (Binary)")
    else:
        plt.imshow(target[0, 0], cmap="plasma")
        plt.title("Ground Truth")
    plt.axis("off")

    # Plot prediction
    plt.subplot(1, 3, 3)
    if train_mode == "binary":
        plt.imshow(prediction[0, 0], cmap="gray")
        plt.title("Prediction (Binary)")
    else:
        plt.imshow(prediction[0, 0], cmap="plasma")
        plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset, val_dataset = create_datasets(
        args.train_data,
        args.val_data,
        args.target_data,
        args.val_split,
        args.train_mode,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Get input channels from the data
    input_tensor, _ = train_dataset[0]
    _, _, channels, _, _ = input_tensor.shape

    # Create model
    model_config = get_model_config(args.model_config, input_channels=channels)

    print(f"Creating ConvLSTM model with configuration: {args.model_config}")
    model = ConvLSTM(
        in_channels=model_config["in_channels"],
        hidden_channels=model_config["hidden_channels"],
        kernel_size=model_config["kernel_size"],
        num_layers=model_config["num_layers"],
        dropout_rate=model_config["dropout_rate"],
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Define loss function and optimizer
    criterion = get_loss_function(args.train_mode)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience, factor=0.5, verbose=True
    )

    # Set up mixed precision training if enabled
    if args.mixed_precision and torch.cuda.is_available():
        print("Using mixed precision training")
        scaler = GradScaler()
    else:
        scaler = None
        if args.mixed_precision and not torch.cuda.is_available():
            print(
                "Mixed precision requested but not available. Using standard precision."
            )
            args.mixed_precision = False

    # Initialize training logs
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_epoch = 1

    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            # Load model state
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded model state")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights")

            # Try to load optimizer, scheduler, and training history
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Loaded optimizer state")

            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"Resuming from epoch {start_epoch}")

            if "train_loss" in checkpoint and "val_loss" in checkpoint:
                if isinstance(checkpoint["train_loss"], list):
                    train_losses = checkpoint["train_loss"]
                    val_losses = checkpoint["val_loss"]
                else:
                    train_losses = [checkpoint["train_loss"]]
                    val_losses = [checkpoint["val_loss"]]
                print("Loaded training history")

            if "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
                print(f"Best validation loss: {best_val_loss:.6f}")
        else:
            print(f"No checkpoint found at {args.resume}, starting from scratch")

    # Training loop
    print(f"Starting training for {args.epochs} epochs from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_mixed_precision=args.mixed_precision,
            scaler=scaler,
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        print(
            f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.7f}"
        )

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_config": model_config,
                    "args": vars(args),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                checkpoint_path,
            )
            print(f"Saved best model checkpoint to {checkpoint_path}")

        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch:03d}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_config": model_config,
                    "args": vars(args),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Visualize results periodically
        if epoch % args.visualize_freq == 0 or epoch == args.epochs:
            visualize_training_results(
                model, val_loader, epoch, args.output_dir, device, args.train_mode
            )

    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "model_config": model_config,
            "args": vars(args),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        final_checkpoint_path,
    )
    print(f"Saved final model to {final_checkpoint_path}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(start_epoch, start_epoch + len(train_losses)),
        train_losses,
        label="Train Loss",
    )
    plt.plot(
        range(start_epoch, start_epoch + len(val_losses)),
        val_losses,
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Save training history as JSON
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "epochs": list(range(start_epoch, args.epochs + 1)),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_config": args.model_config,
        "train_mode": args.train_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Training completed!")


if __name__ == "__main__":
    main()
