import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import ConvLSTM


# Example dataset class - replace with actual data loading logic once we get it done
class CrowdDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=10, height=64, width=64):
        """
        Placeholder dataset class for demonstration
        In real usage, this would load and preprocess density maps
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.height = height
        self.width = width

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic data for demonstration
        # Input: sequence of density maps (batch, seq_len, channels, H, W)
        x = torch.rand(self.seq_len, 1, self.height, self.width)

        # Target: binary crowd-crush risk map (batch, channels, H, W)
        # In real data, this would be the ground truth binary map
        y = torch.zeros(1, self.height, self.width)
        # Create some synthetic pattern (e.g., high risk in the center)
        center_h, center_w = self.height // 2, self.width // 2
        radius = min(self.height, self.width) // 4
        for h in range(self.height):
            for w in range(self.width):
                dist = ((h - center_h) ** 2 + (w - center_w) ** 2) ** 0.5
                if dist < radius:
                    y[0, h, w] = 1.0

        return x, y


def train(
    model, train_loader, val_loader, num_epochs, device, use_mixed_precision=True
):
    """
    Training function with mixed precision support
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Mixed precision training setup
    scaler = GradScaler() if use_mixed_precision else None

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            if use_mixed_precision:
                with autocast():
                    predictions = model(x_batch)
                    loss = criterion(predictions, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)

                loss.backward()
                optimizer.step()

            train_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_convlstm_model.pth")
            print(f"Model saved at epoch {epoch+1}")

    print("Training complete!")


def evaluate_with_uncertainty(model, test_loader, device, num_mc_samples=20):
    """
    Evaluate the model with uncertainty estimation using Monte Carlo dropout
    """
    model.eval()  # Still in eval mode, but dropout remains active for MC sampling

    all_mean_preds = []
    all_uncertainties = []
    all_targets = []

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Get predictions with uncertainty
        mean_pred, uncertainty = model.predict_with_uncertainty(
            x_batch, num_samples=num_mc_samples
        )

        all_mean_preds.append(mean_pred.cpu().numpy())
        all_uncertainties.append(uncertainty.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

    # Stack all batches
    all_mean_preds = np.concatenate(all_mean_preds, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute metrics (example: mean absolute error)
    mae = np.mean(np.abs(all_mean_preds - all_targets))
    mean_uncertainty = np.mean(all_uncertainties)

    print(f"Test MAE: {mae:.4f}, Mean uncertainty: {mean_uncertainty:.4f}")

    return all_mean_preds, all_uncertainties, all_targets


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 8
    num_epochs = 50
    in_channels = 1  # Number of input channels (density maps)
    hidden_channels = [64, 128, 128]  # Hidden dimensions for each layer
    kernel_size = 3  # Kernel size for convolutions
    num_layers = 3  # Number of ConvLSTM layers
    dropout_rate = 0.2  # Dropout probability

    # Create datasets
    train_dataset = CrowdDataset(num_samples=500, seq_len=10, height=64, width=64)
    val_dataset = CrowdDataset(num_samples=100, seq_len=10, height=64, width=64)
    test_dataset = CrowdDataset(num_samples=100, seq_len=10, height=64, width=64)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    model = ConvLSTM(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    train(model, train_loader, val_loader, num_epochs, device, use_mixed_precision=True)

    # Load best model for evaluation
    model.load_state_dict(torch.load("best_convlstm_model.pth"))

    # Evaluate with uncertainty
    mean_preds, uncertainties, targets = evaluate_with_uncertainty(
        model, test_loader, device
    )

    print("Done!")


if __name__ == "__main__":
    main()
