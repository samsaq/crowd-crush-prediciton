import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ConvLSTM
from utils import visualize_prediction, predict_risk_map, risk_level_to_color


def create_synthetic_density_sequence(seq_len=10, height=64, width=64, num_agents=100):
    """Create a synthetic sequence of density maps with moving agents"""
    sequence = torch.zeros(seq_len, 1, height, width)

    # Create random agent positions
    agents_x = torch.randint(0, width, (num_agents,))
    agents_y = torch.randint(0, height, (num_agents,))

    # Create random velocities
    velocity_x = torch.randn(num_agents) * 2  # Random x velocity
    velocity_y = torch.randn(num_agents) * 2  # Random y velocity

    # Simulate agent movement across frames
    for t in range(seq_len):
        # Update positions
        agents_x = (agents_x + velocity_x).round().long()
        agents_y = (agents_y + velocity_y).round().long()

        # Keep agents within bounds
        agents_x = torch.clamp(agents_x, 0, width - 1)
        agents_y = torch.clamp(agents_y, 0, height - 1)

        # Create density map for this frame
        density_map = torch.zeros(1, height, width)
        for i in range(num_agents):
            x, y = agents_x[i], agents_y[i]
            # Apply a small Gaussian kernel around each agent
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        # Distance from center
                        dist = np.sqrt(dx * dx + dy * dy)
                        # Gaussian falloff
                        value = np.exp(-(dist**2) / 2)
                        density_map[0, ny, nx] += value

        # Normalize the density map
        if density_map.max() > 0:
            density_map = density_map / density_map.max()

        sequence[t] = density_map

    return sequence


def create_synthetic_groundtruth(height=64, width=64):
    """Create a synthetic ground truth risk map with high risk in the center"""
    ground_truth = torch.zeros(1, height, width)
    center_h, center_w = height // 2, width // 2
    radius = min(height, width) // 4

    for h in range(height):
        for w in range(width):
            dist = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
            if dist < radius:
                # Higher risk in the center, decreasing outward
                risk = 1.0 - (dist / radius) * 0.7
                ground_truth[0, h, w] = risk

    return ground_truth


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a synthetic density map sequence
    print("Creating synthetic data...")
    height, width = 64, 64
    seq_len = 10
    input_sequence = create_synthetic_density_sequence(
        seq_len, height, width, num_agents=150
    )
    ground_truth = create_synthetic_groundtruth(height, width)

    # Create and initialize the model
    print("Initializing model...")
    model = ConvLSTM(
        in_channels=1,
        hidden_channels=[32, 64, 32],  # Smaller model for demo
        kernel_size=3,
        num_layers=3,
        dropout_rate=0.2,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Forward pass with uncertainty estimation
    print("Making prediction with uncertainty...")
    mean_pred, uncertainty = predict_risk_map(
        model, input_sequence, with_uncertainty=True, num_samples=10
    )

    # Visualize the results
    print("Creating visualization...")
    fig = visualize_prediction(
        input_sequence,
        ground_truth,
        mean_pred,
        uncertainty,
        frame_idx=5,  # Show the middle frame
    )

    # Create color-coded risk visualization
    print("Creating risk visualization...")
    risk_vis = risk_level_to_color(mean_pred, uncertainty)

    plt.figure(figsize=(10, 8))
    plt.imshow(risk_vis)
    plt.title("Risk Map with Uncertainty")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
