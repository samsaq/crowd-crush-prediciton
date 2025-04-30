import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from model import ConvLSTM


def visualize_prediction(
    input_seq, true_map, pred_map, uncertainty_map=None, frame_idx=None
):
    """
    Visualize input sequence, ground truth, prediction, and uncertainty.

    Parameters:
    -----------
    input_seq: Tensor of shape (seq_len, channels, height, width)
    true_map: Tensor of shape (channels, height, width)
    pred_map: Tensor of shape (channels, height, width)
    uncertainty_map: Optional tensor of shape (channels, height, width)
    frame_idx: Optional integer to show a specific frame from the sequence
    """
    # Convert tensors to numpy arrays
    if isinstance(input_seq, torch.Tensor):
        input_seq = input_seq.detach().cpu().numpy()
    if isinstance(true_map, torch.Tensor):
        true_map = true_map.detach().cpu().numpy()
    if isinstance(pred_map, torch.Tensor):
        pred_map = pred_map.detach().cpu().numpy()
    if uncertainty_map is not None and isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.detach().cpu().numpy()

    # Determine number of plots
    n_plots = 3 if uncertainty_map is None else 4

    # If frame_idx is provided, show only that frame
    if frame_idx is not None:
        plt.figure(figsize=(12, 3))

        # Input density map
        plt.subplot(1, n_plots, 1)
        plt.imshow(input_seq[frame_idx, 0], cmap="viridis")
        plt.title(f"Input Frame {frame_idx}")
        plt.colorbar()
        plt.axis("off")

        # Ground truth
        plt.subplot(1, n_plots, 2)
        plt.imshow(true_map[0], cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.title("Ground Truth")
        plt.colorbar()
        plt.axis("off")

        # Prediction
        plt.subplot(1, n_plots, 3)
        plt.imshow(pred_map[0], cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.title("Prediction")
        plt.colorbar()
        plt.axis("off")

        # Uncertainty (if provided)
        if uncertainty_map is not None:
            plt.subplot(1, n_plots, 4)
            plt.imshow(uncertainty_map[0], cmap="Purples")
            plt.title("Uncertainty")
            plt.colorbar()
            plt.axis("off")

    # Otherwise, show each input frame in sequence
    else:
        seq_len = input_seq.shape[0]
        fig, axes = plt.subplots(seq_len, n_plots, figsize=(n_plots * 4, seq_len * 3))

        for t in range(seq_len):
            # Input density map
            if seq_len > 1:
                ax = axes[t, 0]
            else:
                ax = axes[0]

            im = ax.imshow(input_seq[t, 0], cmap="viridis")
            ax.set_title(f"Input Frame {t}")
            plt.colorbar(im, ax=ax)
            ax.axis("off")

            # Ground truth (same for all frames)
            if t == 0:
                if seq_len > 1:
                    ax = axes[t, 1]
                else:
                    ax = axes[1]

                im = ax.imshow(true_map[0], cmap="RdYlGn_r", vmin=0, vmax=1)
                ax.set_title("Ground Truth")
                plt.colorbar(im, ax=ax)
                ax.axis("off")
            elif seq_len > 1:
                axes[t, 1].axis("off")

            # Prediction (same for all frames)
            if t == 0:
                if seq_len > 1:
                    ax = axes[t, 2]
                else:
                    ax = axes[2]

                im = ax.imshow(pred_map[0], cmap="RdYlGn_r", vmin=0, vmax=1)
                ax.set_title("Prediction")
                plt.colorbar(im, ax=ax)
                ax.axis("off")
            elif seq_len > 1:
                axes[t, 2].axis("off")

            # Uncertainty (if provided)
            if uncertainty_map is not None:
                if t == 0:
                    if seq_len > 1:
                        ax = axes[t, 3]
                    else:
                        ax = axes[3]

                    im = ax.imshow(uncertainty_map[0], cmap="Purples")
                    ax.set_title("Uncertainty")
                    plt.colorbar(im, ax=ax)
                    ax.axis("off")
                elif seq_len > 1:
                    axes[t, 3].axis("off")

    plt.tight_layout()
    return plt.gcf()


def create_animation(input_seq, true_map, pred_map, uncertainty_map=None, interval=200):
    """
    Create an animation of the input sequence with ground truth and prediction.

    Parameters:
    -----------
    input_seq: Tensor of shape (seq_len, channels, height, width)
    true_map: Tensor of shape (channels, height, width)
    pred_map: Tensor of shape (channels, height, width)
    uncertainty_map: Optional tensor of shape (channels, height, width)
    interval: Time interval between frames in milliseconds
    """
    # Convert tensors to numpy arrays
    if isinstance(input_seq, torch.Tensor):
        input_seq = input_seq.detach().cpu().numpy()
    if isinstance(true_map, torch.Tensor):
        true_map = true_map.detach().cpu().numpy()
    if isinstance(pred_map, torch.Tensor):
        pred_map = pred_map.detach().cpu().numpy()
    if uncertainty_map is not None and isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.detach().cpu().numpy()

    seq_len = input_seq.shape[0]
    n_plots = 3 if uncertainty_map is None else 4

    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))

    # Initialize plots
    ims = []

    for t in range(seq_len):
        im_list = []

        # Input density map
        im = axes[0].imshow(input_seq[t, 0], cmap="viridis", animated=True)
        axes[0].set_title("Input Density Map")
        axes[0].axis("off")
        im_list.append(im)

        # Ground truth
        im = axes[1].imshow(true_map[0], cmap="RdYlGn_r", vmin=0, vmax=1, animated=True)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        im_list.append(im)

        # Prediction
        im = axes[2].imshow(pred_map[0], cmap="RdYlGn_r", vmin=0, vmax=1, animated=True)
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        im_list.append(im)

        # Uncertainty (if provided)
        if uncertainty_map is not None:
            im = axes[3].imshow(uncertainty_map[0], cmap="Purples", animated=True)
            axes[3].set_title("Uncertainty")
            axes[3].axis("off")
            im_list.append(im)

        ims.append(im_list)

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    plt.tight_layout()

    return ani


def load_model_for_inference(model_path, model_config, device=None):
    """
    Load a trained model for inference.

    Parameters:
    -----------
    model_path: Path to the saved model weights
    model_config: Dictionary with model configuration parameters
    device: Device to load the model on (default: auto-detect)

    Returns:
    --------
    model: Loaded ConvLSTM model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with config
    model = ConvLSTM(
        in_channels=model_config.get("in_channels", 1),
        hidden_channels=model_config.get("hidden_channels", [64, 128, 128]),
        kernel_size=model_config.get("kernel_size", 3),
        num_layers=model_config.get("num_layers", 3),
        dropout_rate=model_config.get("dropout_rate", 0.2),
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def predict_risk_map(model, input_sequence, with_uncertainty=True, num_samples=20):
    """
    Generate risk map prediction from a single input sequence.

    Parameters:
    -----------
    model: Trained ConvLSTM model
    input_sequence: Tensor of shape (seq_len, channels, height, width)
                   or (batch, seq_len, channels, height, width)
    with_uncertainty: Whether to compute uncertainty using MC dropout
    num_samples: Number of MC samples for uncertainty estimation

    Returns:
    --------
    prediction: Risk map prediction
    uncertainty: (Optional) Uncertainty map
    """
    # Ensure input has batch dimension
    if input_sequence.dim() == 4:
        input_sequence = input_sequence.unsqueeze(0)

    # Move to same device as model
    device = next(model.parameters()).device
    input_sequence = input_sequence.to(device)

    if with_uncertainty:
        # Use MC dropout for uncertainty estimation
        mean_pred, uncertainty = model.predict_with_uncertainty(
            input_sequence, num_samples=num_samples
        )
        return mean_pred.cpu(), uncertainty.cpu()
    else:
        # Standard forward pass
        model.eval()
        with torch.no_grad():
            prediction = model(input_sequence)
        return prediction.cpu(), None


def risk_level_to_color(risk_map, uncertainty_map=None, risk_threshold=0.5):
    """
    Convert risk map and uncertainty into a color-coded visualization.

    Parameters:
    -----------
    risk_map: Tensor or array of shape (channels, height, width)
    uncertainty_map: Optional tensor or array of shape (channels, height, width)
    risk_threshold: Threshold for binary risk classification

    Returns:
    --------
    rgb_image: Color-coded risk visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(risk_map, torch.Tensor):
        risk_map = risk_map.detach().cpu().numpy()
    if uncertainty_map is not None and isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.detach().cpu().numpy()

    # Extract dimensions
    height, width = risk_map.shape[-2:]

    # Create RGB image
    rgb_image = np.zeros((height, width, 3))

    # Set colors based on risk level:
    # - Green: low risk
    # - Yellow: medium risk
    # - Red: high risk

    # Normalize risk map to [0,1]
    risk = risk_map[0]

    # Red channel - increases with risk
    rgb_image[:, :, 0] = risk

    # Green channel - decreases with risk
    rgb_image[:, :, 1] = 1 - risk

    # Blue channel - zero
    rgb_image[:, :, 2] = 0

    # If uncertainty is provided, use it to adjust alpha/transparency
    if uncertainty_map is not None:
        # Create RGBA image
        rgba_image = np.zeros((height, width, 4))
        rgba_image[:, :, 0:3] = rgb_image

        # Scale uncertainty to [0,1] for alpha channel
        uncertainty = uncertainty_map[0]
        max_uncertainty = np.max(uncertainty)
        if max_uncertainty > 0:
            uncertainty = uncertainty / max_uncertainty

        # More certain = more opaque
        rgba_image[:, :, 3] = 1 - uncertainty

        return rgba_image

    return rgb_image


def visualize_future_predictions(
    input_sequence, future_predictions, uncertainties, output_path=None
):
    """
    Visualize future predictions with uncertainty

    Parameters:
    ----------
    input_sequence: Input sequence tensor (batch, seq_len, channels, height, width)
    future_predictions: Predicted future frames (batch, future_steps, 1, height, width)
    uncertainties: Uncertainty for each prediction (batch, future_steps, 1, height, width)
    output_path: Path to save visualization
    """
    # Use only first sample in batch
    input_seq = input_sequence[0].cpu().numpy()
    predictions = future_predictions[0].cpu().numpy()
    uncertainty = uncertainties[0].cpu().numpy()

    # Get dimensions
    future_steps = predictions.shape[0]

    # Create figure
    fig, axes = plt.subplots(3, future_steps + 1, figsize=(4 * (future_steps + 1), 12))

    # Plot last input frame
    last_input = input_seq[-1, 0]
    axes[0, 0].imshow(last_input, cmap="viridis")
    axes[0, 0].set_title("Last Input Frame")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")

    # Plot predictions and uncertainties
    for t in range(future_steps):
        # Plot prediction
        axes[0, t + 1].imshow(predictions[t, 0], cmap="plasma")
        axes[0, t + 1].set_title(f"Prediction t+{t+1}")
        axes[0, t + 1].axis("off")

        # Plot uncertainty
        axes[1, t + 1].imshow(uncertainty[t, 0], cmap="inferno")
        axes[1, t + 1].set_title(f"Uncertainty t+{t+1}")
        axes[1, t + 1].axis("off")

        # Plot prediction with uncertainty overlay
        # Higher uncertainty = more transparent prediction
        pred = predictions[t, 0]
        unc = uncertainty[t, 0]

        # Normalize uncertainty to [0,1] for alpha channel
        norm_unc = 1 - (unc / (unc.max() + 1e-8))

        axes[2, t + 1].imshow(pred, cmap="plasma")
        axes[2, t + 1].imshow(pred, cmap="plasma", alpha=norm_unc)
        axes[2, t + 1].set_title(f"Pred with Uncertainty t+{t+1}")
        axes[2, t + 1].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
