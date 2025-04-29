import argparse
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import inspect
from pathlib import Path


# Add ConvLSTM directory to path
def find_project_root():
    # Get the directory of the current file
    current_file_dir = Path(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    )

    # Navigate up to find the project root (where ConvLSTM directory exists)
    potential_root = current_file_dir
    while potential_root != potential_root.parent:
        if (potential_root / "ConvLSTM").exists():
            return str(potential_root)
        potential_root = potential_root.parent

    # If not found, try relative to current working directory
    potential_root = Path(os.getcwd())
    while potential_root != potential_root.parent:
        if (potential_root / "ConvLSTM").exists():
            return str(potential_root)
        potential_root = potential_root.parent

    raise FileNotFoundError("Could not find ConvLSTM directory in project structure.")


# Add the ConvLSTM directory to sys.path
project_root = find_project_root()
convlstm_path = os.path.join(project_root, "ConvLSTM")
sys.path.append(convlstm_path)
print(f"Using ConvLSTM directory: {convlstm_path}")

# Import ConvLSTM model and utils
from ConvLSTM.model import ConvLSTM
from ConvLSTM.utils import visualize_prediction, risk_level_to_color


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ConvLSTM prediction on PET density maps"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the ConvLSTM input tensor file (from prepare_convlstm_input.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prediction results",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="default",
        help='ConvLSTM model configuration: "small", "medium", "large", or "default"',
    )
    parser.add_argument(
        "--with_uncertainty",
        action="store_true",
        help="Enable uncertainty estimation using Monte Carlo dropout",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for uncertainty estimation",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of predictions",
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


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    print(f"Loading input data from {args.input}")
    input_tensor = torch.load(args.input)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Determine if metadata is available
    metadata_path = os.path.join(os.path.dirname(args.input), "sequence_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata with {metadata['num_sequences']} sequences")
    else:
        metadata = None
        print("No metadata found")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    num_sequences, seq_len, channels, height, width = input_tensor.shape
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

    # Make predictions
    print("Running predictions...")
    results = []

    with torch.no_grad():
        for i in tqdm(range(num_sequences)):
            # Get sequence
            sequence = input_tensor[i : i + 1].to(device)

            # Run prediction with or without uncertainty
            if args.with_uncertainty:
                mean_pred, uncertainty = model.predict_with_uncertainty(
                    sequence, num_samples=args.num_samples
                )
                results.append(
                    {
                        "sequence_id": i,
                        "prediction": mean_pred.cpu().numpy(),
                        "uncertainty": uncertainty.cpu().numpy(),
                    }
                )
            else:
                prediction = model(sequence)
                results.append(
                    {
                        "sequence_id": i,
                        "prediction": prediction.cpu().numpy(),
                        "uncertainty": None,
                    }
                )

    # Save results
    print("Saving results...")

    # Save raw predictions
    predictions = np.array([r["prediction"] for r in results])
    predictions_path = os.path.join(args.output_dir, "predictions.npy")
    np.save(predictions_path, predictions)
    print(f"Saved predictions to {predictions_path}")

    if args.with_uncertainty:
        uncertainties = np.array([r["uncertainty"] for r in results])
        uncertainties_path = os.path.join(args.output_dir, "uncertainties.npy")
        np.save(uncertainties_path, uncertainties)
        print(f"Saved uncertainties to {uncertainties_path}")

    # Create visualizations if requested
    if args.visualize:
        print("Creating visualizations...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for i, result in enumerate(tqdm(results)):
            # Get prediction and input sequence
            sequence = input_tensor[i].cpu().numpy()
            prediction = result["prediction"][0]  # Remove batch dimension
            uncertainty = (
                result["uncertainty"][0] if result["uncertainty"] is not None else None
            )

            # Example frame from the sequence (use last frame)
            input_frame = sequence[-1, 0]  # Last frame, first channel

            # Create visualization
            plt.figure(figsize=(15, 5))

            # Plot input frame
            plt.subplot(1, 3, 1)
            plt.imshow(input_frame, cmap="viridis")
            plt.title("Input Density Map")
            plt.axis("off")

            # Plot prediction
            plt.subplot(1, 3, 2)
            plt.imshow(prediction[0], cmap="plasma")  # Remove channel dimension
            plt.title("Predicted Risk Map")
            plt.axis("off")

            # Plot uncertainty or risk color map
            plt.subplot(1, 3, 3)
            if uncertainty is not None:
                # Use the risk_level_to_color function to create a color-coded risk map
                risk_map = risk_level_to_color(
                    torch.from_numpy(prediction), torch.from_numpy(uncertainty)
                )
                plt.imshow(risk_map)
                plt.title("Risk Map with Uncertainty")
            else:
                # Just show a heatmap of the prediction
                plt.imshow(prediction[0], cmap="hot")
                plt.title("Risk Heatmap")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"sequence_{i:04d}.png"), dpi=150)
            plt.close()

        print(f"Saved visualizations to {vis_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
