import argparse
import os
import numpy as np
import torch
import json
from tqdm import tqdm
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ConvLSTM input from PET density maps"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing PET density maps (output from process_video_for_convlstm.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed data for ConvLSTM",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10,
        help="Number of frames to include in each sequence for ConvLSTM",
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Frame stride for creating sequences"
    )
    parser.add_argument(
        "--resize",
        type=str,
        default=None,
        help="Resize density maps to this size (WxH)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each density map to [0, 1] range",
    )

    return parser.parse_args()


def load_metadata(input_dir):
    """
    Load metadata file from PET density map directory
    """
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def create_sequences(density_maps, sequence_length, stride):
    """
    Create sequences of density maps for ConvLSTM
    """
    sequences = []

    for i in range(0, len(density_maps) - sequence_length + 1, stride):
        sequences.append(density_maps[i : i + sequence_length])

    return sequences


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    try:
        metadata = load_metadata(args.input_dir)
        density_map_dir = metadata["density_map_dir"]
        frames = metadata["frames"]
        print(f"Found metadata with {len(frames)} frames")
    except (FileNotFoundError, KeyError):
        # If metadata is not available, try to find density maps directly
        print("Metadata not found, searching for density maps directly")
        density_map_dir = os.path.join(args.input_dir, "density_maps")
        if not os.path.exists(density_map_dir):
            density_map_dir = args.input_dir

        density_map_files = sorted(glob.glob(os.path.join(density_map_dir, "*.npy")))
        frames = [{"density_map": f} for f in density_map_files]
        print(f"Found {len(frames)} density map files")

    if not frames:
        raise ValueError("No density maps found")

    # Load all density maps
    print("Loading density maps...")
    density_maps = []

    for frame in tqdm(frames):
        map_path = frame["density_map"]
        if not os.path.isabs(map_path):
            map_path = os.path.join(args.input_dir, map_path)

        try:
            density_map = np.load(map_path)

            # Resize if requested
            if args.resize:
                import cv2

                target_w, target_h = map(int, args.resize.split("x"))
                density_map = cv2.resize(density_map, (target_w, target_h))

            # Normalize if requested
            if args.normalize and density_map.max() > 0:
                density_map = density_map / density_map.max()

            density_maps.append(density_map)
        except Exception as e:
            print(f"Error loading {map_path}: {e}")

    print(f"Loaded {len(density_maps)} density maps")

    # Create sequences for ConvLSTM
    print(
        f"Creating sequences with length {args.sequence_length} and stride {args.stride}..."
    )
    sequences = create_sequences(density_maps, args.sequence_length, args.stride)
    print(f"Created {len(sequences)} sequences")

    # Format and save data for ConvLSTM
    print("Preparing data for ConvLSTM...")

    # Stack sequences into a tensor of shape [num_sequences, sequence_length, 1, height, width]
    x_data = np.stack(
        [np.stack([np.expand_dims(dm, axis=0) for dm in seq]) for seq in sequences]
    )

    # Convert to torch tensor
    x_tensor = torch.from_numpy(x_data).float()

    # Save as torch tensor
    torch_path = os.path.join(args.output_dir, "convlstm_input.pt")
    torch.save(x_tensor, torch_path)
    print(f"Saved tensor of shape {x_tensor.shape} to {torch_path}")

    # Save metadata
    metadata_path = os.path.join(args.output_dir, "sequence_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "num_sequences": len(sequences),
                "sequence_length": args.sequence_length,
                "stride": args.stride,
                "tensor_shape": list(x_tensor.shape),
                "normalized": args.normalize,
            },
            f,
            indent=2,
        )

    print(f"Saved metadata to {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
