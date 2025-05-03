import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms
from tqdm import tqdm

import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
pet_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach PET directory
sys.path.append(pet_dir)
from PET_Repo.util import misc as utils
from PET_Repo.models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Process Video for ConvLSTM using PET", add_help=False
    )

    # model parameters
    parser.add_argument(
        "--backbone",
        default="vgg16_bn",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned", "fourier"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--dec_layers",
        default=2,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=512,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.0, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )

    # loss parameters
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_point",
        default=0.05,
        type=float,
        help="SmoothL1 point coefficient in the matching cost",
    )
    parser.add_argument("--ce_loss_coef", default=1.0, type=float)
    parser.add_argument("--point_loss_coef", default=5.0, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.5,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="SHA")
    parser.add_argument("--data_path", default="./data/ShanghaiTech/PartA", type=str)

    # video processing parameters
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input video file or directory of image frames",
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="Directory to save density maps"
    )
    parser.add_argument(
        "--frame_step", default=1, type=int, help="Process every Nth frame of the video"
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=str,
        help="Resize frames to this size (WxH) before processing",
    )
    parser.add_argument(
        "--density_map_size",
        default=None,
        type=str,
        help="Resize density maps to this size (WxH)",
    )

    # visualization parameters
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save visualizations of density maps and predictions",
    )

    # model parameters
    parser.add_argument("--device", default="cuda", help="device to use for processing")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the pretrained PET model checkpoint",
    )
    parser.add_argument("--num_workers", default=2, type=int)

    return parser


def create_density_map(pred_points, img_h, img_w, kernel_size=15, sigma=4):
    """
    Create a density map from predicted points
    """
    density_map = np.zeros((img_h, img_w), dtype=np.float32)

    if len(pred_points) == 0:
        return density_map

    # Create gaussian kernel
    h, w = kernel_size, kernel_size
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    d = np.sqrt(x * x + y * y)
    sigma = sigma
    g = np.exp(-((d**2) / (2.0 * sigma**2)))

    # Normalize kernel
    g = g / g.sum()

    # Place gaussians at each head position
    for point in pred_points:
        y, x = int(point[0]), int(point[1])
        if 0 <= y < img_h and 0 <= x < img_w:
            # Calculate bounds for kernel placement
            x1, y1 = max(0, x - kernel_size // 2), max(0, y - kernel_size // 2)
            x2, y2 = min(img_w, x + kernel_size // 2 + 1), min(
                img_h, y + kernel_size // 2 + 1
            )

            # Calculate kernel bounds
            kx1, ky1 = max(0, kernel_size // 2 - x), max(0, kernel_size // 2 - y)
            kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)

            density_map[y1:y2, x1:x2] += g[ky1:ky2, kx1:kx2]

    return density_map


@torch.no_grad()
def process_image(model, image, device, density_map_size=None):
    """
    Process a single image with the PET model
    """
    # Convert image to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Transform image
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img = transform(image)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    # Run inference
    outputs = model(samples, test=True)
    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
    outputs_points = outputs["pred_points"][0]

    # Create points list
    points = [[point[0] * img_h, point[1] * img_w] for point in outputs_points]

    # Create density map
    density_map = create_density_map(points, img_h, img_w)

    # Resize density map if needed
    if density_map_size is not None:
        target_w, target_h = map(int, density_map_size.split("x"))
        density_map = cv2.resize(density_map, (target_w, target_h))

    return density_map, points, outputs_scores


def extract_frames_from_video(video_path, output_dir, frame_step=1, resize=None):
    """
    Extract frames from a video file
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video has {frame_count} frames at {fps} FPS")
    print(f"Extracting every {frame_step} frame")

    # Process frames
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # Resize frame if needed
            if resize is not None:
                target_w, target_h = map(int, resize.split("x"))
                frame = cv2.resize(frame, (target_w, target_h))

            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")

    return saved_count


def process_video(args):
    """
    Process a video or image sequence with PET model
    """
    # Set up device
    device = torch.device(args.device)

    # Build model
    model, _ = build_model(args)
    model.to(device)

    # Load pretrained model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create output directories
    density_maps_dir = os.path.join(args.output_dir, "density_maps")
    os.makedirs(density_maps_dir, exist_ok=True)

    if args.save_visualization:
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

    # Process video or image sequence
    if os.path.isfile(args.input) and args.input.endswith((".mp4", ".avi", ".mov")):
        # For video file, extract frames first
        frames_dir = os.path.join(args.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        extract_frames_from_video(args.input, frames_dir, args.frame_step, args.resize)
        image_paths = sorted(
            [
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith((".jpg", ".png"))
            ]
        )
    elif os.path.isdir(args.input):
        # For directory of images
        image_paths = sorted(
            [
                os.path.join(args.input, f)
                for f in os.listdir(args.input)
                if f.endswith((".jpg", ".png"))
            ]
        )
        image_paths = image_paths[
            :: args.frame_step
        ]  # Apply frame_step to image sequence
    else:
        raise ValueError(
            f"Input {args.input} is neither a video file nor a directory of images"
        )

    print(f"Processing {len(image_paths)} images...")

    # Process each image
    density_maps = []

    for i, img_path in enumerate(tqdm(image_paths)):
        # Load image
        img = cv2.imread(img_path)
        if args.resize is not None:
            target_w, target_h = map(int, args.resize.split("x"))
            img = cv2.resize(img, (target_w, target_h))

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Process image with PET
        density_map, points, _ = process_image(
            model, img_pil, device, args.density_map_size
        )

        # Save density map
        density_map_path = os.path.join(density_maps_dir, f"density_{i:06d}.npy")
        np.save(density_map_path, density_map)

        # Save visualization if requested
        if args.save_visualization:
            # Create heatmap visualization
            norm_density = (density_map / (density_map.max() + 1e-10) * 255).astype(
                np.uint8
            )
            heatmap = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)

            # Combine with original image
            vis_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.resize(vis_img, (heatmap.shape[1], heatmap.shape[0]))
            vis_img = cv2.addWeighted(vis_img, 0.6, heatmap, 0.4, 0)

            # Draw points
            for p in points:
                ratio_h = heatmap.shape[0] / img_pil.height
                ratio_w = heatmap.shape[1] / img_pil.width
                vis_img = cv2.circle(
                    vis_img,
                    (int(p[1] * ratio_w), int(p[0] * ratio_h)),
                    2,
                    (0, 255, 0),
                    -1,
                )

            # Save visualization
            vis_path = os.path.join(vis_dir, f"vis_{i:06d}.jpg")
            cv2.imwrite(vis_path, vis_img)

        # Save metadata
        density_maps.append(
            {
                "file": os.path.basename(img_path),
                "count": len(points),
                "density_map": density_map_path,
            }
        )

    # Save metadata
    import json

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "total_frames": len(density_maps),
                "density_map_dir": density_maps_dir,
                "frames": density_maps,
            },
            f,
            indent=2,
        )

    print(f"Processed {len(density_maps)} frames")
    print(f"Density maps saved to {density_maps_dir}")
    if args.save_visualization:
        print(f"Visualizations saved to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        "Process Video for ConvLSTM using PET", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
