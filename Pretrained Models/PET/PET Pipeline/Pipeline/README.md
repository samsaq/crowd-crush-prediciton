# Crowd Crush Prediction Pipeline

This folder contains a pipeline for crowd crush prediction using two main components:

1. **PET (Point Estimation Transformer)** - Used for crowd counting and density map generation
2. **ConvLSTM** - Used for crowd movement analysis and risk prediction

## Overview

The pipeline processes videos or image sequences in three main steps:

1. Generate density maps from input frames using PET
2. Prepare sequences of density maps for ConvLSTM input
3. Run ConvLSTM to predict risk maps with uncertainty estimates

## Requirements

See the overall repo's requirements.txt file

## Usage

### Step 1: Generate Density Maps with PET

Process a video or image sequence to generate density maps:

```bash
python PET-main/process_video_for_convlstm.py \
    --input path/to/video/or/image/sequence \
    --output_dir output/pet_results \
    --checkpoint "Pretrained Models/PET/SHB_model.pth" \
    --save_visualization \
    --resize 640x480 \
    --frame_step 5
```

Arguments:
- `--input`: Path to input video file or directory containing image frames
- `--output_dir`: Directory to save density maps and visualizations
- `--checkpoint`: Path to the pretrained PET model
- `--save_visualization`: Flag to save visualizations
- `--resize`: Optional resizing of input frames (WxH)
- `--frame_step`: Process every Nth frame (useful for long videos)
- `--density_map_size`: Optional resizing of output density maps (WxH)

Output:
- Density maps saved as .npy files
- Visualizations of density maps
- Metadata file with frame information

### Step 2: Prepare ConvLSTM Input

Prepare sequences of density maps as input for ConvLSTM:

```bash
python prepare_convlstm_input.py \
    --input_dir output/pet_results \
    --output_dir output/convlstm_input \
    --sequence_length 10 \
    --stride 1 \
    --normalize
```

Arguments:
- `--input_dir`: Directory containing PET density maps
- `--output_dir`: Directory to save prepared data for ConvLSTM
- `--sequence_length`: Number of frames in each sequence
- `--stride`: Frame stride for creating sequences
- `--resize`: Optional resizing of density maps (WxH)
- `--normalize`: Flag to normalize each density map to [0,1]

Output:
- PyTorch tensor file with sequences
- Metadata file with sequence information

### Step 3: Run ConvLSTM for Risk Prediction

Process the sequences with ConvLSTM to generate risk predictions:

```bash
python run_convlstm_prediction.py \
    --input output/convlstm_input/convlstm_input.pt \
    --output_dir output/predictions \
    --model_config medium \
    --with_uncertainty \
    --visualize
```

Arguments:
- `--input`: Path to the ConvLSTM input tensor file
- `--output_dir`: Directory to save prediction results
- `--model_config`: ConvLSTM configuration (small, medium, large, default)
- `--with_uncertainty`: Enable uncertainty estimation
- `--num_samples`: Number of samples for uncertainty estimation
- `--visualize`: Generate visualizations of predictions

Output:
- Predictions saved as .npy files
- Uncertainty estimates (if enabled)
- Visualizations of predictions and risk maps

## Example

Here's a complete example workflow for processing a video:

```bash
# Step 1: Process video with PET to generate density maps
python PET-main/process_video_for_convlstm.py \
    --input videos/crowd_video.mp4 \
    --output_dir output/pet_results \
    --checkpoint "Pretrained Models/PET/SHB_model.pth" \
    --save_visualization \
    --resize 640x480 \
    --frame_step 5

# Step 2: Prepare ConvLSTM input
python prepare_convlstm_input.py \
    --input_dir output/pet_results \
    --output_dir output/convlstm_input \
    --sequence_length 10 \
    --stride 1 \
    --normalize

# Step 3: Run ConvLSTM for prediction
python run_convlstm_prediction.py \
    --input output/convlstm_input/convlstm_input.pt \
    --output_dir output/predictions \
    --model_config medium \
    --with_uncertainty \
    --visualize
```

## Notes

- The PET model is used for generating density maps from single images from a video.
- The ConvLSTM model takes sequences of density maps to predict future crowd movement and potential risk areas.
- The uncertainty estimation in ConvLSTM helps identify areas where the model is less confident in its predictions.

## Credits

- PET (Point Estimation Transformer): [Original Repository](https://github.com/Shimmer93/PET)
- ConvLSTM implementation based on ["Convolutional LSTM Networks for Spatio-Temporal Forecasting"](https://arxiv.org/abs/1506.04214)
