# ConvLSTM for Crowd-Crush Prediction

## Overview

This is a PyTorch implementation of a ConvLSTM (Convolutional Long Short-Term Memory) network designed for crowd-crush prediction. The model processes sequences of 2D density maps and outputs a binary crowd-crush probability map with uncertainty estimation through Monte Carlo dropout.

## Architecture

The implementation consists of three main components:

1. **ConvLSTMCell**: A building block that combines convolutional operations with LSTM gating mechanisms to process spatio-temporal data.
2. **ConvLSTM**: A stacked model comprising multiple ConvLSTMCell layers followed by a 1×1 convolutional output layer.
3. **Uncertainty Estimation**: Using Monte Carlo dropout for estimating prediction confidence.

### Key Features

- Configurable architecture with variable number of layers and channels
- Mixed-precision training support for faster training on GPUs
- Uncertainty estimation via Monte Carlo dropout
- Visualization tools for interpreting predictions

## Files

- `model.py`: Core ConvLSTM implementation
- `train.py`: Training script with mixed-precision support
- `utils.py`: Utility functions for visualization and inference
- `example.py`: An example implementation of the model using synthetic data

## Usage

### Model Initialization

```python
from model import ConvLSTM

# Initialize model with 3 layers
model = ConvLSTM(
    in_channels=1,                     # Input channels (e.g., density maps)
    hidden_channels=[64, 128, 128],    # Hidden dimensions for each layer
    kernel_size=3,                     # Conv kernel size
    num_layers=3,                      # Number of ConvLSTM layers
    dropout_rate=0.2                   # Dropout probability
)
```

### Training

```python
import torch
from torch.utils.data import DataLoader
from train import train

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Train model with mixed precision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(model, train_loader, val_loader, num_epochs=50, device=device, use_mixed_precision=True)
```

### Inference with Uncertainty

```python
from utils import predict_risk_map, visualize_prediction

# Load input sequence (seq_len, channels, height, width)
input_sequence = torch.randn(10, 1, 64, 64)

# Get prediction with uncertainty
prediction, uncertainty = predict_risk_map(model, input_sequence, with_uncertainty=True, num_samples=20)

# Visualize results
visualize_prediction(input_sequence, true_map, prediction, uncertainty)
```

## Input Format

The model expects input sequences of shape `(batch, seq_len, channels, height, width)` where:
- `batch`: Batch size
- `seq_len`: Number of frames in the sequence
- `channels`: Number of input channels (typically 1 for density maps)
- `height`, `width`: Spatial dimensions of each frame

## Output Format

The model outputs:
- **Risk Map**: Tensor of shape `(batch, 1, height, width)` with values in range [0,1]
- **Uncertainty Map** (when using MC dropout): Tensor of same shape representing prediction uncertainty

## Requirements

- PyTorch >= 1.7.0
- NumPy
- Matplotlib (for visualization)

## References

This implementation is based on:
- Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
- Gal & Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" 