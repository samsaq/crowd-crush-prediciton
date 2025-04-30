import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)  # concat along channel axis
        conv_output = self.conv(combined)
        (cc_i, cc_f, cc_o, cc_g) = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, kernel_size, num_layers, dropout_rate=0.2
    ):
        super().__init__()
        self.num_layers = num_layers

        # Ensure hidden_channels is a list with length = num_layers
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        assert (
            len(hidden_channels) == num_layers
        ), "Length of hidden_channels must equal num_layers"

        # Ensure kernel_size is a list with length = num_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        assert (
            len(kernel_size) == num_layers
        ), "Length of kernel_size must equal num_layers"

        self.hidden_channels = hidden_channels

        # Create layers
        layers = []
        input_dim = in_channels
        for i in range(num_layers):
            cell = ConvLSTMCell(input_dim, hidden_channels[i], kernel_size[i])
            layers.append(cell)
            input_dim = hidden_channels[i]

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_out = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    def forward(self, x_seq, apply_sigmoid=True):
        """
        Parameters:
        ----------
        x_seq: 5D Tensor of shape (batch, seq_len, channels, height, width)
        apply_sigmoid: Boolean to determine if sigmoid should be applied to output

        Returns:
        -------
        out: 4D Tensor of shape (batch, 1, height, width)
        """
        batch_size, seq_len, _, height, width = x_seq.size()
        device = x_seq.device

        # Initialize hidden state and cell state for each layer
        h_states = []
        c_states = []

        for i in range(self.num_layers):
            h_states.append(
                torch.zeros(
                    batch_size, self.hidden_channels[i], height, width, device=device
                )
            )
            c_states.append(
                torch.zeros(
                    batch_size, self.hidden_channels[i], height, width, device=device
                )
            )

        # Process input sequence
        for t in range(seq_len):
            x = x_seq[:, t]

            for layer_idx, layer in enumerate(self.layers):
                h_states[layer_idx], c_states[layer_idx] = layer(
                    x, h_states[layer_idx], c_states[layer_idx]
                )
                x = self.dropout(h_states[layer_idx])

        # Apply final convolution to the last hidden state of the last layer
        out = self.conv_out(h_states[-1])

        if apply_sigmoid:
            out = torch.sigmoid(out)

        return out

    def predict_with_uncertainty(self, x_seq, num_samples=10):
        """
        Use Monte Carlo dropout for uncertainty estimation

        Parameters:
        ----------
        x_seq: 5D Tensor of shape (batch, seq_len, channels, height, width)
        num_samples: Number of forward passes to perform

        Returns:
        -------
        mean_prediction: Mean prediction across samples
        uncertainty: Standard deviation across samples
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            pred = self.forward(x_seq)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_prediction, uncertainty

    def predict_future_with_uncertainty(self, x_seq, future_steps=5, num_samples=10):
        """
        Predict future frames with uncertainty estimation using Monte Carlo dropout

        Parameters:
        ----------
        x_seq: 5D Tensor of shape (batch, seq_len, channels, height, width)
        future_steps: Number of future frames to predict
        num_samples: Number of Monte Carlo samples for uncertainty estimation

        Returns:
        -------
        predictions: Tensor of shape (batch, future_steps, 1, height, width) - mean predictions
        uncertainties: Tensor of shape (batch, future_steps, 1, height, width) - uncertainty estimates
        """
        self.train()  # Enable dropout for MC sampling
        batch_size, seq_len, channels, height, width = x_seq.size()
        device = x_seq.device

        # Initialize storage for all predictions and current input sequence
        all_sample_predictions = []

        # For each MC sample
        for _ in range(num_samples):
            current_sequence = x_seq.clone()
            future_sequence = []

            # Predict future frames autoregressively
            for step in range(future_steps):
                # Get prediction for next frame
                next_frame_pred = self.forward(
                    current_sequence
                )  # Shape: (batch, 1, height, width)

                # Store prediction
                future_sequence.append(next_frame_pred)

                # Update sequence for next prediction (remove oldest frame, add new prediction)
                # Reshape prediction to match expected input format
                next_frame_pred_expanded = next_frame_pred.unsqueeze(
                    1
                )  # Shape: (batch, 1, 1, height, width)
                # Create channels dimension if needed
                if channels > 1:
                    next_frame_pred_expanded = next_frame_pred_expanded.repeat(
                        1, 1, channels, 1, 1
                    )

                # Remove oldest frame and append new prediction
                current_sequence = torch.cat(
                    [current_sequence[:, 1:], next_frame_pred_expanded], dim=1
                )

            # Stack all future predictions for this sample
            future_preds = torch.stack(
                future_sequence, dim=1
            )  # (batch, future_steps, 1, height, width)
            all_sample_predictions.append(future_preds)

        # Stack all samples
        all_samples = torch.stack(
            all_sample_predictions, dim=0
        )  # (num_samples, batch, future_steps, 1, height, width)

        # Calculate mean and standard deviation across samples
        mean_predictions = torch.mean(
            all_samples, dim=0
        )  # (batch, future_steps, 1, height, width)
        uncertainties = torch.std(
            all_samples, dim=0
        )  # (batch, future_steps, 1, height, width)

        return mean_predictions, uncertainties
