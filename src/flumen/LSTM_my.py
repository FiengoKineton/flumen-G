import torch
import torch.nn as nn

class LSTM_my(nn.Module):
    """
    Custom LSTM model that can be used as a drop-in replacement for torch.nn.LSTM.
    
    Features:
    - Supports variable input sizes.
    - Optionally includes a fully connected (FC) layer for classification tasks.
    - Supports multiple LSTM layers.
    - Allows bidirectional LSTMs.
    - Handles hidden and cell state initialization dynamically.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, output_size=None, 
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        """
        Initializes the LSTM model.

        Parameters:
        - input_size (int): Number of expected input features per time step.
        - hidden_size (int): Number of hidden units in each LSTM layer.
        - num_layers (int, optional): Number of stacked LSTM layers. Default is 1.
        - output_size (int, optional): Number of output features. If None, FC layer is not used.
        - bias (bool, optional): Whether to use bias terms in the LSTM. Default is True.
        - batch_first (bool, optional): If True, input is expected to have shape (batch, seq_len, features). Default is True.
        - dropout (float, optional): Dropout rate applied between layers (ignored if num_layers=1). Default is 0.0.
        - bidirectional (bool, optional): If True, uses a bidirectional LSTM. Default is False.
        """
        super(LSTM_my, self).__init__()

        # Store key parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        # Define LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,         # Input feature size per time step
            hidden_size=hidden_size,       # Number of hidden units per layer
            num_layers=num_layers,         # Number of stacked LSTM layers
            bias=bias,                     # Whether to use bias terms
            batch_first=batch_first,       # If True, input format should be (batch, seq_len, features)
            dropout=dropout,               # Dropout rate (only applies if num_layers > 1)
            bidirectional=bidirectional    # If True, LSTM is bidirectional
        )

        # Fully connected layer (Optional)
        if output_size is not None:
            # If bidirectional, output size needs to be doubled (as both directions concatenate outputs)
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        else:
            self.fc = None  # No FC layer if output_size is not provided

    def forward(self, x, h0=None, c0=None):
        """
        Forward pass through the LSTM.

        Parameters:
        - x (Tensor): Input tensor of shape (batch, seq_len, input_size) if batch_first=True, else (seq_len, batch, input_size).
        - h0 (Tensor, optional): Initial hidden state of shape (num_layers * num_directions, batch, hidden_size). Defaults to zeros.
        - c0 (Tensor, optional): Initial cell state of shape (num_layers * num_directions, batch, hidden_size). Defaults to zeros.

        Returns:
        - out (Tensor): Output tensor of shape (batch, seq_len, hidden_size) if batch_first=True.
                        If self.fc is defined, only the last time step is returned (batch, output_size).
        - (hn, cn): Tuple containing the final hidden state and cell state.
        """
        batch_size = x.size(0)  # Extract batch size from input tensor

        # Determine number of LSTM directions (1 for unidirectional, 2 for bidirectional)
        num_directions = 2 if self.bidirectional else 1

        # Initialize hidden state (h0) if not provided
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * num_directions,  # Adjust for multiple layers & bidirectional setting
                batch_size,                        # Batch size from input
                self.hidden_size,                  # Hidden layer size
                device=x.device                    # Ensure tensor is on the correct device (CPU/GPU)
            )

        # Initialize cell state (c0) if not provided
        if c0 is None:
            c0 = torch.zeros(
                self.num_layers * num_directions,  # Adjust for multiple layers & bidirectional setting
                batch_size,                        # Batch size from input
                self.hidden_size,                  # Hidden layer size
                device=x.device                    # Ensure tensor is on the correct device (CPU/GPU)
            )

        # Forward pass through the LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # If a fully connected layer is defined, apply it to the last time step's output
        if self.fc is not None:
            out = self.fc(out[:, -1, :])  # Extract last time step's output from LSTM sequence

        return out, (hn, cn)  # Return output and final hidden/cell states
