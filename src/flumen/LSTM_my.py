import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    A custom LSTM implementation that mimics `torch.nn.LSTM` but does not use it.
    
    Features:
    - Manually implements LSTM cell operations (input, forget, cell, and output gates).
    - Supports multiple layers.
    - Allows bidirectional processing.
    - Handles hidden and cell state initialization dynamically.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        """
        Initializes the LSTM model.

        Parameters:
        - input_size (int): Number of expected input features per time step.
        - hidden_size (int): Number of hidden units per LSTM layer.
        - num_layers (int, optional): Number of stacked LSTM layers. Default is 1.
        - output_size (int, optional): Output feature size (if using an FC layer).
        - bias (bool, optional): Whether to include bias terms. Default is True.
        - batch_first (bool, optional): If True, input format is (batch, seq_len, features).
        - dropout (float, optional): Dropout rate between layers (ignored if num_layers=1).
        - bidirectional (bool, optional): Whether to use bidirectional LSTM.
        """
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        num_directions = 2 if bidirectional else 1  

        # Create LSTM layers manually instead of using torch.nn.LSTM
        self.lstm_cells = nn.ModuleList([
            nn.ModuleList([
                LSTMCell(input_size if layer == 0 else hidden_size * num_directions, 
                         hidden_size, bias)
            for _ in range(num_directions)])
            for layer in range(num_layers)
        ])

        # Optional fully connected output layer (used when output_size is given)
        if output_size is not None:
            self.fc = nn.Linear(hidden_size * num_directions, output_size)
        else:
            self.fc = None



    def forward(self, x, z, u=None, tau=None):      # def forward(self, x, hidden_state=None):
        """
        Forward pass through the custom LSTM.

        Parameters:
        - x (Tensor or PackedSequence): Input tensor of shape (batch, seq_len, input_size).
        - hidden_state (Tuple[Tensor, Tensor], optional): Tuple containing (h0, c0).

        Returns:
        - out (Tensor): Output tensor.
        - (hn, cn): Tuple containing the final hidden and cell states.

        ________________________________________________________________

        u: control input
        tau: time step factor
        """

        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        print("\nis_packed: ", is_packed)

        if is_packed:
            x_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
            print("\nx_unpacked.dim: ", x_unpacked.dim())
            if x_unpacked.dim() == 2: x_unpacked = x_unpacked.unsqueeze(1).expand(-1, 75, -1)
            print("\nx_unpacked.shape: ", x_unpacked.shape)
            batch_size, seq_len, _ = x_unpacked.shape
        else:
            print("\nx.shape: ", x.shape)
            batch_size, seq_len, _ = x.shape

        num_directions = 2 if self.bidirectional else 1
        print("\nnum_directions: ", num_directions)

        # Initialize hidden and cell states
        if z is not None:
            h0, c0 = z
        else:
            h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size, device=x.device)

        h = h0.clone()
        c = c0.clone()
        print("\nh.shape: ", h.shape)
        print("\nc.shape: ", c.shape)

        outputs = []

        # Iterate over time steps
        for t in range(seq_len):
            xt = x_unpacked[:, t, :] if is_packed else x[:, t, :]
            print("\nxt.shape: ", xt.shape)

            if isinstance(u, tuple): u = u[0]  
            if u.dim() == 2: u = u.unsqueeze(-1)

            ut = u[:, t, :] if u is not None else None

            tau_t = tau[:, t, :] if tau is not None else None

            for layer in range(self.num_layers):
                new_h, new_c = [], []

                # Forward direction (always present)
                # ht, ct = self.lstm_cells[layer][0](xt, h[layer][0], c[layer][0])

                if u is not None and tau is not None:
                    ht, ct = self.lstm_cells[layer][0](z, xt, ut, tau_t)
                else:  # OLD LSTM: Standard LSTM behavior
                    ht, ct = self.lstm_cells[layer][0](xt, h[layer][0], c[layer][0])


                new_h.append(ht)
                new_c.append(ct)

                if self.bidirectional:
                    """# Reverse direction (processes the sequence backwards)
                    xt_rev = x_unpacked[:, seq_len - t - 1, :] if is_packed else x[:, seq_len - t - 1, :]
                    ht_rev, ct_rev = self.lstm_cells[layer][1](xt_rev, h[layer][1], c[layer][1])
                    new_h.append(ht_rev)
                    new_c.append(ct_rev)

                    # Concatenate forward and backward outputs
                    xt = torch.cat([ht, ht_rev], dim=-1)"""

                    xt_rev = x[:, seq_len - t - 1, :]
                    if u is not None and tau is not None:
                        ht_rev, ct_rev = self.lstm_cells[layer][1](z, xt_rev, ut, tau_t)
                    else:
                        ht_rev, ct_rev = self.lstm_cells[layer][1](xt_rev, h[layer][1], c[layer][1])
                    new_h.append(ht_rev)
                    new_c.append(ct_rev)
                    xt = torch.cat([ht, ht_rev], dim=-1)
                else:
                    xt = ht

                # Replace hidden and cell state variables instead of modifying in place
                h_new = torch.stack(new_h).clone()
                c_new = torch.stack(new_c).clone()
                h = h_new
                c = c_new

            outputs.append(xt)

        outputs = torch.stack(outputs, dim=1 if self.batch_first else 0)

        if is_packed:
            # Re-pack the sequence if the input was originally packed
            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)

        # Apply fully connected layer if defined
        if self.fc is not None:
            outputs = self.fc(outputs[:, -1, :])  # Use last time step

        return outputs, (h, c)


class LSTMCell(nn.Module):
    """
    Custom implementation of a single LSTM cell.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        Initializes the LSTM cell.

        Parameters:
        - input_size (int): Number of expected input features.
        - hidden_size (int): Number of hidden units.
        - bias (bool, optional): Whether to use bias terms. Default is True.
        """
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.mhu = 1.5      # \mhu \in [0.1, 3.0] --- fromo wikipedia


        # Input-to-hidden weights and biases (used to process input data)
        self.W_i = nn.Linear(input_size, hidden_size, bias=bias)  # Input gate
        self.W_f = nn.Linear(input_size, hidden_size, bias=bias)  # Forget gate
        self.W_c = nn.Linear(input_size, hidden_size, bias=bias)  # Cell state candidate
        self.W_o = nn.Linear(input_size, hidden_size, bias=bias)  # Output gate

        # Hidden-to-hidden weights (used to process previous hidden state)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, z, x, u=None, tau=None):      # before | def forward(self, x, h, c):
        """
        Forward pass through the LSTM cell.

        Parameters:
        - x (Tensor): Input tensor of shape (batch, input_size).
        - h (Tensor): Previous hidden state of shape (batch, hidden_size).
        - c (Tensor): Previous cell state of shape (batch, hidden_size).

        Returns:
        - h_next (Tensor): Next hidden state.
        - c_next (Tensor): Next cell state.
        """

        if u is not None and tau is not None:
            A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]], device=x.device)

            print(f"x.shape: {x.shape}")
            print(f"tau.shape: {tau.shape}")
            print(f"A.shape: {A.shape}")

            x_next = x + tau * torch.matmul(A, x.T).T
        else:
            x_next = x  # OLD LSTM: No physics update


        # Compute LSTM gate activations
        i = torch.sigmoid(self.W_i(x) + self.U_i(z))  # Input gate
        f = torch.sigmoid(self.W_f(x) + self.U_f(z))  # Forget gate
        g = torch.tanh(self.W_c(x) + self.U_c(z))     # Cell state candidate
        o = torch.sigmoid(self.W_o(x) + self.U_o(z))  # Output gate

        # Update cell state (Avoid in-place modification)
        c_next = f * z + i * g      # before | z <-- c

        # Compute next hidden state
        h_next = o * torch.tanh(c_next)  

        if u is not None and tau is not None: 
            z_next = torch.car([x_next, h_next], dim=1)
        else:
            z_next = h_next

        return z_next, c_next       # before | z_next <-- h_next