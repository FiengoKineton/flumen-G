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
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional


        # Create LSTM layers manually instead of using torch.nn.LSTM
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
            for layer in range(num_layers)
        ])

        # Optional fully connected output layer (used when output_size is given)
        self.fc = nn.Linear(hidden_size, output_size) if output_size is not None else None



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


        before:
          - x correspond to rnn_input
          - hidden_state correspond to (h0_stack, c0)
        
        now:
          - x correspond to rnn_input_packed
          - z correspond to (h0_stack, c0)
          - u correspond to rnn_input   
          - tau correspond to tau
        """

        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        #print("\nis_packed: ", is_packed)       # output | True

        if is_packed:
            x_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
            #print("\nlengths.shape: ", lengths.shape[0])        # output | 128 (=batch_size)   
            #print("\nx_unpacked.dim: ", x_unpacked.dim())       # output | 2 (==tensor with 3 elements)

            #if x_unpacked.dim() == 2: 
             #   seq_len = x_unpacked.shape[1] if len(lengths) > 0 else 1  
             #   x_unpacked = x_unpacked.unsqueeze(1).expand(-1, seq_len, -1)
            
            #print("\nx_unpacked.shape: ", x_unpacked.shape)     # output | torch.Size([128, 2, 2])
            batch_size, seq_len, _ = x_unpacked.shape
        else:
            #print("\nx.shape: ", x.shape)
            batch_size, seq_len, _ = x.shape

        # Initialize hidden and cell states
        if z is not None:
            h0, c0 = z
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)


        h = h0.clone()      #print("\nh.shape: ", h.shape)       # output | torch.Size([1, 128, 14])
        c = c0.clone()      #print("\nc.shape: ", c.shape)       # output | torch.Size([1, 128, 14])
        outputs = []


        # Iterate over time steps
        for t in range(seq_len):
            xt = x_unpacked[:, t, :] if is_packed else x[:, t, :]
            #print("\nxt.shape: ", xt.shape)     # output | torch.Size([128, 2])

            #if isinstance(u, tuple): u = u[0]  
            #if u.dim() == 2: u = u.unsqueeze(-1)

            ut = u[:, t] if u is not None else None
            tau_t = tau if tau is not None else None

            """
            print("\nxt.shape: ", xt.shape)        # output | torch.Size([128, 2])
            print("\nut.shape: ", ut.shape)        # output | torch.Size([128])      
            print("\ntau_t.shape: ", tau_t.shape)  # output | torch.Size([128, 2])
            #"""

            for layer in range(self.num_layers):
                new_h, new_c = [], []
                zt, ct = self.lstm_cells[layer](xt, h[layer], c[layer], ut, tau_t)  
                xt = zt[:, :2]
                ht = zt[:, 2:]    

                #ht, ct = self.lstm_cells[layer](z, xt, ut, tau_t)

            #-- if u is not None and tau is not None:   ht, ct = self.lstm_cells[layer][0](z, xt, ut, tau_t)                # NEW LSTM: Physics-based LSTM
            #-- else:                                   ht, ct = self.lstm_cells[layer][0](xt, h[layer][0], c[layer][0])    # OLD LSTM: Standard LSTM behavior

                # Replace hidden and cell state variables instead of modifying in place
                new_h.append(ht)
                new_c.append(ct)
                #xt = ht
                h_new = torch.stack(new_h).clone()
                c_new = torch.stack(new_c).clone()
                h = h_new
                c = c_new

            outputs.append(xt)

        outputs = torch.stack(outputs, dim=1)

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
        self.W_i = nn.Linear(input_size, hidden_size, bias=bias)    # Input gate
        self.W_f = nn.Linear(input_size, hidden_size, bias=bias)    # Forget gate
        self.W_c = nn.Linear(input_size, hidden_size, bias=bias)    # Cell state candidate
        self.W_o = nn.Linear(input_size, hidden_size, bias=bias)    # Output gate

        # Hidden-to-hidden weights (used to process previous hidden state)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
    

        """
    #-- Weights shapes
        print("\nWEIGHTS SHAPES\n")

        print(f"\nW_i weight shape: {self.W_i.weight.shape}")       # output | torch.Size([14, 2])
        print(f"\nW_f weight shape: {self.W_f.weight.shape}")       # output | torch.Size([14, 2])
        print(f"\nW_c weight shape: {self.W_c.weight.shape}")       # output | torch.Size([14, 2])
        print(f"\nW_o weight shape: {self.W_o.weight.shape}")       # output | torch.Size([14, 2])
        print("\n")

        print(f"\nU_i weight shape: {self.U_i.weight.shape}")       # output | torch.Size([14, 14])
        print(f"\nU_f weight shape: {self.U_f.weight.shape}")       # output | torch.Size([14, 14])
        print(f"\nU_c weight shape: {self.U_c.weight.shape}")       # output | torch.Size([14, 14])
        print(f"\nU_o weight shape: {self.U_o.weight.shape}")       # output | torch.Size([14, 14])
        print("\n\n")
        #"""


    def forward(self, x, z, c, u=None, tau=None):      # before | def forward(self, x, h, c):
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
            x_next = x + tau * torch.matmul(A, x.T).T
        else:
            x_next = x  # OLD LSTM: No physics update


        """
    #-- Initial Variable shapes
        print("\ninit VARIABLE SHAPES\n")

        print(f"x.shape: {x.shape}")            # output | torch.Size([128, 2])
        print(f"x_next.shape: {x_next.shape}")  # output | torch.Size([128, 2])
        print(f"z.shape: {z.shape}")            # output | torch.Size([128, 14])
        print(f"u.shape: {u.shape}")            # output | torch.Size([128])
        print(f"tau.shape: {tau.shape}")        # output | torch.Size([128, 2])
        print(f"A.shape: {A.shape}")            # output | torch.Size([2, 2]) 
        print("\n\n")
        #"""


        # Compute LSTM gate activations
        i = torch.sigmoid(self.W_i(x) + self.U_i(z))  # Input gate
        f = torch.sigmoid(self.W_f(x) + self.U_f(z))  # Forget gate
        g = torch.tanh(self.W_c(x) + self.U_c(z))     # Cell state candidate
        o = torch.sigmoid(self.W_o(x) + self.U_o(z))  # Output gate

        # Update cell state (Avoid in-place modification)
        c_next = f * z + i * g      # before | z <-- c  # Next cell state

        # Compute next hidden state
        h_next = o * torch.tanh(c_next)                 # Next hidden state

        if u is not None and tau is not None: 
            if z.shape[1] == self.hidden_size: 
                z_next = torch.cat([x_next, h_next], dim=1) 
            else:
                z_next = torch.cat([x_next, h_next[:, 2:]], dim=1)
        else:
            z_next = h_next


        #"""
    #-- Final Variable shapes
        print("\nfinal VARIABLE SHAPES\n")

        print(f"i.shape: {i.shape}")            # output | torch.Size([128, 14])
        print(f"g.shape: {g.shape}")            # output | torch.Size([128, 14])
        print(f"f.shape: {f.shape}")            # output | torch.Size([128, 14])
        print(f"o.shape: {o.shape}")            # output | torch.Size([128, 14])
        print(f"c_next.shape: {c_next.shape}")  # output | torch.Size([128, 14])
        print(f"h_next.shape: {h_next.shape}")  # output | torch.Size([128, 14])
        print("\n")

        print(f"z_next.shape: {z_next.shape}")  # output | torch.Size([128, 16])
        print("\n\n")
        #"""

        return z_next, c_next       # before | z_next <-- h_next