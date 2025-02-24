import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, state_dim=None, bias=True):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.bias = bias

        self.mhu = 1.5
        self.A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]])

        # Use TorchScript-optimized LSTM cells
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias)
            for layer in range(num_layers)
        ])

    def forward(self, rnn_input: PackedSequence, hidden_state, tau=None):
        """ Forward pass of the LSTM """
        rnn_input_data = rnn_input.data
        batch_sizes = rnn_input.batch_sizes  # Efficient packed sequence handling
        device = rnn_input_data.device

        h, c = hidden_state
        self.A = self.A.to(device)  # Ensure A is moved to the correct device

        seq_len = batch_sizes.size(0)
        outputs = torch.zeros(seq_len, h.size(1), self.hidden_size, device=device)

        for t in range(seq_len):
            batch_size_t = batch_sizes[t]  # Get batch size at timestep t
            rnn_input_t = rnn_input_data[:batch_size_t]  # Slice the correct batch elements

            x_prev = h[:, :, :self.state_dim]
            if tau is not None and t != 0:
                x_next = x_prev + tau * torch.matmul(x_prev, self.A)
                h[:, :, :self.state_dim] = x_next  # Directly modify `h` to avoid extra allocations

            for layer in range(self.num_layers):
                h[layer, :batch_size_t], c[layer, :batch_size_t] = self.lstm_cells[layer](
                    rnn_input_t, h[layer, :batch_size_t], c[layer, :batch_size_t]
                )

            outputs[t, :batch_size_t] = h[-1, :batch_size_t]

        packed_outputs = PackedSequence(outputs, batch_sizes, enforce_sorted=False)
        return packed_outputs, (h, c)


@torch.jit.script  # Compile with TorchScript for performance
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # One single matrix W for all 4 gates (input, forget, candidate, output)
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=bias)  # input → gates
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=False)  # hidden state → gates

    def forward(self, u: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """ Single LSTM Cell Step """
        gates = self.W(u) + self.U(h)  # Compute all gates in a single matrix multiplication

        # Split the tensor into four parts (input, forget, candidate, output gates)
        i, f, g, o = gates.chunk(4, dim=-1)  

        # Apply activation functions
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)      # Candidate cell state
        o = torch.sigmoid(o)  # Output gate

        # Update cell state and hidden state
        c_next = f * c + i * g  # Update memory cell
        h_next = o * torch.tanh(c_next)  # Compute new hidden state

        return h_next, c_next
