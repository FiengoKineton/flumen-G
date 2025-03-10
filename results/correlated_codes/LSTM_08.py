import torch, sys, time
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, z_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, 
                 state_dim=None, discretisation_mode=None, x_update_mode=None):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.z_size = z_size
        self.hidden_size = z_size - state_dim
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.state_dim = state_dim

        self.mhu = 1.5
        self.A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]])
        self.I = torch.eye(self.A.shape[0], dtype=self.A.dtype)

        function_name = f"discretisation_{discretisation_mode}"
        self.discretisation_function = globals().get(function_name)

        function_name = f"x_update_mode__{x_update_mode}"
        self.x_update_function = globals().get(function_name)

        self.lstm_cells = nn.ModuleList([
            torch.jit.script(LSTMCell(input_size+state_dim if layer==0 else self.hidden_size, self.hidden_size, bias))
            for layer in range(num_layers)
        ])

        self.alpha_gate = nn.Linear(self.hidden_size, self.state_dim, bias=bias)       # gate function to blend h into x
        self.W__h_to_x = nn.Linear(self.hidden_size, self.state_dim, bias=bias)        # mapping form h to x

    

    def forward(self, rnn_input: PackedSequence, hidden_state, tau):

        rnn_input_unpacked, lengths = pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        _, seq_len, _ = rnn_input_unpacked.shape      # output | batch_size, seq_len, input_size = torch.Size([512, 75, 2])
        device = rnn_input_unpacked.device

        z, c_z = hidden_state
        self.A = self.A.to(device)
        self.I = self.I.to(device)

        outputs = torch.zeros(z.size(1), seq_len, self.z_size, device=device)

        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :]       
            tau_t = tau[:, t, :] 

            x_prev = z[:, :, :self.state_dim]
            c_x = c_z[:, :, :self.state_dim]

            h = z[:, :, self.state_dim:]
            c = c_z[:, :, self.state_dim:]


            h_new_list, c_new_list = [], []     
            for layer in range(self.num_layers):
                u_t = torch.cat((x_prev.squeeze(0), rnn_input_t), dim=1)   
                h_new, c_new = self.lstm_cells[layer](u_t, h[layer], c[layer])

                h_new_list.append(h_new)        
                c_new_list.append(c_new)        


            x_mid = self.discretisation_function(x_prev, self.A, tau_t, self.I)     ###x_mid = discretisation_TU_old(x_prev, self.A, 0.05, self.I)
            h = torch.stack(h_new_list, dim=0)
            c = torch.stack(c_new_list, dim=0)

            x_next = self.x_update_function(x_mid, h, self.alpha_gate, self.W__h_to_x)

            z = torch.cat((x_next, h), dim=-1)  
            c_z = torch.cat((c_x, c), dim=-1)

            outputs[:, t, :] = z[-1]       
            ###sys.exit()

        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)
        return packed_outputs, (z, c_z)


def discretisation_FE(x_prev, A, tau, I):
    batch_size = tau.shape[0]  # tau is (batch, 1)
    state_dim = A.shape[0]  # A is (state_dim, state_dim)

    # Reshape tau for broadcasting
    tau = tau.view(batch_size, 1, 1)  # Shape: [batch, 1, 1]

    # Expand A and I to match batch size
    A = A.expand(batch_size, state_dim, state_dim)  # Shape: [batch, state_dim, state_dim]
    I = I.expand(batch_size, state_dim, state_dim)  # Shape: [batch, state_dim, state_dim]

    # Compute transformation matrix
    transform_matrix = I + tau * A  # Shape: [batch, state_dim, state_dim]

    # Reshape x_prev for batch matrix multiplication
    x_prev = x_prev.squeeze(0).unsqueeze(1)  # Convert (1, batch, state_dim) → (batch, 1, state_dim)

    # Apply transformation
    x_next = torch.bmm(x_prev, transform_matrix)  # Shape: [batch, 1, state_dim]

    # Restore the extra sequence dimension: (1, batch, state_dim)
    return x_next.permute(1, 0, 2)  # Shape: [1, batch, state_dim]

def discretisation_BE(x_prev, A, tau, I):
    batch_size = tau.shape[0]  # tau is (batch, 1)
    state_dim = A.shape[0]  # A is (state_dim, state_dim)

    # Reshape tau for broadcasting
    tau = tau.view(batch_size, 1, 1)  # Shape: [batch, 1, 1]

    # Expand A and I to match batch size
    A = A.expand(batch_size, state_dim, state_dim)  # Shape: [batch, state_dim, state_dim]
    I = I.expand(batch_size, state_dim, state_dim)  # Shape: [batch, state_dim, state_dim]

    # Compute inverse matrix
    inv_matrix = torch.inverse(I - tau * A)  # Shape: [batch, state_dim, state_dim]

    # Reshape x_prev for batch matrix multiplication
    x_prev = x_prev.squeeze(0).unsqueeze(1)  # Convert (1, batch, state_dim) → (batch, 1, state_dim)

    # Apply transformation
    x_next = torch.bmm(x_prev, inv_matrix)  # Shape: [batch, 1, state_dim]

    # Restore the extra sequence dimension: (1, batch, state_dim)
    return x_next.permute(1, 0, 2)  # Shape: [1, batch, state_dim]

def discretisation_TU(x_prev, A, tau, I):
    batch_size = tau.shape[0]  # tau is (128, 1)
    state_dim = A.shape[0]  # A is (2, 2)

    # Reshape tau for broadcasting (batch, 1, 1)
    tau = tau.view(batch_size, 1, 1)  # Shape: [128, 1, 1]

    # Expand A and I to batch dimension (batch, state_dim, state_dim)
    A = A.expand(batch_size, state_dim, state_dim)  # Shape: [128, 2, 2]
    I = I.expand(batch_size, state_dim, state_dim)  # Shape: [128, 2, 2]

    # Compute transformation matrices
    A_pos = I + (tau / 2) * A  # Shape: [128, 2, 2]
    A_neg = I - (tau / 2) * A  # Shape: [128, 2, 2]

    # Compute inverse: (batch, state_dim, state_dim)
    A_neg_inv = torch.inverse(A_neg)  # Shape: [128, 2, 2]

    # Compute transformation matrix
    transform_matrix = torch.bmm(A_pos, A_neg_inv)  # [128, 2, 2]

    # Reshape x_prev for batch matrix multiplication
    x_prev = x_prev.squeeze(0)  # Convert from (1, 128, 2) → (128, 2)
    x_prev = x_prev.unsqueeze(1)  # Now: [128, 1, 2]

    # Apply Tustin discretization
    x_next = torch.bmm(x_prev, transform_matrix)  # [128, 1, 2]

    # Restore the extra sequence dimension: (1, 128, 2)
    return x_next.permute(1, 0, 2)  # Reshape to [1, 128, 2]


def x_update_mode__alpha(x_mid, h, alpha_gate, W__h_to_x):
    alpha = torch.sigmoid(alpha_gate(h[-1]))
    return (1-alpha) * x_mid + alpha * W__h_to_x(h[-1])


def x_update_mode__lamda(x_mid, h, alpha_gate, W__h_to_x):
    eps = 1e-5
    x_norm = torch.norm(x_mid, dim=-1, keepdim=True) + eps
    h_norm = torch.norm(h[-1], dim=-1, keepdim=True) + eps

    lambda_factor = x_norm / (x_norm + h_norm)
    lambda_factor = torch.clamp(lambda_factor, min=0.1, max=0.9)#.mean().item()
    #print(lambda_factor)

    x_next = lambda_factor * x_mid + (1-lambda_factor) * W__h_to_x(h[-1])
    #print(x_next)
    ###time.sleep(0.5)
    #print(x_mid.shape, h[-1].shape, x_next.shape)           # torch.Size([1, 128, 2]), torch.Size([128, 8]), torch.Size([1, 128, 2])

    return x_next 




class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.mode = False
        
        self.W = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=False)


    def forward(self, u, h, c):        
        gates = self.W(u) + self.U(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)                            # ingate
        f = torch.sigmoid(f)                            # forgetgate
        g = torch.tanh(g)                               # cellgate
        o = torch.sigmoid(o)                            # outgate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next