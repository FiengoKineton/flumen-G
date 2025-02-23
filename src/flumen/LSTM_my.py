import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = dropout

        self.mhu = 1.5
        self.state_dim = 2

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if layer==0 else hidden_size, hidden_size, bias)
            for layer in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size) if output_size is not None else None
        self._initialize_weights()

    
    def _initialize_weights(self): 
        for layer in self.lstm_cells:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)


    def forward(self, rnn_input, hidden_state, tau=None):

        is_packed = isinstance(rnn_input, torch.nn.utils.rnn.PackedSequence)

        if is_packed:
            rnn_input_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
            batch_size, seq_len, _ = rnn_input_unpacked.shape
            device = rnn_input_unpacked.device
        else:
            device = rnn_input_unpacked.device
            batch_size, seq_len, _ = rnn_input.shape


        h0, c0 = hidden_state if hidden_state is not None else (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        )

        self.A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]], device=device)

        h_prev = h0.clone()
        c_prev = c0.clone()
        outputs = []

        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :] if is_packed else rnn_input[:, t, :]

            if t==0: 
                tau_prev = tau[t, :] if tau is not None else None
                h_prev = h_prev[:, t] 
            else: 
                tau_prev = tau[t-1, :] if tau is not None else None
                x_prev = h_prev[:, :self.state_dim]
                x_next = x_prev + tau_prev * torch.mathmul(self.A, x_prev.T).T if tau is not None else x_prev
                h_prev[:, :self.state_dim] = x_next

            for layer in range(self.num_layers):
                new_h, new_c = [], []
                h_next, c_next = self.lstm_cells[layer](rnn_input_t, h_prev[layer], c_prev[layer])

                new_h.append(h_next)
                new_c.append(c_next)

                rnn_input_t = h_next

                h_new = torch.stack(new_h).clone()
                c_new = torch.stack(new_c).clone()
                h_prev = h_new
                c_prev = c_new

            outputs.append(rnn_input_t)

        outputs = torch.stack(outputs, dim=1 if self.batch_first else 0)

        if is_packed:
            outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)

        if self.fc is not None:
            outputs = self.fc(outputs[:, -1, :])  # Use last time step

        return outputs, (h_next, c_next)




class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Gates: Weights for input, forget, candidate, and output gates
        self.W_i = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_f = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_g = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_o = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Recurrent weights (for the hidden state)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Biases for each gate (optional depending on `bias` flag)
        self.b_i = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.b_f = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.b_g = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.b_o = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, u, h, c):
        """
        u = u_t         h = h_{t-1}     c = c_{t-1}
        i = i_t         f = f_t         g = g_t         o = o_t
        c_next = c_t    h_next = h_t
        """
        
        # Compute the gates (input, forget, candidate, output)
        i = torch.sigmoid(self.W_i(u) + self.U_i(h) + (self.b_i if self.b_i is not None else 0))
        f = torch.sigmoid(self.W_f(u) + self.U_f(h) + (self.b_f if self.b_f is not None else 0))
        g = torch.tanh(self.W_g(u) + self.U_g(h) + (self.b_g if self.b_g is not None else 0))
        o = torch.sigmoid(self.W_o(u) + self.U_o(h) + (self.b_o if self.b_o is not None else 0))
        
        # Update cell and hidden states
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

