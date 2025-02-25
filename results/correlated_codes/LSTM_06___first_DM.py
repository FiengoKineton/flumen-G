import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, state_dim=None):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.state_dim = state_dim

        """
        print("\n\nLSTM init variables:\n---------------------------\n")           
        print("\tinput_size:", input_size)              # output | 2
        print("\thidden_size:", hidden_size)            # output | 10
        print("\tnum_layers:", num_layers)              # output | 1
        print("\toutput_size:", output_size)            # output | None
        print("\tbias:", bias)                          # output | True
        print("\tbatch_first:", batch_first)            # output | True
        print("\tdropout:", dropout)                    # output | 0
        print("\tbidirectional:", bidirectional)        # output | False
        print("\tstate_dim:", state_dim)                # output | 2
        #"""

        self.mhu = 1.5

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if layer==0 else hidden_size, hidden_size, bias)
            for layer in range(num_layers)
        ])

        ###self.fc = nn.Linear(hidden_size, output_size) if output_size is not None else None
        ###self._initialize_weights()

    
    def _initialize_weights(self): 
        for layer in self.lstm_cells:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)


    def forward(self, rnn_input, hidden_state, tau=None):

        rnn_input_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        _, seq_len, _ = rnn_input_unpacked.shape
        device = rnn_input_unpacked.device


        h0, c0 = hidden_state
        self.A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]], device=device)

        h = h0.clone()
        c = c0.clone()
        outputs = []

        """
        print("\n\nLSTM forward variables:\n---------------------------\n")
        print("\trnn_input_unpacked.shape:", rnn_input_unpacked.shape)  # output | torch.Size([128, 75, 2])
        print("\tlengths.shape:", lengths.shape[0])                     # output | 128
        print("\tseq_len:", seq_len)                                    # output | 75
        print("\n") 
        print("\th0.shape:", h0.shape)                                  # output | torch.Size([1, 128, 10])
        print("\tc0.shape:", c0.shape)                                  # output | torch.Size([1, 128, 10])
        print("\tdevice:", device)                                      # output | cpu
        print("\th.shape:", h.shape)                                    # output | torch.Size([1, 128, 10])
        print("\tc.shape:", c.shape)                                    # output | torch.Size([1, 128, 10])
        print("\n")
        print("\tA.shape:", self.A.shape)                               # output | torch.Size([2, 2])
        ###print("\ttau.shape:", tau.shape if tau is not None else None)   # output | torch.Size([128, 2])         # now tau is a float
        print("\tx.shape:", h[:, :, :self.state_dim].shape)             # output | torch.Size([1, 128, 2])    
        #"""


        ###print("\n\nLSTM forward loop:\n---------------------------\n")
        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :]

            x_prev = h[:, :, :self.state_dim]
            x_next = x_prev + tau * torch.matmul(x_prev, self.A) if tau is not None and t!=0 else x_prev
            h[:, :, :self.state_dim] = x_next

            """
            print("\tt:", t)
            print("\tnn_input_t.shape:", rnn_input_t.shape)             # output | torch.Size([128, 2])
            print("\tx_prev.shape:", x_prev.shape)                      # output | torch.Size([1, 128, 2])
            print("\tx_next.shape:", x_next.shape)                      # output | torch.Size([1, 128, 2])
            print("\th.shape (before):", h.shape)                       # output | torch.Size([1, 128, 10])
            print("\tc.shape (before):", c.shape)                       # output | torch.Size([])
            #"""

            for layer in range(self.num_layers):
                new_h, new_c = [], []
                ht, ct = self.lstm_cells[layer](rnn_input_t, h[layer], c[layer])
                new_h.append(ht)
                new_c.append(ct)
                h = torch.stack(new_h)
                c = torch.stack(new_c)

                #"""
                print("\t\tlayer:", layer)
                print("\t\tht.shape:", ht.shape)                        # output | torch.Size([128, 10])
                print("\t\tct.shape:", ct.shape)                        # output | torch.Size([])        
                print("\t\th[layer].shape:", h[layer].shape)            # output | torch.Size([])
                print("\t\tc[layer].shape:", c[layer].shape)            # output | torch.Size([])
                print("\t\th.shape (after):", h.shape)                  # output | torch.Size([1, 128, 10])
                print("\t\tc.shape (after):", c.shape)                  # output | torch.Size([])
                print("\n")
                #"""

            outputs.append(ht)

        outputs = torch.stack(outputs, dim=1 if self.batch_first else 0)
        ###print(outputs.shape)                                            # output | torch.Size([128, 75, 10])
        outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)
        ###outputs = self.fc(outputs[:, -1, :])  if self.fc is not None else outputs

        return outputs, (h, c)
        ###return ht, ht.dim()




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

        """
        print("\n\nLSTMCell init variables:\n---------------------------\n")
        print("\tinput_size:", input_size)              # output | 2
        print("\thidden_size:", hidden_size)            # output | 10
        print("\tbias:", bias)                          # output | True
        print("\n")
        print("\tW_i.shape:", self.W_i.weight.shape)    # output | torch.Size([10, 2])
        print("\tW_f.shape:", self.W_f.weight.shape)    # output | torch.Size([10, 2])
        print("\tW_g.shape:", self.W_g.weight.shape)    # output | torch.Size([10, 2])
        print("\tW_o.shape:", self.W_o.weight.shape)    # output | torch.Size([10, 2])
        print("\n")
        print("\tU_i.shape:", self.U_i.weight.shape)    # output | torch.Size([10, 10])
        print("\tU_f.shape:", self.U_f.weight.shape)    # output | torch.Size([10, 10])
        print("\tU_g.shape:", self.U_g.weight.shape)    # output | torch.Size([10, 10])
        print("\tU_o.shape:", self.U_o.weight.shape)    # output | torch.Size([10, 10])
        print("\n")
        print("\tb_i.shape:", self.b_i.shape)           # output | torch.Size([10])
        print("\tb_f.shape:", self.b_f.shape)           # output | torch.Size([10])
        print("\tb_g.shape:", self.b_g.shape)           # output | torch.Size([10])
        print("\tb_o.shape:", self.b_o.shape)           # output | torch.Size([10])
        #"""


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

        """
        print("\n\nLSTMCell forward variables:\n---------------------------\n")
        print("\tu.shape:", u.shape)                    # output | torch.Size([128, 2])
        print("\th.shape:", h.shape)                    # output | torch.Size([128, 10])
        print("\tc.shape:", c.shape)                    # output | torch.Size([128, 10])
        print("\n")
        print("\ti.shape:", i.shape)                    # output | torch.Size([128, 10])
        print("\tf.shape:", f.shape)                    # output | torch.Size([128, 10])
        print("\tg.shape:", g.shape)                    # output | torch.Size([128, 10])
        print("\to.shape:", o.shape)                    # output | torch.Size([128, 10])
        print("\n")
        print("\tc_next.shape:", c_next.shape)          # output | torch.Size([128, 10])
        print("\th_next.shape:", h_next.shape)          # output | torch.Size([128, 10])
        #"""
        return h_next, c_next