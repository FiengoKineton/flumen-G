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
        self.A = torch.tensor([[self.mhu, -self.mhu], [1/self.mhu, 0]])

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if layer==0 else hidden_size, hidden_size, bias)
            for layer in range(num_layers)
        ])

        ###self.fc = nn.Linear(hidden_size, output_size) if output_size is not None else None




    def forward(self, rnn_input: PackedSequence, hidden_state, tau=None):

        rnn_input_unpacked = rnn_input.data
        batch_sizes = rnn_input.batch_sizes
        device = rnn_input_unpacked.device
        seq_len = batch_sizes.size(0)

        lengths = torch.zeros(batch_sizes.max().item(), dtype=torch.long, device=device)
        for i in range(len(batch_sizes)):   lengths[:batch_sizes[i]] += 1  

        h, c = hidden_state
        self.A = self.A.to(device)

        outputs = torch.zeros(h.size(1), seq_len, self.hidden_size, device=device)

        """
        print("\n\nLSTM forward variables:\n---------------------------\n")
        print("\trnn_input_unpacked.shape:", rnn_input_unpacked.shape)  # output | torch.Size([128, 75, 2])
        print("\tlengths.shape:", lengths.shape[0])                     # output | 128
        print("\tseq_len:", seq_len)                                    # output | 75
        print("\n") 
        print("\th0.shape:", h.shape)                                   # output | torch.Size([1, 128, 10])
        print("\tc0.shape:", c.shape)                                   # output | torch.Size([1, 128, 10])
        print("\tdevice:", device)                                      # output | cpu
        print("\th.shape:", h.shape)                                    # output | torch.Size([1, 128, 10])
        print("\tc.shape:", c.shape)                                    # output | torch.Size([1, 128, 10])
        print("\n")
        print("\tA.shape:", self.A.shape)                               # output | torch.Size([2, 2])
        print("\tx.shape:", h[:, :, :self.state_dim].shape)             # output | torch.Size([1, 128, 2])    
        #"""


        ###print("\n\nLSTM forward loop:\n---------------------------\n")
        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:batch_sizes[t]]           

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
                h_new, c_new = self.lstm_cells[layer](rnn_input_t, h[layer, :batch_sizes[t]], c[layer, :batch_sizes[t]])
                h, c = h.clone(), c.clone()
                h[layer, :batch_sizes[t]], c[layer, :batch_sizes[t]] = h_new, c_new

                #"""
                print("\t\tlayer:", layer)
                print("\t\tht.shape:", h[layer, :batch_sizes[t]].shape) # output | torch.Size([128, 10])
                print("\t\tct.shape:", h[layer, :batch_sizes[t]].shape) # output | torch.Size([128, 10])                                
                print("\t\th[layer].shape:", h[layer].shape)            # output | torch.Size([])
                print("\t\tc[layer].shape:", c[layer].shape)            # output | torch.Size([])
                print("\t\th.shape (after):", h.shape)                  # output | torch.Size([1, 128, 10])
                print("\t\tc.shape (after):", c.shape)                  # output | torch.Size([])
                print("\n")
                #"""

            outputs[:batch_sizes[t], t, :] = h[-1, :batch_sizes[t]]        

        ###print("\n\toutputs.shape:", outputs.shape)                      # output | torch.Size([128, 75, 10])
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)

        return packed_outputs, (h, c)



###@torch.jit.script
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        self.W = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=False)


    def forward(self, u, h, c):        
        gates = self.W(u) + self.U(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        """
        print("\n\nLSTMCell forward variables:\n---------------------------\n")
        print("\tW.shape:", self.W.weight.shape)        # output | torch.Size([40, 2])
        print("\tU.shape:", self.U.weight.shape)        # output | torch.Size([40, 10])
        print("\n")
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