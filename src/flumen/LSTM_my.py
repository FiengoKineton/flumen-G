import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import sys


class LSTM(nn.Module):
    def __init__(self, input_size, z_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, state_dim=None, discretisation_mode=None):
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

        function_name = f"discretisation_{discretisation_mode}"
        self.discretisation_function = globals().get(function_name)

        ###jit_lstm_cells = torch.jit.script(LSTMCell(input_size=10, hidden_size=20))
        self.lstm_cells = nn.ModuleList([
            torch.jit.script(LSTMCell(input_size+state_dim if layer==0 else self.hidden_size, self.hidden_size, bias))
            for layer in range(num_layers)
        ])

        ###self.fc = nn.Linear(hidden_size, output_size) if output_size is not None else None



    def forward(self, rnn_input: PackedSequence, hidden_state, tau):

        rnn_input_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        _, seq_len, _ = rnn_input_unpacked.shape      # output | batch_size, seq_len, input_size = torch.Size([512, 75, 2])
        device = rnn_input_unpacked.device

        z, c = hidden_state
        self.A = self.A.to(device)
        self.I = torch.eye(self.A.shape[0], dtype=self.A.dtype, device=device)

        outputs = torch.zeros(z.size(1), seq_len, self.hidden_size+self.state_dim, device=device)


        #print("\n\tdiscretisation_mode:", discretisation_mode)
        #print("\tfunction_name:", function_name)
        #print("\tdiscretisation_function:", discretisation_function)

        # if discretisation_function is None: raise ValueError(f"Unknown discretisation mode: {discretisation_mode}. Available modes: none, FE, BE, TU")

        """
        print("\n\nLSTM forward variables:\n---------------------------\n")
        # NOTE: the h is is the z variable!
        print("\trnn_input_unpacked.shape:", rnn_input_unpacked.shape)  # output | torch.Size([128, 75, 2])
        print("\tlengths.shape:", lengths.shape[0])                     # output | 128
        print("\batch_size:", batch_size)                               # output | 512
        print("\tseq_len:", seq_len)                                    # output | 75
        print("\tinput_size:", input_size)                              # output | 2
        print("\n") 
        print("\th0.shape:", h.shape)                                   # output | torch.Size([1, 128, 10])
        print("\tc0.shape:", c.shape)                                   # output | torch.Size([1, 128, 10])
        print("\tdevice:", device)                                      # output | cpu
        print("\th.shape:", h.shape)                                    # output | torch.Size([1, 128, 10])
        print("\tc.shape:", c.shape)                                    # output | torch.Size([1, 128, 10])
        print("\n")
        print("\tA.shape:", self.A.shape)                               # output | torch.Size([2, 2])
        print("\tx.shape:", h[:, :, :self.state_dim].shape)             # output | torch.Size([1, 128, 2]) 
        print("\ttau.shape:", tau.shape)                                # output | torch.Size([128, 75, 1])
        print("\n")
        print("\toutputs.shape (before):", outputs.shape)               # output | torch.Size([])
        #"""


        ###print("\n\nLSTM forward loop:\n---------------------------\n")
        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :]       
            tau_t = tau[:, t, :] 

            x_prev = z[:, :, :self.state_dim]
            h = z[:, :, self.state_dim:]
            #x_next = self.discretisation_function(x_prev, self.A, tau_t, self.I)

            #print("\tx_prev.shape:", x_prev.shape)                      # output | torch.Size([1, 128, 2])
            #print("\th.shape:", h.shape)                                # output | torch.Size([1, 128, 8])
            #print("\tx_next.shape:", x_next.shape)                      # output | torch.Size([1, 128, 2])

        #-- F.E.    (forward euler)     --- s = (z-1) / tau
            #x_next = self.FE(x_prev, self.A, tau, t, self.I)
        #-- B.E.    (backward euler)    --- s = (z-1)/(tau*z)
            #x_next = self.BE(x_prev, self.A, tau, t, self.I)
        #-- TU.     (tustin)            --- s = 2/tau * (z-1)/(z+1)
            #x_next = self.TU(x_prev, self.A, tau, t, self.I)

            """
            print("\tt:", t)
            print("\tseq_len:", seq_len)
            print("\tnn_input_t.shape:", rnn_input_t.shape)             # output | torch.Size([128, 2])
            print("\tx_prev.shape:", x_prev.shape)                      # output | torch.Size([1, 128, 2])
            print("\tx_next.shape:", x_next.shape)                      # output | torch.Size([1, 128, 2])
            print("\th.shape (before):", h.shape)                       # output | torch.Size([1, 128, 10])
            print("\tc.shape (before):", c.shape)                       # output | torch.Size([[1, 128, 10])
            print("\ttau_t.shape:", tau_t.shape)                        # outpuy | toch.Size([128, 1])
            print("\n")
            print("\tx_k:", x_prev)                                     
            print("\tx_{k+1} (before):", x_next)
            #"""

            """
            z = [x, h]

            x = f_x(x, h)
            h = f_{lstm_cell}(h, [rnn_input, x])
            """


            #h_new_list, c_new_list = [], []
            for layer in range(self.num_layers):
                u_t = torch.cat((x_prev.squeeze(0), rnn_input_t), dim=1)   
                ###print(u_t.shape)                                     # output | torch.Size([128, 4])

                h_new, c_new = self.lstm_cells[layer](u_t, h[layer], c[layer])
                h, c = h.clone(), c.clone()
                h[layer], c[layer] = h_new, c_new

                #h_new_list.append(h_new)
                #c_new_list.append(c_new)

                """
                print("\t\tlayer:", layer)
                print("\t\tht.shape:", h[layer].shape)                  # output | torch.Size([128, 10])
                print("\t\tct.shape:", h[layer].shape)                  # output | torch.Size([128, 10])                                
                print("\t\th[layer].shape:", h[layer].shape)            # output | torch.Size([128, 10])
                print("\t\tc[layer].shape:", c[layer].shape)            # output | torch.Size([128, 10])
                print("\t\th.shape (after):", h.shape)                  # output | torch.Size([1, 128, 10])
                print("\t\tc.shape (after):", c.shape)                  # output | torch.Size([1, 128, 10])
                #"""

            x_next = self.discretisation_function(x_prev, self.A, tau_t, self.I)
            #h = torch.stack(h_new_list, dim=0)
            #c = torch.stack(c_new_list, dim=0)

            z[:, :, :self.state_dim] = x_next
            z[:, :, self.state_dim:] = h

            outputs[:, t, :] = z[-1]        

            """
            print("\tx_{k+1} (after):", h[:, :, :self.state_dim])
            print("\n\n")
            if t == 1 : sys.exit()
            #"""

        ###print("\n\toutputs.shape (after):", outputs.shape)          # output | torch.Size([128, 75, 10])
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)

        return packed_outputs, (z, c)


def discretisation_none(x_prev, A, tau, I):
    #print("\tdiscretisation_none")
    return x_prev

def discretisation_FE(x_prev, A, tau, I):
    #print("\tdiscretisation_FE")
    return x_prev + tau * torch.matmul(x_prev, A) 

def discretisation_BE(x_prev, A, tau, I):
    #print("\tdiscretisation_BE")
    return torch.matmul(x_prev, torch.inverse(I-tau*A)) 


#           print("\tx_prev.shape:", x_prev.shape)                      # output | torch.Size([1, 128, 2])
#           print("\ttau_t.shape:", tau_t.shape)                        # outpuy | toch.Size([128, 1])
#           print("\tA.shape:", self.A.shape)                           # output | torch.Size([2, 2])
#           print("\tI.shape:", self.I.shape)                           # output | torch.Size([2, 2])

def discretisation_TU_old(x_prev, A, tau, I):
    #print("\tdiscretisation_TU")
    return torch.matmul(x_prev, torch.matmul(I+tau/2*A, torch.inverse(I-tau/2*A))) #if tau is not None and t!=0 else x_prev


def discretisation_TU(x_prev, A, tau, I):
    """
    Tustin (bilinear) discretization applied per time step.

    Args:
    - x_prev: Previous state (1, batch, state_dim) → [1, 128, 2]
    - A: System matrix (state_dim, state_dim) → [2, 2]
    - tau: Time step vector for current timestep (batch, 1) → [128, 1]
    - I: Identity matrix (state_dim, state_dim) → [2, 2]

    Returns:
    - x_next: Discretized next state (1, batch, state_dim) → [1, 128, 2]
    """

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


#@torch.jit.script
class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.mode = False
        
        self.W = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=False)

        """self.weight_u = nn.Parameter(torch.randn(4*hidden_size, input_size))
        self.weight_h = nn.Parameter(torch.randn(4*hidden_size, hidden_size))
        self.bias_u = nn.Parameter(torch.randn(4*hidden_size))
        self.bias_h = nn.Parameter(torch.randn(4*hidden_size))
        """

    def forward(self, u, h, c):        
        #gates = ((torch.mm(u, self.weight_u.t()) + self.bias_u + torch.mm(h, self.weight_h.t()) + self.bias_h) 
                 #if self.mode else (self.W(u) + self.U(h)))
        gates = self.W(u) + self.U(h)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)                            # ingate
        f = torch.sigmoid(f)                            # forgetgate
        g = torch.tanh(g)                               # cellgate
        o = torch.sigmoid(o)                            # outgate

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        """
        print("\n\nLSTMCell forward variables:\n---------------------------\n")
        print("\tW.shape:", self.W.weight.shape)        # output | torch.Size([32, 4])
        print("\tU.shape:", self.U.weight.shape)        # output | torch.Size([32, 8])
        print("\n")
        print("\tu.shape:", u.shape)                    # output | torch.Size([128, 4])
        print("\th.shape:", h.shape)                    # output | torch.Size([128, 8])
        print("\tc.shape:", c.shape)                    # output | torch.Size([128, 8])
        print("\n")
        print("\ti.shape:", i.shape)                    # output | torch.Size([128, 8])
        print("\tf.shape:", f.shape)                    # output | torch.Size([128, 8])
        print("\tg.shape:", g.shape)                    # output | torch.Size([128, 8])
        print("\to.shape:", o.shape)                    # output | torch.Size([128, 8])
        print("\n")
        print("\tc_next.shape:", c_next.shape)          # output | torch.Size([128, 8])
        print("\th_next.shape:", h_next.shape)          # output | torch.Size([128, 8])
        #"""
        return h_next, c_next