import torch
from torch import nn
from flumen.LSTM_my import LSTM



class CausalFlowModel(nn.Module):

    def __init__(self,
                 state_dim,
                 control_dim,
                 output_dim,
                 control_rnn_size,
                 control_rnn_depth,
                 encoder_size,
                 encoder_depth,
                 decoder_size,
                 decoder_depth,
                 discretisation_mode,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size
        self.control_rnn_depth = control_rnn_depth   

        self.mode = True                                    # if True then h0_stack, else h0


        self.u_rnn = LSTM(
            input_size=control_dim + 1,                     # output | 2
            z_size=control_rnn_size + state_dim,            # output | 10
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0, 
            state_dim=self.state_dim,
            discretisation_mode=discretisation_mode
        ) if self.mode else torch.nn.LSTM(
            input_size=control_dim + 1,                     # output | 2
            hidden_size=control_rnn_size,                   # output | 8
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0      
        )

    #-- ENCODER
        x_dnn_osz = control_rnn_size * self.control_rnn_depth
        self.x_dnn = FFNet(in_size=state_dim,               # output | 2
                        out_size=x_dnn_osz,                 # output | 8
                        hidden_size=encoder_depth *         # output | 16
                        (encoder_size * x_dnn_osz, ),
                        use_batch_norm=use_batch_norm)

    #-- Updated DECODER that takes [x_tilde, h_tilde]
        u_dnn_isz = control_rnn_size + state_dim if self.mode else control_rnn_size
        self.u_dnn = FFNet(in_size=u_dnn_isz,               # output | 10
                        out_size=output_dim,                # output | 2
                        hidden_size=decoder_depth *         # output | 20
                        (decoder_size * u_dnn_isz, ),
                        use_batch_norm=use_batch_norm)
            



    def forward(self, x, rnn_input, deltas):
        h0 = self.x_dnn(x) 
        z = torch.cat((x, h0), dim=1)       

        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1)) if not self.mode else h0
        z = z.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)
        c0 = torch.zeros_like(z if self.mode else h0) 

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (z, c0), deltas) if self.mode else self.u_rnn(rnn_input, (h0, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed, batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)   
        h_temp = z[-1] if self.mode else h0[-1]
        h_shift[:, 0, :] = h_temp

        encoded_controls = (1 - deltas) * h_shift + deltas * h  
        output = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]), h_lens - 1, :])
        output = output[:, :self.state_dim]  

        return output




class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,
                 use_batch_norm=False):
        super(FFNet, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_size, hidden_size[0]))

        if use_batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_size[0]))

        self.layers.append(activation())

        for isz, osz in zip(hidden_size[:-1], hidden_size[1:]):
            self.layers.append(nn.Linear(isz, osz))

            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(osz))

            self.layers.append(activation())

        self.layers.append(nn.Linear(hidden_size[-1], out_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)

        return input