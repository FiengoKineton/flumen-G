import torch, sys
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
                 x_update_mode,
                 model_name,
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size
        self.control_rnn_depth = control_rnn_depth   

        self.mode_rnn = "new"                               # if new then h0_stack, else h0
        self.mode_dnn = True                                # if True then old dnn
        print("\n\nmode_rnn:", self.mode_rnn, "\nmode_dnn:", self.mode_dnn, "\n\n")

        function_name = f"mode_rnn_{self.mode_rnn}"
        self.structure_function = getattr(self, function_name, None)


        self.u_rnn = LSTM(
            input_size=control_dim + 1,                     # output | 2
            z_size=control_rnn_size + state_dim,            # output | 10
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0, 
            state_dim=self.state_dim,
            discretisation_mode=discretisation_mode, 
            x_update_mode=x_update_mode, 
            model_name=model_name
        ) if self.mode_rnn=="new" else torch.nn.LSTM(
            input_size=control_dim + 1,                     # output | 2
            hidden_size=control_rnn_size,                   # output | 8
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0      
        )

    #-- ENCODER
        x_dnn_osz = control_rnn_size * self.control_rnn_depth
        self.x_dnn = FFNet(
            in_size=state_dim,                              # output | 2
            out_size=x_dnn_osz,                             # output | 8
            hidden_size=encoder_depth *                     # output | 16
            (encoder_size * x_dnn_osz, ),
            use_batch_norm=use_batch_norm
        ) if self.mode_dnn else FFNet(
            in_size=state_dim, 
            out_size=x_dnn_osz, 
            hidden_size=[state_dim * 2, x_dnn_osz]
        )

    #-- Updated DECODER that takes [x_tilde, h_tilde]
        u_dnn_isz = control_rnn_size + state_dim if self.mode_rnn=="new" else control_rnn_size
        self.u_dnn = FFNet(
            in_size=u_dnn_isz,                              # output | 10
            out_size=output_dim,                            # output | 2
            hidden_size=decoder_depth *                     # output | 20
            (decoder_size * u_dnn_isz, ),
            use_batch_norm=use_batch_norm
        ) if self.mode_dnn else FFNet(
            in_size=u_dnn_isz, 
            out_size=output_dim, 
            hidden_size=[u_dnn_isz * 2, output_dim]
        )
            



    def forward(self, x, rnn_input, deltas):
        ###h0, c0, tau = self.structure_function(x, deltas)
        ###rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0), tau)

        h0, rnn_out_seq_packed, coefficients = self.structure_function(x, deltas, rnn_input)    ###############
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed, batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)   
        h_temp = h0[-1]
        h_shift[:, 0, :] = h_temp

        encoded_controls = (1 - deltas) * h_shift + deltas * h  
        output = self.u_dnn(encoded_controls[range(encoded_controls.shape[0]), h_lens - 1, :])
        output = output[:, :self.state_dim]  

        ###sys.exit()
        return output, coefficients ###############


    def mode_rnn_new(self, x, deltas, rnn_input):
        h0 = self.x_dnn(x)
        z = torch.cat((x, h0), dim=1) 
        z = z.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)
        c0 = torch.zeros_like(z)

        rnn_out_seq_packed, _, coefficients = self.u_rnn(rnn_input, (z, c0), deltas)    ###############

        return z, rnn_out_seq_packed, coefficients  ###############


    def mode_rnn_old(self, x, _, rnn_input):
        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))

        return h0, rnn_out_seq_packed, torch.tensor([[0, 0], [0, 0]], dtype=torch.float32) ###############


class FFNet(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 activation=nn.Tanh,            # try | nn.ReLU --- not good!
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