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
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size
        self.control_rnn_depth = control_rnn_depth                   

        
        """
        self.check = True

        if self.check:
            print("control_dim:", control_dim)
            print("control_rnn_size:", control_rnn_size)
            print("control_rnn_depth:", control_rnn_depth)
            print("state_dim:", state_dim)
            print("output_dim:", output_dim)
            print("\n")
        """



    ### LSTM with depth=1, as suggested ----- LSTM() eller torch.nn.LSTM()
        self.u_rnn = torch.nn.LSTM(
            input_size=control_dim + 1,                     # wrong, before | + state_dim
            hidden_size=control_rnn_size + state_dim,       # before | no +state_dim
            batch_first=True,
            num_layers=self.control_rnn_depth,    
            dropout=0,
        )

    ### ENCODER
        x_dnn_osz = control_rnn_size * self.control_rnn_depth       # wrong, before | + state_dim   
        self.x_dnn = FFNet(in_size=state_dim,
                        out_size=x_dnn_osz,
                        hidden_size=encoder_depth *
                        (encoder_size * x_dnn_osz, ),
                        use_batch_norm=use_batch_norm)

    ### Updated DECODER that takes [x_tilde, h_tilde]
        u_dnn_isz = control_rnn_size + state_dim        # before | no +state_dim (added to concatenate the state x_tilde)
        self.u_dnn = FFNet(in_size=u_dnn_isz,
                        out_size=output_dim,
                        hidden_size=decoder_depth *
                        (decoder_size * u_dnn_isz, ),
                        use_batch_norm=use_batch_norm)
            



    def forward(self, x, rnn_input, deltas):
    

        h0 = self.x_dnn(x)  
        h0_stack = torch.cat((x, h0), dim=1)  # [x, h0]
        # h0 = torch.stack(h0_stack.split(self.control_rnn_size+self.state_dim, dim=1))
        h0_stack = h0_stack.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)

        c0 = torch.zeros_like(h0_stack)  
        # torch.nn.init.xavier_uniform_(c0)


        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0_stack, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed, batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0_stack[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h  

        decoder_input = encoded_controls  
        output = self.u_dnn(decoder_input[range(encoded_controls.shape[0]), h_lens - 1, :])
        output = output[:, :self.state_dim]  # Ensure only x_tilde is output


        """
        if self.check:
            self.check = False

            print("\nx shape:", x.shape)                                    # output | torch.Size([512, 2])
            print("\nh0 shape:", h0.shape)                                  # output | torch.Size([512, 16])
            print("\nh0_stack shape:", h0_stack.shape)                      # output | torch.Size([1, 512, 18])
            print("\nc0 shape:", c0.shape)                                  # output | torch.Size([1, 512, 18])

            # print("\nrnn_input_unpacked shape:", rnn_input_unpacked.shape)  # output | torch.Size([512, 75, 2])
            # print("\nrnn_input shape:", rnn_input.shape)                    # output | bho!  
            print("\ndecoder_input shape:", decoder_input.shape)            # output | torch.Size([512, 75, 18])
            print("\n\n")
        """

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