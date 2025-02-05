import torch
from torch import nn


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
        self.control_rnn_depth = control_rnn_depth                      # before | = control_rnn_depth (bc now num_layers=1))


        self.check = True

        if self.check:
            print("\nRIGTH model.py\n")
            print("control_dim:", control_dim)
            print("control_rnn_size:", control_rnn_size)
            print("control_rnn_depth:", control_rnn_depth)
            print("state_dim:", state_dim)
            print("output_dim:", output_dim)
            print("\n")



    ### LSTM with depth=1, as suggested
        self.u_rnn = torch.nn.LSTM(
            input_size=state_dim + control_dim + 1,         # before | =1 + control_dim,
            hidden_size=control_rnn_size + state_dim,       # before | no +state_dim
            batch_first=True,
            num_layers=self.control_rnn_depth,    
            dropout=0,
        )

    ### ENCODER
        x_dnn_osz = control_rnn_size * self.control_rnn_depth + state_dim   # before | no +state_dim    
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
        if self.check: print("x shape:", x.shape)

        h0 = self.x_dnn(x)  
        if self.check: print("h0 shape:", h0.shape)

        # h0_stack = torch.cat((x, h0), dim=1)  # [x, h0]
        h0_stack = h0[:, :self.control_rnn_size + self.state_dim]  
        if self.check: print("h0_stack shape:", h0_stack.shape)


    # ----------------------------------------------------------------------------------------------------------------------- #
        """
        #h0_stack = h0_stack.unsqueeze(0)  # Shape: [1, batch_size, state_dim + hidden_size]
        #h0_stack = torch.stack(h0_stack.split(self.control_rnn_size+self.state_dim, dim=1))
        
        chunks = h0_stack.split(self.control_rnn_size + self.state_dim, dim=1)

        # If last chunk is smaller, pad it
        if chunks[-1].shape[1] != self.control_rnn_size + self.state_dim:
            padding = torch.zeros_like(chunks[0][:, :self.control_rnn_size + self.state_dim])
            chunks = list(chunks[:-1]) + [padding]

        h0_stack = torch.stack(chunks)"""
    # ----------------------------------------------------------------------------------------------------------------------- #


        h0_stack = h0_stack.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)  


        if self.check: print("h0_stack-2 shape:", h0_stack.shape)
        
        c0 = torch.zeros_like(h0_stack)  
        if self.check: print("c0 shape:", c0.shape)



    # ----------------------------------------------------------------------------------------------------------------------- #
        rnn_input_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_input, batch_first=True)
        if self.check: print("rnn_input shape before:", rnn_input_unpacked.shape)

        # Ensure rnn_input contains the correct feature dimensions
        x_expanded = x.unsqueeze(1).expand(-1, rnn_input_unpacked.shape[1], -1)
        rnn_input = torch.cat((x_expanded, rnn_input_unpacked), dim=2)

        assert rnn_input.shape[-1] == self.state_dim + self.control_dim + 1, \
            f"rnn_input shape mismatch: expected {self.state_dim + self.control_dim + 1}, got {rnn_input.shape[-1]}"

        if self.check: print("rnn_input shape after fix:", rnn_input.shape)


        lengths = (rnn_input.abs().sum(dim=2) != 0).sum(dim=1)  
        lengths = lengths.cpu()
        rnn_input_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
    # ----------------------------------------------------------------------------------------------------------------------- #


        rnn_out_seq_packed, _ = self.u_rnn(rnn_input_packed, (h0_stack, c0))
        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed, batch_first=True)

        h_shift = torch.roll(h, shifts=1, dims=1)
        h_shift[:, 0, :] = h0_stack[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h  

        decoder_input = encoded_controls  
        output = self.u_dnn(decoder_input[range(encoded_controls.shape[0]), h_lens - 1, :])
        output = output[:, :self.state_dim]  # Ensure only x_tilde is output


        if self.check:
            self.check = False
            # print("x shape:", x.shape)
            # print("h0 shape (after encoder):", h0.shape)
            # print("h0_stack shape (LSTM input):", h0_stack.shape)
            print("\nrnn_input shape:", rnn_input.shape)
            print("decoder_input shape:", decoder_input.shape)


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
