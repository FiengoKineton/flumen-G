import torch
from torch import nn
from flumen.LSTM_my import LSTM


"""
COMMANDs:

python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/vdp.yaml vdp_test_data
python experiments/train_wandb.py data/vdp_test_data.pkl vdp_test
python experiments/interactive_test.py --wandb (name of the best model from a general experiment)
________________________________________________________________________________________________________________________

W&B workspace:  https://wandb.ai/aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis?nw=nwuserg7fiengo

Slack general:  https://app.slack.com/client/T080VKDGZMY/C080SPVEXA9
________________________________________________________________________________________________________________________

GITs:

>> git reset --hard origin/master
>> git pull origin master
________________________________________________________________________________________________________________________
"""



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

        self.mode = True                

        print("\n\nCausalFlowModel init variables:")
        print("\tstate_dim: ", state_dim)                 # output | 2
        print("\tcontrol_dim: ", control_dim)             # output | 1
        print("\toutput_dim: ", output_dim)               # output | 2
        print("\tcontrol_rnn_size: ", control_rnn_size)   # output | 8
        print("\tcontrol_rnn_depth: ", control_rnn_depth) # output | 1
        print("\tencoder_size: ", encoder_size)           # output | 1
        print("\tencoder_depth: ", encoder_depth)         # output | 2
        print("\tdecoder_size: ", decoder_size)           # output | 1
        print("\tdecoder_depth: ", decoder_depth)         # output | 2



    ### LSTM with depth=1, as suggested ----- LSTM() eller torch.nn.LSTM()
        self.u_rnn = LSTM(
            input_size=control_dim + 1,                     # output | 2
            hidden_size=control_rnn_size + state_dim,       # output | 14
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0, 
            state_dim=self.state_dim
        ) if self.mode else torch.nn.LSTM(
            input_size=control_dim + 1,                     # output | 2
            hidden_size=control_rnn_size + state_dim,       # output | 14
            batch_first=True,
            num_layers=self.control_rnn_depth,              # output | 1
            dropout=0      
        )

    ### ENCODER
        x_dnn_osz = control_rnn_size * self.control_rnn_depth
        self.x_dnn = FFNet(in_size=state_dim,               # output | 2
                        out_size=x_dnn_osz,                 # output | 12
                        hidden_size=encoder_depth *         # output | 24
                        (encoder_size * x_dnn_osz, ),
                        use_batch_norm=use_batch_norm)

    ### Updated DECODER that takes [x_tilde, h_tilde]
        u_dnn_isz = control_rnn_size + state_dim            # before | no +state_dim | for new Encoder
        self.u_dnn = FFNet(in_size=u_dnn_isz,               # output | 14
                        out_size=output_dim,                # output | 2
                        hidden_size=decoder_depth *         # output | 28
                        (decoder_size * u_dnn_isz, ),
                        use_batch_norm=use_batch_norm)
            



    def forward(self, x, rnn_input, deltas, tau=None):
        h0 = self.x_dnn(x)  
        h0_stack = torch.cat((x, h0), dim=1)        # [x, h0]
        h0_stack = h0_stack.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)
        c0 = torch.zeros_like(h0_stack)  

        tau = torch.full((x.shape[0], x.shape[1]), 0.01, device=x.device) if self.mode else None

        """
        print("\n\nCasualFlowModel variables's shape:")
        print("\tx.shape", x.shape)                     # output | torch.Size([128, 2])
        print("\th0.shape", h0.shape)                   # output | torch.Size([128, 8])
        print("\th0_stack.shape: ", h0_stack.shape)     # output | torch.Size([1, 128, 10])
        print("\tc0.shape: ", c0.shape)                 # output | torch.Size([1, 128, 10])
        print("\ttau.shape: ", tau.shape)               # output | torch.Size([128, 2])
        #"""

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0_stack, c0), tau)

        h, h_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn_out_seq_packed, batch_first=True)
        print("\n")
        print("\th.shape:", h.shape)                        # output | torch.Size([128, 50, 14])
        print("\th_lens.shape:", h_lens.shape)              # output | torch.Size([128])

        h_shift = torch.roll(h, shifts=1, dims=1)
        print("\n")
        print("\th_shift.shape:", h_shift.shape)            # output | torch.Size([128, 50, 14])
        print("\th0_stack[-1].shape:", h0_stack[-1].shape)  # output | torch.Size([128, 14]) 
        h_shift[:, 0, :] = h0_stack[-1]

        encoded_controls = (1 - deltas) * h_shift + deltas * h  

        decoder_input = encoded_controls  
        output = self.u_dnn(decoder_input[range(encoded_controls.shape[0]), h_lens - 1, :])
        output = output[:, :self.state_dim]  # Ensure only x_tilde is output | for new Encoder

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