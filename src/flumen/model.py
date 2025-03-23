import torch, sys
from torch import nn
from flumen.LSTM_my import LSTM
from flumen.GRU_my import GRU
from flumen.Seq2Seq import FFNet, ConvNet, SelfAttention, ResidualBlock, GRUEncoderDecoder



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
                 mode_rnn='new',
                 mode_dnn='FFNet',
                 use_batch_norm=False):
        super(CausalFlowModel, self).__init__()

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim

        self.control_rnn_size = control_rnn_size
        self.control_rnn_depth = control_rnn_depth  
        self.encoder_size = encoder_size
        self.encoder_depth = encoder_depth
        self.decoder_size = decoder_size
        self.decoder_depth = decoder_depth
        self.use_batch_norm = use_batch_norm


        self.mode_rnn = mode_rnn                            # {"new", "old", "gru"} | if new then h0_stack, else h0
        self.mode_dnn = mode_dnn                            # if True then old dnn  | better always True
        print("\n\nmode_rnn:", self.mode_rnn, "\nmode_dnn:", self.mode_dnn, "\n\n")

        function_name = f"mode_rnn_{self.mode_rnn}"
        self.structure_function = getattr(self, function_name, None)


        if self.mode_rnn=='new': 
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
            ) 
        elif self.mode_rnn=='old': 
            self.u_rnn = torch.nn.LSTM(
                input_size=control_dim + 1,                     # output | 2
                hidden_size=control_rnn_size,                   # output | 8
                batch_first=True,
                num_layers=self.control_rnn_depth,              # output | 1
                dropout=0      
            )
        elif self.mode_rnn=='gru': 
            self.u_rnn = GRU(
                input_size=control_dim + 1,                     # output | 2
                z_size=control_rnn_size + state_dim,            # output | 10
                batch_first=True,
                num_layers=self.control_rnn_depth,              # output | 1
                dropout=0, 
                state_dim=self.state_dim,
                discretisation_mode=discretisation_mode, 
                x_update_mode=x_update_mode, 
                model_name=model_name
            ) 


    #-- ENCODER
        self.x_dnn_osz = self.control_rnn_size * self.control_rnn_depth
        self.x_dnn = self.seq_to_seq("encoder")
        
        """FFNet(
            in_size=self.state_dim,                            
            out_size=self.x_dnn_osz,                           
            hidden_size=self.encoder_depth *                   
            (self.encoder_size * self.x_dnn_osz, ),
            use_batch_norm=self.use_batch_norm
        ) """

    #-- Updated DECODER that takes [x_tilde, h_tilde]
        self.u_dnn_isz = self.control_rnn_size if self.mode_rnn=="old" else self.control_rnn_size + self.state_dim
        self.u_dnn = self.seq_to_seq("decoder")
        
        """FFNet(
            in_size=self.u_dnn_isz,                        
            out_size=self.output_dim,                          
            hidden_size=self.decoder_depth *                     
            (self.decoder_size * self.u_dnn_isz, ),
            use_batch_norm=self.use_batch_norm
        ) """
            

    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    def seq_to_seq(self, which):

        if which=="encoder":

            if self.mode_dnn=="FFNet": 
                return FFNet(
                    in_size=self.state_dim,                            
                    out_size=self.x_dnn_osz,                           
                    hidden_size=self.encoder_depth *                   
                    (self.encoder_size * self.x_dnn_osz, ),
                    use_batch_norm=self.use_batch_norm
                ) 
            
            elif self.mode_dnn=="ConvNet": 
                return ConvNet(
                    in_size=self.state_dim, 
                    out_size=self.x_dnn_osz, 
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="SelfAttention":
                return SelfAttention(
                    input_dim=self.state_dim,                            
                    output_dim=self.x_dnn_osz,                           
                    hidden_dim=self.encoder_depth *                   
                    (self.encoder_size * self.x_dnn_osz, ),
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="ResidualBlock": 
                return ResidualBlock(
                    input_dim=self.state_dim,                            
                    output_dim=self.x_dnn_osz,                           
                    hidden_dim=self.encoder_depth *                   
                    (self.encoder_size * self.x_dnn_osz, ),
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="GRUEncoderDecoder": 
                return GRUEncoderDecoder(
                    input_size=self.state_dim,                            
                    output_size=self.x_dnn_osz,                           
                    hidden_size=self.encoder_depth *                   
                    (self.encoder_size * self.x_dnn_osz, ),
                    use_batch_norm=self.use_batch_norm
                )
        

        elif which=="decoder": 

            if self.mode_dnn=="FFNet": 
                return FFNet(
                    in_size=self.u_dnn_isz,                        
                    out_size=self.output_dim,                          
                    hidden_size=self.decoder_depth *                     
                    (self.decoder_size * self.u_dnn_isz, ),
                    use_batch_norm=self.use_batch_norm
                ) 

            elif self.mode_dnn=="ConvNet": 
                return ConvNet(
                    in_size=self.u_dnn_isz, 
                    out_size=self.output_dim, 
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="SelfAttention":
                return SelfAttention(
                    input_dim=self.u_dnn_isz,                            
                    output_dim=self.output_dim,                           
                    hidden_dim=self.decoder_depth *                     
                    (self.decoder_size * self.u_dnn_isz, ),
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="ResidualBlock": 
                return ResidualBlock(
                    input_dim=self.u_dnn_isz,                            
                    output_dim=self.output_dim,                           
                    hidden_dim=self.decoder_depth *                     
                    (self.decoder_size * self.u_dnn_isz, ),
                    use_batch_norm=self.use_batch_norm
                )
            
            elif self.mode_dnn=="GRUEncoderDecoder": 
                return GRUEncoderDecoder(
                    input_size=self.u_dnn_isz,                            
                    output_size=self.output_dim,                           
                    hidden_size=self.decoder_depth *                     
                    (self.decoder_size * self.u_dnn_isz, ),
                    use_batch_norm=self.use_batch_norm
                )

    # ----------------------------------------------------------------------- #
    def mode_rnn_new(self, x, deltas, rnn_input):
        h0 = self.x_dnn(x)
        z = torch.cat((x, h0), dim=1) 
        z = z.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)
        c0 = torch.zeros_like(z)

        rnn_out_seq_packed, _, coefficients = self.u_rnn(rnn_input, (z, c0), deltas)    ###############

        return z, rnn_out_seq_packed, coefficients  ###############

    def mode_rnn_gru(self, x, deltas, rnn_input):
        h0 = self.x_dnn(x)
        z = torch.cat((x, h0), dim=1) 
        z = z.unsqueeze(0).expand(self.control_rnn_depth, -1, -1)

        rnn_out_seq_packed, _, coefficients = self.u_rnn(rnn_input, z, deltas)    ###############

        return z, rnn_out_seq_packed, coefficients  ###############

    def mode_rnn_old(self, x, _, rnn_input):
        h0 = self.x_dnn(x)
        h0 = torch.stack(h0.split(self.control_rnn_size, dim=1))
        c0 = torch.zeros_like(h0)

        rnn_out_seq_packed, _ = self.u_rnn(rnn_input, (h0, c0))

        return h0, rnn_out_seq_packed, torch.tensor([[0, 0], [0, 0]], dtype=torch.float32) ###############



# ---------------- Encoder/Decoder ------------------------------------------ #
"""
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
"""


# ---------------- Temp ----------------------------------------------------- #
"""
ConvNet

Traceback (most recent call last):
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 454, in <module>
    else: main(SWEEP)
          ^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 345, in main
    train_loss, coeff = validate(train_dl, loss, model, device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\train.py", line 49, in validate
    y_pred, coefficients = model(x0, u, deltas) ###############
                           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 114, in forward
    h0, rnn_out_seq_packed, coefficients = self.structure_function(x, deltas, rnn_input)    ###############
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 243, in mode_rnn_old
    h0 = self.x_dnn(x)
         ^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\Seq2Seq.py", line 85, in forward
    x = self.conv1(x)
        ^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\conv.py", line 308, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\conv.py", line 304, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Given groups=1, weight of size [64, 2, 3], expected input[1, 256, 2] to have 2 channels, but got 256 channels instead


---------------------------------------------------------------------------------

SelfAttraction

Traceback (most recent call last):
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 454, in <module>
    else: main(SWEEP)
          ^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 289, in main
    model = CausalFlowModel(**model_args)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 86, in __init__
    self.x_dnn = self.seq_to_seq("encoder")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 150, in seq_to_seq
    return SelfAttention(
           ^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\Seq2Seq.py", line 116, in __init__
    self.attn_weights = nn.Parameter(torch.rand(input_dim, hidden_dim))        
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: rand(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got tuple"

---------------------------------------------------------------------------------

ResidualBlock

Traceback (most recent call last):
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 454, in <module>
    else: main(SWEEP)
          ^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 289, in main
    model = CausalFlowModel(**model_args)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 86, in __init__
    self.x_dnn = self.seq_to_seq("encoder")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 159, in seq_to_seq
    return ResidualBlock(
           ^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\Seq2Seq.py", line 153, in __init__
    self.fc1 = nn.Linear(input_dim, hidden_dim)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\linear.py", line 99, in __init__
    self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of:
 * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)


---------------------------------------------------------------------------------

GRUEncoderDecoder

Traceback (most recent call last):
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 454, in <module>
    else: main(SWEEP)
          ^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\experiments\train_wandb.py", line 289, in main
    model = CausalFlowModel(**model_args)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 86, in __init__
    self.x_dnn = self.seq_to_seq("encoder")
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\model.py", line 168, in seq_to_seq
    return GRUEncoderDecoder(
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\g7fie\OneDrive\Documenti\GitHub\flumen-G\src\flumen\Seq2Seq.py", line 194, in __init__
    self.encoder_gru = nn.GRU(input_size, hidden_size, batch_first=True)       
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\rnn.py", line 1078, in __init__
    super().__init__('GRU', *args, **kwargs)
  File "C:\Users\g7fie\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch-2.4.1-py3.12-win-amd64.egg\torch\nn\modules\rnn.py", line 94, in __init__
    raise TypeError(f"hidden_size should be of type int, got: {type(hidden_size).__name__}")
TypeError: hidden_size should be of type int, got: tuple

"""
