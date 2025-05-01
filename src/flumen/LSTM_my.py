import torch, sys
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import yaml 
from pathlib import Path
from pprint import pprint
import torch.nn.functional as F
import numpy as np
from scipy.special import expit
from .trajectory import TrajectoryDataset
from torch.utils.data import DataLoader


"""
COMMANDs:

python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/vdp.yaml vdp_test_data
python experiments/train_wandb.py data/vdp_test_data.pkl vdp_test 
"""


# ---------------- LSTM ----------------------------------------------------- #

class LSTM(nn.Module):
    def __init__(self, input_size, z_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, 
                 state_dim=None, discretisation_mode=None, x_update_mode=None, 
                 model_name=None, linearisation_mode=None, batch_size=128, radius=None):
        super(LSTM, self).__init__()

    # -------------------------------------------
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
        self.batch_size = batch_size
        self.radius = radius

        #self.num_directions = 2 if bidirectional else 1

    # -------------------------------------------
        self.model_name = model_name
        self.data = self.get_model_data()
        self.dtype = torch.float32

        self.param = self.get_dyn_matrix_params()
        self.I = torch.eye(self.state_dim, dtype=self.dtype)
    
    # -------------------------------------------
        self.discretisation_function = globals().get(f"discretisation_{discretisation_mode}")
        self.x_update_function = globals().get(f"x_update_mode__{x_update_mode}")

        if linearisation_mode=='static': 
            self.linearisation_function = globals().get(f"linearisation_static__{self.model_name}")
        elif linearisation_mode=='current': 
            self.linearisation_function = globals().get(f"linearisation_curr__{self.model_name}")
        elif linearisation_mode=='lpv': 
            self.linearisation_function = globals().get(f"linearisation_lpv__{self.model_name}")
            print("lpv radius:", self.radius)


        print("'lin_mode':", self.linearisation_function)
        print("'dis_mode':", self.discretisation_function)
        print("'upt_mode':", self.x_update_function)
        print("\n")

    # ------------------------------------------- # torch.jit.script() 
        self.lstm_cells = nn.ModuleList([
                LSTMCell(input_size + state_dim if layer == 0 else self.hidden_size, self.hidden_size, bias)   
            for layer in range(num_layers)
        ]) 
        """if not bidirectional else nn.ModuleList([
            nn.ModuleList([
                bdrLSTMCell(input_size + state_dim if layer == 0 else self.hidden_size * self.num_directions, self.hidden_size, bias)
                for _ in range(self.num_directions)
            ]) for layer in range(num_layers)
        ])"""

    # -------------------------------------------
        """Dropout layer per regolarizzare la rete tra i layer interni della LSTM
        Viene attivato solo se: c'è più di un layer e il dropout è > 0.0
        Aiuta a prevenire l'overfitting spegnendo casualmente dei neuroni a ogni forward pass"""
        #self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()


        """LayerNorm applicato separatamente a ogni layer della LSTM
         Stabilizza la distribuzione delle attivazioni (h_new), normalizzandole rispetto ai feature di ogni step
        Questo migliora la convergenza e riduce l'effetto di vanishing/exploding gradient"""
        #self.ln_layers = nn.ModuleList([nn.LayerNorm(self.hidden_size) for _ in range(num_layers)]) 
        """if not bidirectional else nn.ModuleList([
                nn.ModuleList([
                    nn.LayerNorm(self.hidden_size) for _ in range(self.num_directions)
                ]) for _ in range(num_layers)
            ])#"""

    # -------------------------------------------
        self.alpha_gate = nn.Linear(self.hidden_size, self.state_dim, bias=bias)  # Gate function
        torch.nn.init.xavier_uniform_(self.alpha_gate.weight)
        if bias: torch.nn.init.constant_(self.alpha_gate.bias, 0.0)

        self.W__h_to_x = nn.Linear(self.hidden_size, self.state_dim, bias=bias)   # Mapping function
        torch.nn.init.xavier_uniform_(self.W__h_to_x.weight)
        if bias: torch.nn.init.constant_(self.W__h_to_x.bias, 0.0)

    # -------------------------------------------
        #self.fc = nn.Linear(self.hidden_size * self.num_directions, output_size) if output_size is not None else None



    def forward(self, rnn_input: PackedSequence, hidden_state, tau):
        rnn_input_unpacked, lengths = pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        batch_size, seq_len, _ = rnn_input_unpacked.shape
        device = rnn_input_unpacked.device

        z, c_z = hidden_state
        self.I, tau = self.I.to(device, dtype=self.dtype), tau.to(device, dtype=self.dtype)

        outputs = torch.empty(batch_size, seq_len, self.z_size, device=device)  # Preallocate tensor | before: torch.zeros
        coefficients = torch.empty(batch_size, seq_len, self.state_dim, device=device)  
        matrices = torch.empty(seq_len, self.state_dim, self.state_dim, device=device)
        r_old = self.radius

        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :]
            tau_t = tau[:, t, :]

            x_prev, c_x = z[:, :, :self.state_dim], c_z[:, :, :self.state_dim]
            h, c = z[:, :, self.state_dim:], c_z[:, :, self.state_dim:]

            # Generalized fix: Ensure proper tensor shape for single and multi-layer cases
            x_in = x_prev.squeeze(0) ### if x_prev.dim() == 2 else x_prev[-1]  # Take last layer if multi-layer
            u_t = torch.cat((x_in, rnn_input_t), dim=1)

            h_list, c_list = [], []
            for layer, cell in enumerate(self.lstm_cells):
                h_new, c_new = cell(u_t, h[layer], c[layer])

                #h_new = self.ln_layers[layer](h_new)
                #if layer < self.num_layers - 1: h_new = self.dropout_layer(h_new)

                h_list.append(h_new)
                c_list.append(c_new)

            """h_list, c_list = [], []
            for layer, cells in enumerate(self.lstm_cells):
                new_h, new_c = [], []

                # Forward direction
                h_fwd, c_fwd = cells[0](u_t, h[layer], c[layer])
                h_fwd = self.ln_layers[layer](h_fwd)
                new_h.append(h_fwd)
                new_c.append(c_fwd)

                if self.bidirectional:
                    t_rev = seq_len - t - 1
                    rnn_input_rev = rnn_input_unpacked[:, t_rev, :]
                    x_rev = x_prev.squeeze(0)
                    u_rev = torch.cat((x_rev, rnn_input_rev), dim=1)
                    h_bwd, c_bwd = cells[1](u_rev, h[layer], c[layer])
                    h_bwd = self.ln_layers[layer](h_bwd)
                    new_h.append(h_bwd)
                    new_c.append(c_bwd)
                    u_t = torch.cat([h_fwd, h_bwd], dim=-1)
                else:
                    u_t = h_fwd

                if layer < self.num_layers - 1:
                    u_t = self.dropout_layer(u_t)

                h_list.append(torch.stack(new_h, dim=0))
                c_list.append(torch.stack(new_c, dim=0))"""

            u_dyn = rnn_input_t[:, :1]
            A_matrix, B_matrix, f_eq, r_old = self.linearisation_function(self.param, (batch_size, self.radius, r_old), x_prev, u_dyn)
            x_mid = self.discretisation_function(x_prev, (A_matrix, tau_t, self.I, B_matrix, f_eq), u_dyn)   

            h, c = torch.stack(h_list, dim=0), torch.stack(c_list, dim=0)
            x_next, coeff = self.x_update_function(x_mid, h, self.alpha_gate, self.W__h_to_x)      

            z, c_z = torch.cat((x_next, h), dim=-1), torch.cat((c_x, c), dim=-1)
            outputs[:, t, :].copy_(z[-1])
            coefficients[:, t, :].copy_(coeff)  
            matrices[t, :, :].copy_(A_matrix[0])

            ###print("checkpoint"), sys.exit()     # Debugging

        #if torch.isnan(outputs).any() or torch.isinf(outputs).any(): sys.exit()
        out = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)
        ###if self.fc is not None and not isinstance(outputs, torch.nn.utils.rnn.PackedSequence): out = self.fc(out[:, -1, :])
        return out, (z, c_z), coefficients, matrices


    def get_dyn_matrix_params(self): 
        """
        Dynamics are located in semble/semble/dynamics.py 
        -------------------------------------------------
        Computes the linearized dynamics matrix A and equilibrium points for different systems based on self.data.
        Returns:
            A (torch.Tensor): Linearized system dynamics matrix.
            eq_point (torch.Tensor): Equilibrium state (x*).
        """
        model_name = self.model_name
        dyn_factor = self.data["control_delta"]

        if model_name == "VanDerPol":
            mhu = self.data["dynamics"]["args"]["damping"]
            if self.radius==None: self.radius = 3

            """train_data = TrajectoryDataset(self.data["train"])
            train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            A, B = self.BLA(train_dl)
            print(A)
            print(B)
            sys.exit()"""

            param = {
                'dyn_factor': dyn_factor,
                'dtype': self.dtype,
                'mhu': mhu,  
                'x1_eq': 0.0, 
                'x2_eq': 0.0, 
                'u_eq': 0.0,
            }
            
        elif model_name == "FitzHughNagumo":
            tau = self.data["dynamics"]["args"]["tau"]
            a = self.data["dynamics"]["args"]["a"]
            b = self.data["dynamics"]["args"]["b"]
            v_fact = 50

            if self.radius==None: self.radius = 1

            from scipy.optimize import fsolve

            # Solve for v* such that dv/dt = 0 and dw/dt = 0
            def fhn_equilibrium(v):
                w = (v - a) / b
                return v - v**3 - w

            v_star = fsolve(fhn_equilibrium, 0)[0]
            w_star = (v_star - a) / b
            u_star = 0.0    # w_star - (v_star - v_star**3)  # from dv = 0 → u* = w* - v* + v*^3

            param = {
                'dyn_factor': dyn_factor,
                'dtype': self.dtype,
                'tau': tau, 
                'a': a, 
                'b': b, 
                'v_fact': v_fact, 
                'x1_eq': v_star, 
                'x2_eq': w_star, 
                'u_eq': u_star,
            }

        elif model_name == "NonlinearActivationDynamics": 
            state_dim = self.data["dynamics"]["args"]["state_dim"]
            control_dim = self.data["dynamics"]["args"]["control_dim"]
            activation = self.data["dynamics"]["args"]["activation"]
            mode = self.data["dynamics"]["args"]["mode"]
            a_s = self.data["dynamics"]["args"]["a_s"]
            a_m = self.data["dynamics"]["args"]["a_m"]
            a_b = self.data["dynamics"]["args"]["a_b"]
            b = self.data["dynamics"]["args"]["b"]
            b_b = self.data["dynamics"]["args"]["b_b"]
            
            if self.radius==None: self.radius = 3

            if mode=="stable": 
                a = a_s #if mode=="stable" else a_m
                b = b
            elif mode=="big": 
                a = a_b
                b = b_b
            else: 
                a = a_m
                b = b

            def nad_equilibrium(z): 
                x = z[:state_dim]
                u = z[state_dim:]       

                if activation == "sigmoid":
                    eq_x = - x + expit(a @ x + b @ u)
                else:
                    eq_x = - x + np.tanh(a @ x + b @ u)
                
                eq_u = -u 
                return np.concatenate([eq_x, eq_u])    
            
            from scipy.optimize import fsolve
            eq = fsolve(nad_equilibrium, np.zeros(state_dim+control_dim))  
            x_star, u_star = eq[:state_dim], eq[state_dim:]  # [0.42599762 0.42599282 0.42591131 0.4245284  0.40105814], [0.]

            a = torch.tensor(a, dtype=self.dtype)
            b = torch.tensor(b, dtype=self.dtype)
            x_star = torch.tensor(x_star, dtype=self.dtype)
            
            param = {
                'dyn_factor': dyn_factor,
                'dtype': self.dtype,
                'state_dim': state_dim, 
                'control_dim': control_dim, 
                'activation': activation, 
                'mode': mode, 
                'A': a, 
                'B': b,
                'x_eq': x_star,
                'u_eq': u_star,
            }

        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        #print(A.shape[0])
        return param

    def get_model_data(self): 
        """
        Loads model-specific data from YAML files located in 'data_generation/'.
        The file name is determined by self.model_name.
        
        Returns:
            dict: Parsed YAML data containing settings for the model.
        """

        if self.model_name == "VanDerPol": model_ID = "vdp"
        elif self.model_name == "FitzHughNagumo": model_ID = "fhn"
        elif self.model_name == "GreenshieldsTraffic": model_ID = "greenshields"
        elif self.model_name == "HodgkinHuxleyFFE": model_ID = "hhffe"
        elif self.model_name == "HodgkinHuxleyFS": model_ID = "hhfs"
        elif self.model_name == "LinearSys": model_ID = "linsys"
        elif self.model_name == "TwoTank": model_ID = "twotank"
        elif self.model_name == "NonlinearActivationDynamics": model_ID = "nad"
        else: model_ID = ""

        # Define the file path
        file_path = Path(f"data_generation/{model_ID.lower()}.yaml")

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {file_path} not found!")

        # Load the YAML file
        with file_path.open('r') as f:
            model_data = yaml.safe_load(f)
        
        return model_data

    def BLA(self, train_dl): 
        all_u, all_y = [], []

        for batch in train_dl:
            _, y, u, lengths = batch

            for i in range(y.shape[0]):
                seq_len = lengths[i]
                u_i = u[i, :seq_len]  # (seq_len, control_dim)
                y_i = y[i, :seq_len]  # (seq_len, output_dim)

                all_u.append(u_i)
                all_y.append(y_i)

        u_all = torch.cat(all_u, dim=0)  # (N, control_dim)
        y_all = torch.cat(all_y, dim=0)  # (N, output_dim)

        u_tilde = u_all - u_all.mean(dim=0, keepdim=True)
        y_tilde = y_all - y_all.mean(dim=0, keepdim=True)

        def estimate_bla_fir(u_tilde, y_tilde, n_order=self.state_dim):
            N = u_tilde.shape[0] - n_order

            X = torch.stack([u_tilde[i:N+i] for i in range(n_order)], dim=2)  # shape: (N, control_dim, n_order)
            X = X.reshape(N, -1)  # flatten for regression: (N, control_dim * n_order)

            Y = y_tilde[n_order:]  # shape: (N, output_dim)

            # Risolvi Y = X @ theta con least squares
            G_bla, _ = torch.lstsq(Y, X)
            return G_bla[:X.shape[1]]  # (control_dim * n_order, output_dim)

        G_bla = estimate_bla_fir(u_tilde, y_tilde, n_order=10)
        print("G_bla shape:", G_bla.shape)
        print("G_bla:", G_bla)

        return G_bla[:2], G_bla[2:]  # A, B matrices



# ---------------- LSTMCell ------------------------------------------------- #

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size

        self.WU = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        torch.nn.init.xavier_uniform_(self.WU.weight)
        if bias: torch.nn.init.constant_(self.WU.bias, 0.0)

    def forward(self, u, h, c):
        gates = self.WU(torch.cat((u, h), dim=1))  # Single matrix multiplication
        i, f, g, o = gates.chunk(4, dim=1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c.mul_(f).add_(i * g)               # c_next = f * c + i * g            | old
        h = o * torch.tanh(c)               # h_next = o * torch.tanh(c_next)   | old

        return h, c

"""
class bdrLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, use_ln: bool=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.WU = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        torch.nn.init.xavier_uniform_(self.WU.weight)
        if bias:
            torch.nn.init.constant_(self.WU.bias, 0.0)

        self.use_ln = use_ln
        if use_ln:
            self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, u, h, c):
        gates = self.WU(torch.cat((u, h), dim=1))
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c.mul_(f).add_(i * g)
        h = o * torch.tanh(c)

        if self.use_ln:
            h = self.layernorm(h)

        return h, c
"""

# --------------------------------------------------------------------------- #
# ---------------- Linearisation static ------------------------------------- #
# --------------------------------------------------------------------------- #

def linearisation_static__VanDerPol(param, const, x, u):                                                # --nope--
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    mhu = param['mhu']

    batch_size, _, _ = const

    A = dyn_factor* torch.tensor([[0.0, 1.0],
                        [-1.0 - 2 * mhu * x1_eq * x2_eq,
                        mhu * (1 - x1_eq**2)]], 
                        dtype=dtype)
    
    B = dyn_factor * torch.tensor([[0.0], [1.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        [x2_eq],
        [-x1_eq + mhu * (1 - x1_eq**2) * x2_eq + u_eq]
    ], dtype=dtype)

    A = A.unsqueeze(0).expand(batch_size, -1, -1) 
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 
    f_eq = f_eq.unsqueeze(0).expand(batch_size, -1, -1)
    return A, B, f_eq, None

def linearisation_static__FitzHughNagumo(param, const, x, u):                                           # --nope--
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    tau = param['tau']
    a = param['a']
    b = param['b']
    v_fact = param['v_fact']

    batch_size, _, _ = const

    v = x1_eq
    w = x2_eq
    u_val = u_eq

    df_dv = v_fact * (1 - 3 * v**2)
    df_dw = -v_fact
    dg_dv = 1 / tau
    dg_dw = -b / tau

    A = dyn_factor * torch.tensor([[df_dv, df_dw],
                    [dg_dv, dg_dw]], dtype=dtype)

    B = dyn_factor * torch.tensor([[v_fact], [0.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        [v_fact * (x1_eq - x1_eq**3 - x2_eq + u_eq)],
        [(x1_eq - a - b * x2_eq) / tau]
    ], dtype=dtype)

    A = A.unsqueeze(0).expand(batch_size, -1, -1) 
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 
    f_eq = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    return A, B, f_eq, None

def linearisation_static__NonlinearActivationDynamics(param, const, x, u):                              ### USE THIS!
    A = param['A']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    activation = param['activation']
    B = param['B']
    x_eq = param['x_eq']
    u_eq = param['u_eq']
    state_dim = param['state_dim']

    batch_size, _, _ = const

    def sigma_prime(z): 
        sigma = 1 / (1 + torch.exp(-z))
        return sigma * (1 - sigma)
    
    def sigma(z): 
        return 1 / (1 + torch.exp(-z))

    z = A @ x_eq + B @ u_eq
    f_eq = -x_eq + sigma(z) # before, wrongly, i wrote -z /// #if activation == "sigmoid" else -x_eq + np.tanh(z)
    S = torch.diag(sigma_prime(z))
    S = torch.tensor(S, dtype=dtype)
    A = -torch.eye(state_dim) + S @ A
    B = S @ B

    A = dyn_factor * torch.tensor(A, dtype=dtype)
    B = dyn_factor * torch.tensor(B, dtype=dtype)
    f_eq = dyn_factor * torch.tensor(f_eq, dtype=dtype).unsqueeze(-1)

    A = A.unsqueeze(0).expand(batch_size, -1, -1) 
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 
    f_eq = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    """
    print(f"state_dim: {state_dim}")
    print("A:", A.shape)
    print("B:", B.shape)
    print("f_eq:", f_eq.shape)
    sys.exit() #"""
    return A, B, f_eq, None


# ─────────────────────────────────────────────────────────────────────────── #
# ---------------- Linearisation functions ---------------------------------- #
# ─────────────────────────────────────────────────────────────────────────── #

def linearisation_curr__VanDerPol(param, const, x, u):                                                  # --nope--
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    mhu = param['mhu']

    batch_size, _, _ = const

    def f_eq_vdp(x):
        x1 = x[:, 0] - x1_eq
        x2 = x[:, 1] - x2_eq

        f1 = x2
        f2 = -x1 + mhu * (1 - x1**2) * x2 + u_eq

        f_eq = dyn_factor * torch.stack([f1, f2], dim=1)  
        return f_eq.to(dtype=dtype).unsqueeze(-1)  

    def jacobian_vdp(x):
        x1 = x[:, 0] - x1_eq
        x2 = x[:, 1] - x2_eq

        a11 = torch.zeros_like(x1)                     
        a12 = torch.ones_like(x1)                       
        a21 = -1.0 - 2 * mhu * x1 * x2
        a22 = mhu * (1 - x1**2)

        A = dyn_factor * torch.stack([
            torch.stack([a11, a12], dim=1),   
            torch.stack([a21, a22], dim=1)   
        ], dim=1) 

        return A.to(dtype=dtype)  

    x = torch.randn(batch_size, 2)  
    A = jacobian_vdp(x)             
    f_eq = f_eq_vdp(x)              
    B = dyn_factor * torch.tensor([[0.0], [1.0]], dtype=dtype)
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 

    return A, B, f_eq, None

def linearisation_curr__FitzHughNagumo(param, const, x, u):                                             # --nope--
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    tau = param['tau']
    a = param['a']
    b = param['b']
    v_fact = param['v_fact']

    batch_size, _, _ = const

    x_sample = x[0, 0]
    u_sample = u[0]

    v = x_sample[0] - x1_eq
    w = x_sample[1] - x2_eq
    u_val = u_sample[0] - u_eq

    df_dv = v_fact * (1 - 3 * v**2)
    df_dw = -v_fact
    dg_dv = 1 / tau
    dg_dw = -b / tau

    A = dyn_factor * torch.tensor([[df_dv, df_dw],
                    [dg_dv, dg_dw]], dtype=dtype)

    B = dyn_factor * torch.tensor([[v_fact], [0.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        [v_fact * (x1_eq - x1_eq**3 - x2_eq + u_eq)],
        [(x1_eq - a - b * x2_eq) / tau]
    ], dtype=dtype)

    A = A.unsqueeze(0).expand(batch_size, -1, -1) 
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 
    f_eq = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    return A, B, f_eq, None

def linearisation_curr__NonlinearActivationDynamics(param, const, x, u):                                # --nope--
    A = param['A']
    B = param['B']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    activation = param['activation']
    x_eq = param['x_eq']
    u_eq = param['u_eq']
    state_dim = param['state_dim']

    batch_size, _, _ = const

    x_sample = x[0, 0]  # dimensione: [state_dim]
    u_sample = u[0]     # dimensione: [control_dim]

    x_, u_ = x_sample - x_eq, u_sample - u_eq
    x_, u_ = torch.tensor(x_, dtype=dtype), torch.tensor(u_, dtype=dtype)

    def sigma_prime(z): 
        if activation == "sigmoid":
            sigma = 1 / (1 + torch.exp(-z))
            return sigma * (1 - sigma)
        else:
            return 1 - torch.tanh(z) ** 2

    z = A @ x_ + B @ u_
    S = torch.diag(sigma_prime(z))
    S = torch.tensor(S, dtype=dtype)

    A_dyn = -torch.eye(state_dim) + S @ A
    B_dyn = S @ B
    f_eq = -x_ + (1 / (1 + torch.exp(-z))) #if activation == "sigmoid" else -x_eq + torch.tanh(A @ x_eq + B @ u_eq)

    A = dyn_factor * torch.tensor(A_dyn, dtype=dtype)
    B = dyn_factor * torch.tensor(B_dyn, dtype=dtype)
    f_eq = dyn_factor * torch.tensor(f_eq, dtype=dtype).unsqueeze(-1)

    A = A.unsqueeze(0).expand(batch_size, -1, -1) 
    B = B.unsqueeze(0).expand(batch_size, -1, -1) 
    f_eq = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    """print("A:", A)
    print("B:", B)
    print("f_eq:", f_eq)
    sys.exit()"""
    return A, B, f_eq, None


# ─────────────────────────────────────────────────────────────────────────── #
# ---------------- Linearisation LPV functions ------------------------------ #
# ─────────────────────────────────────────────────────────────────────────── #

def linearisation_lpv__VanDerPol(param, const, x, u, epsilon=1e-4):                                     ### USE THIS!
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    mhu = param['mhu']

    batch_size, radius, _ = const

    batch_size = u.shape[0]
    x_target = x[0].unsqueeze(1)    # [1, 128, 2] -> [128, 1, 2]
    u_target = u                    # [128, 1]
    
    # ----------------------------------------------
    B = dyn_factor * torch.tensor([[0.0], [1.0]], dtype=dtype)
    B = B.unsqueeze(0).expand(batch_size, -1, -1)       # Size | [128, 2, 1]

    def f_eq_vdp(x): 
        x1, x2 = x[0], x[1]
        f_eq = dyn_factor * torch.tensor([
            x2,
            -x1 + mhu * (1 - x1**2) * x2 + u_eq
        ], dtype=dtype)
        return f_eq

    # ----------------------------------------------
    def jacobian_vdp(x):
        x1, x2 = x[0], x[1]
        A = dyn_factor * torch.tensor([
            [0.0, 1.0],
            [-1.0 - 2 * mhu * x1 * x2, mhu * (1 - x1**2)]
        ], dtype=dtype)
        return A
    
    #-- Define 8 direction vectors (circle-like)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    deltas = torch.tensor([[np.cos(a), np.sin(a)] for a in angles], dtype=dtype)

    #-- Generate sample points around the origin
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)
    sampled_points = x_eq + radius * deltas  # [8, 2]

    #-- Compute A_i for each sampled point
    A_list = [jacobian_vdp(xi) for xi in sampled_points]
    A_list = torch.stack(A_list, dim=0)  # [8, 2, 2]
    A_list = A_list.unsqueeze(0).expand(batch_size, -1, -1, -1) 
    # Size | torch.Size([128, 8, 2, 2])

    f_eq_list = [f_eq_vdp(xi) for xi in sampled_points]
    f_eq_list = torch.stack(f_eq_list, dim=0)  # [8, 2, 1]
    f_eq_list = f_eq_list.unsqueeze(0).expand(batch_size, -1, -1) 
    # Size | torch.Size([128, 8, 2])

    #-- Compute weights k_i = 1 / (||x - xi||^2 + epsilon)
    sampled_points = sampled_points.unsqueeze(0).expand(batch_size, -1, -1) # Size | [128, 8, 2]
    distances = torch.norm(x_target - sampled_points, dim=2)  # [8]
    weights = torch.exp(-(distances**2+epsilon))    # before | weights = 1.0 / (distances**2 + epsilon)  # [8]
    weights = weights / weights.sum()  # normalize, Size | [128, 8]

    #-- Compute weighted sum: A(x) = sum_i A_i * w_i
    w_A = weights.unsqueeze(-1).unsqueeze(-1)   # Size | [128, 8, 1, 1]
    w_f = weights.unsqueeze(-1)                 # Size | [128, 8, 1]

    A = torch.sum(w_A * A_list, dim=1)                      # Size; before | [128, 2, 2]; for w, A in zip(weights, A_list))
    f_eq = torch.sum(w_f * f_eq_list, dim=1).unsqueeze(-1)  # Size; before | [128, 2, 1]; for w, f_eq in zip(weights, f_eq_list))

    """# ----------------- PLOTTING -----------------
    import matplotlib.pyplot as plt
    print("A:\n", A[0], "\n\nA_eq:\n", jacobian_vdp(x_eq))

    # Use CPU and detach for plotting
    #sampled_np = sampled_points[0].cpu().detach().numpy()
    x_target_np = x_target.cpu().detach().squeeze(1).numpy()
    x_eq_np = np.array([x1_eq, x2_eq])

    # Create a smooth circle around x_eq
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = x_eq_np[0] + radius * np.cos(theta)
    circle_y = x_eq_np[1] + radius * np.sin(theta)

    # Inside stability shape
    distances = np.linalg.norm(x_target_np - x_eq_np, axis=1)
    inside_mask = distances <= radius
    percent_inside = 100.0 * np.sum(inside_mask) / len(x_target_np)
    print(f"{percent_inside:.2f}% of the x_target points are inside the circle.")

    plt.figure(figsize=(6, 6))
    plt.plot(circle_x, circle_y, color='red', linestyle='--', label='Sampling Circle')
    plt.scatter(x_target_np[:, 0], x_target_np[:, 1], color='blue', s=10, label='x_target')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('x_target and Sampling Region')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    # -------------------------------------------"""

    return A, B, f_eq, radius

def linearisation_lpv__FitzHughNagumo(param, const, x, u, epsilon=1e-4, alpha=0.0, adp=0):              ### USE THIS!
    x1_eq = param['x1_eq'] + 1.0*0
    x2_eq = param['x2_eq'] - 0.5*0
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    tau = param['tau']
    a = param['a']
    b = param['b']
    v_fact = param['v_fact']

    batch_size, rr, r_old = const

    #batch_size = u.shape[0]
    x_target = x[0].unsqueeze(1)
    u_target = u
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)

    # ----------------------------------------------
    B = dyn_factor * torch.tensor([[v_fact], [0.0]], dtype=dtype)
    B = B.unsqueeze(0).expand(batch_size, -1, -1)

    def f_eq_fhn(x): 
        v, w = x[0], x[1]
        f_eq = dyn_factor * torch.tensor([
            v_fact * (v - v**3 - w + u_eq),
            (v - a - b * w) / tau
        ], dtype=dtype)
        return f_eq

    # ----------------------------------------------
    def jacobian_fhn(x):
        v, w = x[0], x[1]
        df_dv = v_fact * (1 - 3 * v**2)
        df_dw = -v_fact
        dg_dv = 1 / tau
        dg_dw = -b / tau

        A = dyn_factor * torch.tensor([[df_dv, df_dw],
                        [dg_dv, dg_dw]], dtype=dtype)
        return A
    
    """# Circle or radius r
    
    # Define 8 direction vectors (circle-like)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    deltas = torch.tensor([[np.cos(a), np.sin(a)] for a in angles], dtype=dtype)

    # Generate sample points around the origin
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)
    sampled_points = x_eq + radius * deltas  # [8, 2]"""

    """# Elipse

    # Define 8 direction vectors (oval-like)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    deltas = torch.tensor([
        [radius * np.cos(a), (radius * 0.8/3) * np.sin(a)] for a in angles
    ], dtype=dtype)

    # Generate sample points around x_eq
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)
    sampled_points = x_eq + deltas  # [8, 2]"""

    from sklearn.decomposition import PCA

    # 1. Compute PCA over x_target
    x_target_np = x_target.squeeze(1).cpu().detach().numpy()
    pca = PCA(n_components=2)
    pca.fit(x_target_np)

    # === Adatta il raggio in base alla distanza media ===
    def adaptive_radius(distances):
        """
        Decide il raggio da usare in base a una metrica robusta (percentile 90).
        """
        robust_dist = np.percentile(distances, 90)  # Oppure mean + std

        if robust_dist > 3.0:       return 3.5
        elif robust_dist > 2.5:     return 3.0
        elif robust_dist > 1.5:     return 2.5
        else:                       return 2.0
    
    distances = np.linalg.norm(x_target_np - np.mean(x_target_np, axis=0), axis=1)
    r = adaptive_radius(distances)
    radius = (alpha * r_old + (1-alpha) * r) * adp + (1-adp) * rr
    radius = np.clip(radius, 1.5, 3.5)

    # 2. Get ellipse axes and center
    center = pca.mean_
    axes = np.sqrt(pca.explained_variance_) * radius  # Scaled by radius
    basis = pca.components_

    # 3. Sample 8 points around the ellipse using PCA basis
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    ellipse_points = np.array([
        center + axes[0]*np.cos(a)*basis[0] + axes[1]*np.sin(a)*basis[1]
        for a in angles
    ], dtype=np.float32)
    sampled_points = torch.tensor(ellipse_points, dtype=dtype)


    # Compute A_i and f_i for each sampled point
    A_list = [jacobian_fhn(xi) for xi in sampled_points]
    A_list = torch.stack(A_list, dim=0)
    A_list = A_list.unsqueeze(0).expand(batch_size, -1, -1, -1) 

    f_eq_list = [f_eq_fhn(xi) for xi in sampled_points]
    f_eq_list = torch.stack(f_eq_list, dim=0)  # [8, 2, 1]
    f_eq_list = f_eq_list.unsqueeze(0).expand(batch_size, -1, -1) 

    # Compute weights k_i = 1 / (||x - xi||^2 + epsilon)
    sampled_points = sampled_points.unsqueeze(0).expand(batch_size, -1, -1)
    distances = torch.norm(x_target - sampled_points, dim=2)
    weights = torch.exp(-(distances**2+epsilon))    # old | #weights = 1.0 / (distances**2 + epsilon)
    weights = weights / weights.sum()  # normalize

    # Compute weighted sum: A(x) = sum_i A_i * w_i
    w_A = weights.unsqueeze(-1).unsqueeze(-1)
    w_f = weights.unsqueeze(-1)           

    A = torch.sum(w_A * A_list, dim=1)                          # A = sum(w * A for w, A in zip(weights, A_list))
    f_eq = torch.sum(w_f * f_eq_list, dim=1).unsqueeze(-1)      # f_eq = sum(w * f_eq for w, f_eq in zip(weights, f_eq_list))

    
    
    """# ----------------- PLOTTING -----------------
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    # Convert center and PCA info
    center = pca.mean_
    width, height = 2 * axes  # Diametri (2 * semiassi)
    angle = np.degrees(np.arctan2(basis[0, 1], basis[0, 0]))  # Orientamento

    # Dati di input
    x = x_target_np[:, 0]
    y = x_target_np[:, 1]

    # Maschera dei punti dentro l'ellisse (metodo analitico)
    x0, y0 = center
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    dx, dy = x - x0, y - y0
    xp = cos_a * dx + sin_a * dy
    yp = -sin_a * dx + cos_a * dy
    inside_mask = (xp / (width / 2))**2 + (yp / (height / 2))**2 <= 1
    percent_inside = 100.0 * np.sum(inside_mask) / len(x)

    print(f"{percent_inside:.2f}% of the x_target points are inside the PCA-based ellipse.")

    # Plot PCA ellipse and points
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=10, color='blue', label='x_target')
    ax.scatter(x0, y0, s=30, color='green', label='PCA center')
    ax.scatter(x1_eq, x2_eq, s=30, color='red', label='x_eq')

    ellipse_patch = Ellipse(xy=center, width=width, height=height, angle=angle,
                            edgecolor='red', fc='None', lw=2, linestyle='--', label='PCA Ellipse')
    ax.add_patch(ellipse_patch)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('x_target and PCA-based LPV Region')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------
    print(f"↪ Adapting radius to r = {radius:.1f} based on robust_dist = {np.percentile(distances, 90):.2f}")

    # Se i punti blu (distanze medie) sono sempre sotto la linea rossa, significa che l’ellisse è "abbastanza grande" da coprire bene la regione di interesse.
    # Se i punti salgono sopra, potrebbe voler dire che:
    # l’ellisse è troppo piccola rispetto alla dispersione attuale,
    # oppure che i x_target si stanno allontanando e servirebbe un raggio più grande.
    #L’effetto dello smoothing è visibile come una variazione più graduale della linea rossa nel tempo.
    
    mean_distances = []
    sampled_points = sampled_points.squeeze(1).cpu().detach().numpy()

    for batch in x_target_np:
        batch = np.array(batch)
        dists = np.linalg.norm(batch - sampled_points, axis=1)
        mean_distances.append(np.mean(dists))

    plt.figure(figsize=(10, 4))
    plt.plot(mean_distances, marker='o', color='royalblue')
    #plt.axhline(y=2.0, color='orange', linestyle='--', label='r = 2')
    #plt.axhline(y=3.0, color='red', linestyle='--', label='r = 3')
    plt.axhline(y=radius, color='red', linestyle='--', label=f'r = {radius}')
    plt.title('Mean Distance of x_target from Reference Point')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Distance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nr_curr: {radius}; r: {r}; r_old: {r_old}")
    # -------------------------------------------"""
    
    return A, B, f_eq, radius

def linearisation_lpv__NonlinearActivationDynamics(param, const, x, u, epsilon=1e-4):                   # --nope--
    A_base = param['A']
    B_base = param['B']
    x_eq = param['x_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    activation = param['activation']

    batch_size, radius, _ = const

    #batch_size = u.shape[0]
    state_dim = param['state_dim']
    I = torch.eye(state_dim, dtype=dtype)

    def sigma_prime(z): 
        if activation == "sigmoid":
            sigma = 1 / (1 + torch.exp(-z))
            return sigma * (1 - sigma)
        else:
            return 1 - torch.tanh(z) ** 2

    def compute_jacobian(x_sample, u_sample):
        z = A_base @ x_sample + B_base @ u_sample
        S = torch.diag(sigma_prime(z))
        S = torch.tensor(S, dtype=dtype)
        A_dyn = dyn_factor * (-I+ S @ A_base)
        B_dyn = dyn_factor * (S @ B_base)
        return A_dyn, B_dyn

    def compute_f_eq(x_sample, u_sample):
        z = A_base @ x_sample + B_base @ u_sample
        if activation == "sigmoid":
            return dyn_factor * (-x_sample + expit(z))
        else:
            return dyn_factor * (-x_sample + torch.tanh(z))

    # Generate sampling points
    num_samples = 32  # o 64 se vuoi più precisione
    deltas = torch.randn(num_samples, state_dim, dtype=dtype)
    deltas = deltas / torch.norm(deltas, dim=1, keepdim=True)

    x_eq_t = torch.tensor(x_eq, dtype=dtype)
    sampled_points = x_eq_t + radius * deltas  # [num_samples, state_dim]

    # Expand dimensions
    x_target = x[0].unsqueeze(1)  # [batch, 1, state_dim]
    u_target = u                 # [batch, control_dim]
    
    A_list = [compute_jacobian(xi, u_eq)[0] for xi in sampled_points]
    B_list = [compute_jacobian(xi, u_eq)[1] for xi in sampled_points]
    f_eq_list = [compute_f_eq(xi, u_eq) for xi in sampled_points]

    A_list = torch.stack(A_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    B_list = torch.stack(B_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    f_eq_list = torch.stack(f_eq_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1)

    sampled_points = sampled_points.unsqueeze(0).expand(batch_size, -1, -1)
    distances = torch.norm(x_target - sampled_points, dim=2)
    weights = torch.exp(-(distances ** 2 + epsilon))
    weights = weights / weights.sum(dim=1, keepdim=True)

    w_m = weights.unsqueeze(-1).unsqueeze(-1)
    w_f = weights.unsqueeze(-1)

    A = torch.sum(w_m * A_list, dim=1)
    B = torch.sum(w_m * B_list, dim=1)
    f_eq = torch.sum(w_f * f_eq_list, dim=1).unsqueeze(-1)

    A = torch.tensor(A, dtype=dtype)
    B = torch.tensor(B, dtype=dtype)
    f_eq = torch.tensor(f_eq, dtype=dtype)

    """print("A:", A[0])
    print("B:", B[0])
    print("f_eq:", f_eq[0])
    sys.exit()"""
    return A, B, f_eq, radius


# ═══════════════════════════════════════════════════════════════════════════ #
# ---------------- Discretisation functions --------------------------------- #
# ═══════════════════════════════════════════════════════════════════════════ #

def discretisation_FE(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)

    transform_matrix = I + tau * A 
    input_matrix = tau * B
    input_matrix = input_matrix.transpose(1, 2)

    ev_lib = torch.bmm(x_prev, transform_matrix)
    ev_for = torch.bmm(u, input_matrix)
    f_eq_term = (tau * f_eq).transpose(1, 2)

    x_next = ev_lib + ev_for + f_eq_term
    return x_next.squeeze(1).unsqueeze(0)

def discretisation_BE(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)

    A_neg = I - tau * A
    u_effect = torch.bmm(B, u)
    u_effect_scaled = tau * (u_effect + f_eq)

    ev_lib = torch.linalg.solve(A_neg, x_prev)
    ev_for = torch.linalg.solve(A_neg, u_effect_scaled)

    x_next = ev_lib + ev_for
    return x_next.squeeze(2).unsqueeze(0)

def discretisation_TU(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)

    A_pos = I + (tau / 2) * A
    A_neg = I - (tau / 2) * A

    rhs = torch.bmm(A_pos, x_prev) + tau * (torch.bmm(B, u) + f_eq)
    x_next = torch.linalg.solve(A_neg, rhs)
    return x_next.squeeze(2).unsqueeze(0)


def discretisation_RK4(x_prev, mat, u):
    """
    Runge-Kutta 4th order (RK4) discretisation.
    Uses the classical RK4 method to approximate x_next.
    """
    A, tau, _, B, f_eq = mat
    batch_size = tau.shape[0]
    B = B.transpose(1, 2)
    f_eq = f_eq.transpose(1, 2)

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)


    def f(x): return torch.bmm(x, A) + torch.bmm(u, B) + f_eq

    k1 = f(x_prev)
    k2 = f(x_prev + 0.5 * tau * k1)
    k3 = f(x_prev + 0.5 * tau * k2)
    k4 = f(x_prev + tau * k3)

    x_next = x_prev + (tau / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next.squeeze(1).unsqueeze(0)

def discretisation_exact(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)

    exp_matrix = torch.matrix_exp(tau * A)
    rhs = exp_matrix - I
    integral_term = torch.linalg.solve(A, rhs)  # torch.linalg.solve: (Batch element 40): The solver failed because the input matrix is singular.
    input_term = torch.bmm(B, u) + f_eq

    x1 = torch.bmm(x_prev, exp_matrix)
    x2 = torch.bmm(integral_term.expand(batch_size, -1, -1), input_term)

    x_next = x1 + x2.transpose(1, 2)
    return x_next.squeeze(1).unsqueeze(0)



# ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈ #
# ---------------- x_update_mode -------------------------------------------- #
# ≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈ #

def x_update_mode__alpha(x_mid, h, alpha_gate, W__h_to_x):      # GOOD results, balanced
    """
    Alpha-based update rule (Sigmoid function, bounded between 0 and 1).
    
    - **alpha = 0** → `x_next = x_mid` (relies entirely on past dynamics).
    - **alpha = 0.5** → Equal mix of `x_mid` and `W__h_to_x(h[-1])`.
    - **alpha = 1** → `x_next = W__h_to_x(h[-1])` (fully determined by learned influence).
    
    This means:
    - When **alpha is low**, the system relies on past dynamics.
    - When **alpha is high**, the system is heavily influenced by the learned transformation.
    """
    alpha = torch.sigmoid(alpha_gate(h[-1]))
    x_next = (1 - alpha) * x_mid + alpha * W__h_to_x(h[-1])
    return x_next, alpha    ###############

def x_update_mode__beta(x_mid, h, alpha_gate, W__h_to_x):       # good but beta can get negative
    """
    Beta-based update rule (Tanh function, bounded between -1 and 1).
    
    - **beta = -1** → Strong reversal: `x_next = -x_mid + 2 * W__h_to_x(h[-1])`.
    - **beta = 0** → `x_next = W__h_to_x(h[-1])` (ignores past dynamics).
    - **beta = 1** → `x_next = x_mid` (fully follows past dynamics).
    
    This means:
    - When **beta is near -1**, past dynamics are reversed, leading to strong corrective behavior.
    - When **beta is near 1**, past dynamics dominate.
    - When **beta is near 0**, the update is fully controlled by `h[-1]`.
    """
    beta = torch.tanh(alpha_gate(h[-1]))
    x_next = beta * x_mid + (1 - beta) * W__h_to_x(h[-1])
    return x_next, beta ###############

def x_update_mode__lamda(x_mid, h, alpha_gate, W__h_to_x):      # not that efficient
    """
    Lambda-based update rule (Adaptive scaling, bounded between ~0.1 and 0.9).
    
    - **lambda ≈ 0.1** → `x_next` mostly determined by `W__h_to_x(h[-1])` (learned influence dominates).
    - **lambda ≈ 0.5** → Equal contribution from `x_mid` and `W__h_to_x(h[-1])`.
    - **lambda ≈ 0.9** → `x_next` mostly follows past dynamics.
    
    This means:
    - When **x_prev is large**, lambda is high → the system follows past dynamics.
    - When **h[-1] is large**, lambda is low → the system relies on learned influence.
    """
    x_norm = torch.norm(x_mid, dim=-1, keepdim=True).clamp_min(1e-5)
    h_norm = torch.norm(h[-1], dim=-1, keepdim=True).clamp_min(1e-5)

    lambda_factor = x_norm / (x_norm + h_norm).clamp(0.1, 0.9)
    x_next = lambda_factor * x_mid + (1 - lambda_factor) * W__h_to_x(h[-1])
    return x_next, lambda_factor    ###############


def x_update_mode__relu(x_mid, h, alpha_gate, W__h_to_x):       # coeff super small
    """
    ReLU-based gate: values above 0 are passed, below 0 are zeroed.
    
    - The more activated h[-1] is (positively), the more it influences x_next.
    - Acts like a sparse activation gating — only strongly activated features influence the output.
    """
    gate = F.relu(alpha_gate(h[-1]))
    gate = gate / (gate + 1e-5)  # Normalize for safety, values in (0, 1)
    x_next = (1 - gate) * x_mid + gate * W__h_to_x(h[-1])
    return x_next, gate

def x_update_mode__switch(x_mid, h, alpha_gate, W__h_to_x):     # coeff either 0 or 1
    """
    Hard switch: Uses a threshold to select between x_mid and transformed input.
    
    - If activation > 0 → rely on learned influence.
    - Else → follow past dynamics.

    This is like a "hard attention" — can simulate decision boundaries.
    """
    thresholded = (alpha_gate(h[-1]) > 0).float()  # Binary mask
    x_next = (1 - thresholded) * x_mid + thresholded * W__h_to_x(h[-1])
    return x_next, thresholded

def x_update_mode__entropy(x_mid, h, alpha_gate, W__h_to_x):    # GOOD
    """
    Gate based on entropy of softmax over hidden state projection.
    
    - High entropy → uncertain → favor past (x_mid).
    - Low entropy → confident → favor W(h).

    This one is smart when you want uncertainty to drive conservatism.
    """
    logits = alpha_gate(h[-1])
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)  # shape [batch, 1]
    entropy = torch.sigmoid(entropy)  # squash to (0, 1)
    
    gate = 1 - entropy  # High entropy = rely on x_mid
    x_next = gate * x_mid + (1 - gate) * W__h_to_x(h[-1])
    return x_next, gate
