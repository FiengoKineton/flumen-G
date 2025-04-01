import torch, sys
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import yaml 
from pathlib import Path
from pprint import pprint
import torch.nn.functional as F
import numpy as np


"""
COMMANDs:

python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/vdp.yaml vdp_test_data
python experiments/train_wandb.py data/vdp_test_data.pkl vdp_test 
"""


# ---------------- LSTM ----------------------------------------------------- #

class LSTM(nn.Module):
    def __init__(self, input_size, z_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, 
                 state_dim=None, discretisation_mode=None, x_update_mode=None, model_name=None): #model_data=None):
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

    # -------------------------------------------
        self.model_name = model_name
        self.data = self.get_model_data()
        self.dtype = torch.float32

        self.param = self.get_dyn_matrix()
        self.I = torch.eye(self.state_dim, dtype=self.dtype)

        """#self.B = torch.tensor([[0], [0]])
        ###pprint(self.data)
        print("dyn matrix:")
        pprint(self.A)       
        print("\ninput matrix:")
        pprint(self.B)
        print("\neq_points:", eq_points, "\n\n")
        ###sys.exit()"""
    
    # -------------------------------------------
        self.discretisation_function = globals().get(f"discretisation_{discretisation_mode}")
        self.x_update_function = globals().get(f"x_update_mode__{x_update_mode}")
        self.linearisation_function = globals().get(f"linearisation_lpv__{self.model_name}")         # lpv__

        print("'lin_mode':", self.linearisation_function)
        print("'dis_mode':", self.discretisation_function)
        print("'upt_mode':", self.x_update_function)

    # -------------------------------------------
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size + state_dim if layer == 0 else self.hidden_size, self.hidden_size, bias)        # torch.jit.script()    
            for layer in range(num_layers)
        ])

    # -------------------------------------------
        self.alpha_gate = nn.Linear(self.hidden_size, self.state_dim, bias=bias)  # Gate function
        torch.nn.init.xavier_uniform_(self.alpha_gate.weight)
        if bias: torch.nn.init.constant_(self.alpha_gate.bias, 0.0)

        self.W__h_to_x = nn.Linear(self.hidden_size, self.state_dim, bias=bias)   # Mapping function
        torch.nn.init.xavier_uniform_(self.W__h_to_x.weight)
        if bias: torch.nn.init.constant_(self.W__h_to_x.bias, 0.0)


    def forward(self, rnn_input: PackedSequence, hidden_state, tau):
        rnn_input_unpacked, lengths = pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        batch_size, seq_len, _ = rnn_input_unpacked.shape
        device = rnn_input_unpacked.device

        z, c_z = hidden_state
        #self.A, self.B = self.A.to(device, dtype=self.dtype), self.B.to(device, dtype=self.dtype)
        self.I, tau = self.I.to(device, dtype=self.dtype), tau.to(device, dtype=self.dtype)

        outputs = torch.empty(batch_size, seq_len, self.z_size, device=device)  # Preallocate tensor | before: torch.zeros
        coefficients = torch.empty(batch_size, seq_len, self.state_dim, device=device)  
        matrices = torch.empty(seq_len, self.state_dim, self.state_dim, device=device)


        for t in range(seq_len):
            rnn_input_t = rnn_input_unpacked[:, t, :]
            tau_t = tau[:, t, :]

            x_prev, c_x = z[:, :, :self.state_dim], c_z[:, :, :self.state_dim]
            h, c = z[:, :, self.state_dim:], c_z[:, :, self.state_dim:]

            # Generalized fix: Ensure proper tensor shape for single and multi-layer cases
            x_in = x_prev.squeeze(0) if x_prev.dim() == 2 else x_prev[-1]  # Take last layer if multi-layer
            u_t = torch.cat((x_in, rnn_input_t), dim=1)

            h_list, c_list = [], []
            for layer, cell in enumerate(self.lstm_cells):
                h_new, c_new = cell(u_t, h[layer], c[layer])
                h_list.append(h_new)
                c_list.append(c_new)

            u_dyn = rnn_input_t[:, :1]
            A_matrix, B_matrix, f_eq = self.linearisation_function(self.param, x_prev, u_dyn)
            x_mid = self.discretisation_function(x_prev, (A_matrix, tau_t, self.I, B_matrix, f_eq), u_dyn)            

            h, c = torch.stack(h_list, dim=0), torch.stack(c_list, dim=0)
            x_next, coeff = self.x_update_function(x_mid, h, self.alpha_gate, self.W__h_to_x)      

            z, c_z = torch.cat((x_next, h), dim=-1), torch.cat((c_x, c), dim=-1)            # same as the old one
            outputs[:, t, :].copy_(z[-1])  # In-place assignment
            coefficients[:, t, :].copy_(coeff)  
            matrices[t, :, :].copy_(A_matrix)

        #if torch.isnan(outputs).any() or torch.isinf(outputs).any(): sys.exit()
        out = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False)
        return out, (z, c_z), coefficients, matrices


    def get_dyn_matrix(self): 
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

            from scipy.optimize import fsolve

            # Solve for v* such that dv/dt = 0 and dw/dt = 0
            def fhn_equilibrium(v):
                w = (v - a) / b
                return v - v**3 - w

            v_star = fsolve(fhn_equilibrium, 0)[0]
            w_star = (v_star - a) / b
            u_star = w_star - (v_star - v_star**3)  # from dv = 0 → u* = w* - v* + v*^3

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
        else: model_ID = ""

        # Define the file path
        file_path = Path(f"data_generation/{model_ID.lower()}.yaml")

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {file_path} not found!")

        # Load the YAML file
        with file_path.open('r') as f:
            model_data = yaml.safe_load(f)
        
        return model_data


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


# --------------------------------------------------------------------------- #
# ---------------- Linearisation functions ---------------------------------- #
# --------------------------------------------------------------------------- #

def linearisation_VanDerPol(param, x, u): 
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    mhu = param['mhu']

    x_sample = x[0, 0]
    u_sample = u[0]

    x1 = x_sample[0] - x1_eq
    x2 = x_sample[1] - x2_eq
    u = u_sample[0] - u_eq

    A = dyn_factor* torch.tensor([[0.0, 1.0],
                        [-1.0 - 2 * mhu * x1 * x2,
                        mhu * (1 - x1**2)]], 
                        dtype=dtype)
    
    B = dyn_factor * torch.tensor([[0.0], [1.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        x2_eq,
        -x1_eq + mhu * (1 - x1_eq**2) * x2_eq + u_eq
    ], dtype=dtype)

    return A, B, f_eq


def linearisation_FitzHughNagumo(param, x, u): 
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    tau = param['tau']
    a = param['a']
    b = param['b']
    v_fact = param['v_fact']

    x_sample = x[0, 0]
    u_sample = u[0]

    x1 = x_sample[0] - x1_eq
    x2 = x_sample[1] - x2_eq
    u = u_sample[0] - u_eq

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
        v_fact * (x1_eq - x1_eq**3 - x2_eq + u_eq),
        (x1_eq - a - b * x2_eq) / tau
    ], dtype=dtype)

    return A, B, f_eq


# ─────────────────────────────────────────────────────────────────────────── #
# ---------------- Linearisation LPV functions ------------------------------ #
# ─────────────────────────────────────────────────────────────────────────── #

def linearisation_lpv__VanDerPol(param, x, u, radius=0.2, epsilon=1e-4):
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    mhu = param['mhu']

    x_target = x[0, 0]
    u_target = u[0]

    # ----------------------------------------------
    B = dyn_factor * torch.tensor([[0.0], [1.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        x2_eq,
        -x1_eq + mhu * (1 - x1_eq**2) * x2_eq + u_eq
    ], dtype=dtype)

    # ----------------------------------------------
    def jacobian_vdp(x):
        x1, x2 = x[0], x[1]
        A = dyn_factor * torch.tensor([
            [0.0, 1.0],
            [-1.0 - 2 * mhu * x1 * x2, mhu * (1 - x1**2)]
        ], dtype=dtype)
        return A
    
    # Define 8 direction vectors (circle-like)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    deltas = torch.tensor([[np.cos(a), np.sin(a)] for a in angles], dtype=dtype)

    # Generate sample points around the origin
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)
    sampled_points = x_eq + radius * deltas  # [8, 2]

    # Compute A_i for each sampled point
    A_list = [jacobian_vdp(xi) for xi in sampled_points]

    # Compute weights k_i = 1 / (||x - xi||^2 + epsilon)
    distances = torch.norm(x_target - sampled_points, dim=1)  # [8]
    weights = 1.0 / (distances**2 + epsilon)  # [8]
    weights = weights / weights.sum()  # normalize

    # Compute weighted sum: A(x) = sum_i A_i * w_i
    A = sum(w * A for w, A in zip(weights, A_list))

    return A, B, f_eq


def linearisation_lpv__FitzHughNagumo(param, x, u, radius=0.2, epsilon=1e-4):
    x1_eq = param['x1_eq']
    x2_eq = param['x2_eq']
    u_eq = param['u_eq']
    dyn_factor = param['dyn_factor']
    dtype = param['dtype']
    tau = param['tau']
    a = param['a']
    b = param['b']
    v_fact = param['v_fact']

    x_target = x[0, 0]
    u_target = u[0]

    # ----------------------------------------------
    B = dyn_factor * torch.tensor([[v_fact], [0.0]], dtype=dtype)

    f_eq = dyn_factor * torch.tensor([
        v_fact * (x1_eq - x1_eq**3 - x2_eq + u_eq),
        (x1_eq - a - b * x2_eq) / tau
    ], dtype=dtype)

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
    
    # Define 8 direction vectors (circle-like)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]
    deltas = torch.tensor([[np.cos(a), np.sin(a)] for a in angles], dtype=dtype)

    # Generate sample points around the origin
    x_eq = torch.tensor([x1_eq, x2_eq], dtype=dtype)
    sampled_points = x_eq + radius * deltas  # [8, 2]

    # Compute A_i for each sampled point
    A_list = [jacobian_fhn(xi) for xi in sampled_points]

    # Compute weights k_i = 1 / (||x - xi||^2 + epsilon)
    distances = torch.norm(x_target - sampled_points, dim=1)  # [8]
    weights = 1.0 / (distances**2 + epsilon)  # [8]
    weights = weights / weights.sum()  # normalize

    # Compute weighted sum: A(x) = sum_i A_i * w_i
    A = sum(w * A for w, A in zip(weights, A_list))

    return A, B, f_eq



# ═══════════════════════════════════════════════════════════════════════════ #
# ---------------- Discretisation functions --------------------------------- #
# ═══════════════════════════════════════════════════════════════════════════ #

def discretisation_FE(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)    # 2, 1).transpose(1, 2)
    x_prev = x_prev.squeeze(0).unsqueeze(1)

    transform_matrix = I + tau * A 
    input_matrix = tau * B.unsqueeze(0).expand(batch_size, -1, -1)
    input_matrix = input_matrix.transpose(1, 2)

    #B_exp = B.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)
    f_eq_exp = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    ev_lib = torch.bmm(x_prev, transform_matrix)
    ev_for = torch.bmm(u, input_matrix)
    f_eq_term = tau * f_eq_exp

    x_next = ev_lib + ev_for + f_eq_term

    return x_next.squeeze(1).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_BE(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)
    f_eq = f_eq.view(-1, 1)

    A_neg = I - tau * A
    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)
    f_eq_exp = f_eq.unsqueeze(0).expand(batch_size, -1, -1)

    u_effect = torch.bmm(B_exp, u)
    u_effect_scaled = tau * (u_effect + f_eq_exp)

    ev_lib = torch.linalg.solve(A_neg, x_prev)
    ev_for = torch.linalg.solve(A_neg, u_effect_scaled)

    """inv_matrix = torch.inverse(A_neg)
    x_next = torch.bmm(x_prev, inv_matrix)"""

    x_next = ev_lib + ev_for
    return x_next.squeeze(2).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_TU(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)

    A_pos = I + (tau / 2) * A
    A_neg = I - (tau / 2) * A

    """v = torch.linalg.solve(A_neg, x_prev)
    x_next = torch.bmm(A_pos, v)"""

    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)
    f_eq_exp = f_eq.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)

    rhs = torch.bmm(A_pos, x_prev) + tau * (torch.bmm(B_exp, u) + f_eq_exp)

    x_next = torch.linalg.solve(A_neg, rhs)
    return x_next.squeeze(2).unsqueeze(0)


def discretisation_RK4(x_prev, mat, u):
    """
    Runge-Kutta 4th order (RK4) discretisation.
    Uses the classical RK4 method to approximate x_next.
    """
    A, tau, _, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)
    A = A.expand(batch_size, -1, -1)

    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)
    f_eq_exp = f_eq.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)

    def f(x): return torch.bmm(x, A) + torch.bmm(u, B_exp) + f_eq_exp

    k1 = f(x_prev)
    k2 = f(x_prev + 0.5 * tau * k1)
    k3 = f(x_prev + 0.5 * tau * k2)
    k4 = f(x_prev + tau * k3)

    x_next = x_prev + (tau / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next.squeeze(1).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_exact(x_prev, mat, u):
    A, tau, I, B, f_eq = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)  # [batch, 1, state]

    # Assuming A is shared (not batched)
    A_exp = torch.matrix_exp(tau[0, 0] * A)  # [2, 2]
    exp_matrix = A_exp.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 2, 2]

    rhs = exp_matrix - I
    integral_term = torch.linalg.solve(A, rhs)  # [2, 2] result, broadcasted

    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 2, 1]

    f_eq = f_eq.view(-1, 1)  # [2, 1]
    f_eq_exp = f_eq.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 2, 1]

    input_term = torch.bmm(B_exp, u) + f_eq_exp  # [batch, 2, 1]

    x1 = torch.bmm(x_prev, exp_matrix)          # [batch, 1, 2]
    x2 = torch.bmm(integral_term.expand(batch_size, -1, -1), input_term)  # [batch, 2, 1]

    x_next = x1 + x2.transpose(1, 2)  # [batch, 1, 2] + [batch, 1, 2]
    return x_next.squeeze(1).unsqueeze(0)  # [1, batch, state]



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
