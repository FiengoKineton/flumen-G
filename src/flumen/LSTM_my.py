import torch, sys
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import yaml 
from pathlib import Path
from pprint import pprint
import torch.nn.functional as F


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

        self.A, self.B, eq_points = self.get_dyn_matrix()
        self.I = torch.eye(self.A.shape[0])
        self.dtype = torch.float32

        #self.B = torch.tensor([[0], [0]])
        ###pprint(self.data)
        print("dyn matrix:")
        pprint(self.A)       
        print("\ninput matrix:")
        pprint(self.B)
        print("\neq_points:", eq_points, "\n\n")
        ###sys.exit()
    
    # -------------------------------------------
        self.discretisation_function = globals().get(f"discretisation_{discretisation_mode}")
        self.x_update_function = globals().get(f"x_update_mode__{x_update_mode}")

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
        self.A, self.B, self.I, tau = self.A.to(device, dtype=self.dtype), self.B.to(device, dtype=self.dtype), self.I.to(device, dtype=self.dtype), tau.to(device, dtype=self.dtype)

        outputs = torch.empty(batch_size, seq_len, self.z_size, device=device)  # Preallocate tensor | before: torch.zeros
        coefficients = torch.empty(batch_size, seq_len, self.state_dim, device=device)  ###############

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
            x_mid = self.discretisation_function(x_prev, (self.A, tau_t, self.I, self.B), u_dyn)             # same as the old one
            h, c = torch.stack(h_list, dim=0), torch.stack(c_list, dim=0)
            x_next, coeff = self.x_update_function(x_mid, h, self.alpha_gate, self.W__h_to_x)   ###############   

            z, c_z = torch.cat((x_next, h), dim=-1), torch.cat((c_x, c), dim=-1)            # same as the old one
            outputs[:, t, :].copy_(z[-1])  # In-place assignment
            coefficients[:, t, :].copy_(coeff)  ###############

        #if torch.isnan(outputs).any() or torch.isinf(outputs).any(): sys.exit()
        return torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False), (z, c_z), coefficients    ###############


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
        B = torch.tensor([[0], [0]])

        if model_name == "VanDerPol":
            mhu = self.data["dynamics"]["args"]["damping"]
            A = dyn_factor * torch.tensor([[0, 1], [-1, mhu]])          # before | self.A = dyn_factor * torch.tensor([[mhu, -mhu], [1/mhu, 0]])
            B = dyn_factor * torch.tensor([[0], [1]])

            # Equilibrium point for VdP: (x*, y*) = (0, 0)
            eq_point = torch.tensor([[0], [0]])

        elif model_name == "FitzHughNagumo":
            tau = self.data["dynamics"]["args"]["tau"]
            a = self.data["dynamics"]["args"]["a"]
            b = self.data["dynamics"]["args"]["b"]
            v_fact = 50
            
            # Solve equilibrium equations
            # w* = (v* + a) / b
            # v* - v*^3 / 3 - (v* + a)/b = 0 (solve numerically)
            from scipy.optimize import fsolve
            
            def fhn_equilibrium(v):
                return v - v**3 - (v - a) / b  

            v_star = fsolve(fhn_equilibrium, 0)[0]
            w_star = (v_star - a) / b
            
            A = dyn_factor * torch.tensor([[v_fact*(1 - 3*v_star**2), -v_fact], [1 / tau, -b / tau]])
            B = dyn_factor * torch.tensor([[v_fact], [0]])

            eq_point = torch.tensor([v_star, w_star])

            # A: [[7.1408, -10.0000], [0.2500, -0.3500]]
            # eq_point: [0.3087, 0.4348]

        elif model_name == "GreenshieldsTraffic":
            v0 = self.data["dynamics"]["args"]["v0"]
            n = self.data["dynamics"]["args"]["n"]
            A = dyn_factor * torch.tensor([[-v0 / n]])
            eq_point = torch.tensor([n])  # Equilibrium at max density

        elif model_name == "HodgkinHuxleyFS":
            # HHFS has 4 states: [V, n, m, h]
            # We'll use a plausible resting state and a 4x4 zero Jacobian as placeholder
            eq_point = torch.tensor([-0.7, 0.0032035, 0.00070115, 0.99988])  # Normalized units

            # Linearized system matrices at the equilibrium point
            A = dyn_factor * torch.tensor([
                [-3.0000e+00, -5.2600e-05,  1.9819e-02,  4.6326e-06],
                [ 5.5440e+00, -9.0943e+01,  0.0000e+00,  0.0000e+00],
                [ 2.4460e+01,  0.0000e+00, -1.5075e+03,  0.0000e+00],
                [-2.1701e-01,  0.0000e+00,  0.0000e+00, -7.0857e+01]
            ])

            B = dyn_factor * torch.tensor([
                [2.0],
                [0.0],
                [0.0],
                [0.0]
            ])

        elif model_name == "HodgkinHuxleyFFE":
            # HHFFE has 10 states: 2 RSA neurons (5 vars each)
            # Resting point is a zero vector as placeholder
            eq_point = torch.zeros(10)
            A = dyn_factor * torch.zeros((10, 10))

            """elif model_name in ["HodgkinHuxleyFFE", "HodgkinHuxleyFS"]:
                # HH model equilibrium is at resting potential (V*, n*, m*, h*)
                # This is complex, so a placeholder value is used
                A = dyn_factor * torch.tensor([[-1, 1], [-1, 0]])  # Placeholder
                eq_point = torch.tensor([0, 0])  # Should be the HH resting potential"""

        elif model_name == "LinearSys":
            A = dyn_factor * torch.tensor(self.data["dynamics"]["args"]["a"])
            B = dyn_factor * torch.tensor([[0], [1]])
            eq_point = torch.zeros(A.shape[0])  # Equilibrium at x* = 0

        elif model_name == "TwoTank":
            # Placeholder values for resistances (R1, R2) and capacitances (C1, C2)
            ###R1, R2, C1, C2 = 1, 1, 1, 1  # These should be properly defined
            ###A = dyn_factor * torch.tensor([[-1 / (R1 * C1), 0], [1 / (R1 * C2), -1 / (R2 * C2)]])
            
            c1 = self.data["dynamics"]["args"].get("c1", 0.08)
            c2 = self.data["dynamics"]["args"].get("c2", 0.04)
            epsilon = 1e-3
            h1_star, h2_star = epsilon, epsilon

            A = dyn_factor * torch.tensor([
                [-c2 / (2 * torch.sqrt(h1_star)), 0.0],
                [ c2 / (2 * torch.sqrt(h1_star)), -c2 / (2 * torch.sqrt(h2_star))]
            ])
            eq_point = torch.tensor([h1_star, h2_star])  # At rest with no input

        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        #print(A.shape[0])
        return A, B, eq_point

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


# ---------------- Discretisation functions --------------------------------- #

def discretisation_FE(x_prev, mat, u):
    A, tau, I, B = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)    # 2, 1).transpose(1, 2)
    x_prev = x_prev.squeeze(0).unsqueeze(1)

    transform_matrix = I + tau * A 
    input_matrix = tau * B.unsqueeze(0).expand(batch_size, -1, -1)
    input_matrix = input_matrix.transpose(1, 2)

    ev_lib = torch.bmm(x_prev, transform_matrix)
    ev_for = torch.bmm(u, input_matrix)

    x_next = ev_lib + ev_for

    return x_next.squeeze(1).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_BE(x_prev, mat, u):
    A, tau, I, B = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)

    A_neg = I - tau * A
    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)
    u_effect = torch.bmm(B_exp, u)
    u_effect_scaled = tau * u_effect

    ev_lib = torch.linalg.solve(A_neg, x_prev)
    ev_for = torch.linalg.solve(A_neg, u_effect_scaled)

    """inv_matrix = torch.inverse(A_neg)
    x_next = torch.bmm(x_prev, inv_matrix)"""

    x_next = ev_lib + ev_for
    return x_next.squeeze(2).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_TU(x_prev, mat, u):
    A, tau, I, B = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(2)

    A_pos = I + (tau / 2) * A
    A_neg = I - (tau / 2) * A

    """v = torch.linalg.solve(A_neg, x_prev)
    x_next = torch.bmm(A_pos, v)"""

    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)
    u_effect = torch.bmm(B_exp, u)
    u_scaled = tau * u_effect

    rhs = torch.bmm(A_pos, x_prev)  # ev_lib pre-soluzione
    rhs_total = rhs + u_scaled      # ev_lib + ev_for (prima di risolvere)


    x_next = torch.linalg.solve(A_neg, rhs_total)
    return x_next.squeeze(2).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_RK4(x_prev, mat, u):
    """
    Runge-Kutta 4th order (RK4) discretisation.
    Uses the classical RK4 method to approximate x_next.
    """
    A, tau, _, B = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)
    A = A.expand(batch_size, -1, -1)

    B_exp = B.unsqueeze(0).expand(batch_size, -1, -1)
    Bu = torch.bmm(u, B_exp.transpose(1, 2))  # [batch, 1, state]

    def f(x): return torch.bmm(x, A) + Bu

    k1 = f(x_prev)
    k2 = f(x_prev + 0.5 * tau * k1)
    k3 = f(x_prev + 0.5 * tau * k2)
    k4 = f(x_prev + tau * k3)

    ev_lib = x_prev
    ev_for = (tau / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    x_next = ev_lib + ev_for
    return x_next.squeeze(1).unsqueeze(0)   # .permute(1, 0, 2)


def discretisation_exact(x_prev, mat, u):
    """
    Exact discretisation: x_next = exp(tau * A) * x_prev
    Uses the matrix exponential function from PyTorch.
    """
    A, tau, _, B = mat
    batch_size = tau.shape[0]

    tau = tau.view(batch_size, 1, 1)
    u = u.view(batch_size, 1, 1)
    x_prev = x_prev.squeeze(0).unsqueeze(1)

    exp_matrix = torch.matrix_exp(tau * A)
    ev_lib = torch.bmm(x_prev, exp_matrix)

    B_exp = B.T.unsqueeze(0).expand(batch_size, -1, -1)
    ev_for = torch.bmm(u, tau * B_exp)

    x_next = ev_lib + ev_for
    return x_next.squeeze(1).unsqueeze(0)   # .permute(1, 0, 2)


# ---------------- x_update_mode -------------------------------------------- #

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
