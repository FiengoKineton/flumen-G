import torch, sys
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


# ------------------------- LSTM ------------------------- #

class LSTM(nn.Module):
    def __init__(self, input_size, z_size, num_layers=1, output_size=None,
                 bias=True, batch_first=True, dropout=0.0, bidirectional=False, 
                 state_dim=None, discretisation_mode=None, x_update_mode=None, model_data=None):
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
        self.data = model_data

        self.A, _ = self.get_dyn_matrix()
        self.I = torch.eye(self.A.shape[0])
        self.dtype = torch.float32

        self.discretisation_function = globals().get(f"discretisation_{discretisation_mode}")
        self.x_update_function = globals().get(f"x_update_mode__{x_update_mode}")

        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size + state_dim if layer == 0 else self.hidden_size, self.hidden_size, bias)        # torch.jit.script()    
            for layer in range(num_layers)
        ])

        self.alpha_gate = nn.Linear(self.hidden_size, self.state_dim, bias=bias)  # Gate function
        self.W__h_to_x = nn.Linear(self.hidden_size, self.state_dim, bias=bias)   # Mapping function


    def forward(self, rnn_input: PackedSequence, hidden_state, tau):
        rnn_input_unpacked, lengths = pad_packed_sequence(rnn_input, batch_first=self.batch_first)
        batch_size, seq_len, _ = rnn_input_unpacked.shape
        device = rnn_input_unpacked.device

        z, c_z = hidden_state
        self.A, self.I, tau = self.A.to(device, dtype=self.dtype), self.I.to(device, dtype=self.dtype), tau.to(device, dtype=self.dtype)

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

            x_mid = self.discretisation_function(x_prev, self.A, tau_t, self.I)             # same as the old one
            h, c = torch.stack(h_list, dim=0), torch.stack(c_list, dim=0)
            x_next, coeff = self.x_update_function(x_mid, h, self.alpha_gate, self.W__h_to_x)   ###############   

            z, c_z = torch.cat((x_next, h), dim=-1), torch.cat((c_x, c), dim=-1)            # same as the old one
            outputs[:, t, :].copy_(z[-1])  # In-place assignment
            coefficients[:, t, :].copy_(coeff)  ###############

        return torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=self.batch_first, enforce_sorted=False), (z, c_z), coefficients    ###############


    def get_dyn_matrix(self): 
        """
        Computes the linearized dynamics matrix A and equilibrium points for different systems based on self.data.
        Returns:
            A (torch.Tensor): Linearized system dynamics matrix.
            eq_point (torch.Tensor): Equilibrium state (x*).
        """
        model_name = self.data["settings"]["dynamics"]["name"]
        dyn_factor = self.data["settings"]["control_delta"]

        if model_name == "VanDerPol":
            mhu = self.data["settings"]["dynamics"]["args"]["damping"]
            A = dyn_factor * torch.tensor([[0, 1], [-1, mhu]])          # before | self.A = dyn_factor * torch.tensor([[mhu, -mhu], [1/mhu, 0]])
            # Equilibrium point for VdP: (x*, y*) = (0, 0)
            eq_point = torch.tensor([0, 0])

        elif model_name == "FitzHughNagumo":
            tau = self.data["settings"]["dynamics"]["args"]["tau"]
            a = self.data["settings"]["dynamics"]["args"]["a"]
            b = self.data["settings"]["dynamics"]["args"]["b"]
            
            # Solve equilibrium equations
            # w* = (v* + a) / b
            # v* - v*^3 / 3 - (v* + a)/b = 0 (solve numerically)
            from scipy.optimize import fsolve
            
            def fhn_equilibrium(v):
                return v - v**3 / 3 - (v + a) / b  # R * I_ext not considered

            v_star = fsolve(fhn_equilibrium, 0)[0]
            w_star = (v_star + a) / b
            
            A = dyn_factor * torch.tensor([[1 - v_star**2, -1], [1 / tau, -b / tau]])
            eq_point = torch.tensor([v_star, w_star])

        elif model_name == "GreenshieldsTraffic":
            v0 = self.data["settings"]["dynamics"]["args"]["v0"]
            n = self.data["settings"]["dynamics"]["args"]["n"]
            A = dyn_factor * torch.tensor([[-v0 / n]])
            eq_point = torch.tensor([n])  # Equilibrium at max density

        elif model_name in ["HodgkinHuxleyFFE", "HodgkinHuxleyFS"]:
            # HH model equilibrium is at resting potential (V*, n*, m*, h*)
            # This is complex, so a placeholder value is used
            A = dyn_factor * torch.tensor([[-1, 1], [-1, 0]])  # Placeholder
            eq_point = torch.tensor([0, 0])  # Should be the HH resting potential

        elif model_name == "LinearSys":
            A = dyn_factor * torch.tensor(self.data["settings"]["dynamics"]["args"]["a"])
            eq_point = torch.zeros(A.shape[0])  # Equilibrium at x* = 0

        elif model_name == "TwoTank":
            # Placeholder values for resistances (R1, R2) and capacitances (C1, C2)
            R1, R2, C1, C2 = 1, 1, 1, 1  # These should be properly defined
            A = dyn_factor * torch.tensor([[-1 / (R1 * C1), 0], [1 / (R1 * C2), -1 / (R2 * C2)]])
            eq_point = torch.tensor([0, 0])  # At rest with no input

        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        #print(A.shape[0])
        return A, eq_point



# ------------------------- LSTMCell ------------------------- #

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.WU = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, u, h, c):
        gates = self.WU(torch.cat((u, h), dim=1))  # Single matrix multiplication
        i, f, g, o = gates.chunk(4, dim=1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c.mul_(f).add_(i * g)               # c_next = f * c + i * g            | old
        h = o * torch.tanh(c)               # h_next = o * torch.tanh(c_next)   | old

        return h, c


# ------------------------- Discretisation functions ------------------------- #

def discretisation_FE(x_prev, A, tau, I):
    batch_size = tau.shape[0]
    tau = tau.view(batch_size, 1, 1)

    transform_matrix = I + tau * A
    x_prev = x_prev.squeeze(0).unsqueeze(1)  

    return torch.bmm(x_prev, transform_matrix).permute(1, 0, 2)


def discretisation_BE(x_prev, A, tau, I):
    batch_size = tau.shape[0]
    tau = tau.view(batch_size, 1, 1)

    inv_matrix = torch.inverse(I - tau * A)
    x_prev = x_prev.squeeze(0).unsqueeze(1)  

    return torch.bmm(x_prev, inv_matrix).permute(1, 0, 2)


def discretisation_TU(x_prev, A, tau, I):
    batch_size = tau.shape[0]
    tau = tau.view(batch_size, 1, 1)

    A_pos = I + (tau / 2) * A
    A_neg = I - (tau / 2) * A
    A_neg_inv = torch.inverse(A_neg)  

    transform_matrix = torch.bmm(A_pos, A_neg_inv)

    x_prev = x_prev.squeeze(0).unsqueeze(1)  
    return torch.bmm(x_prev, transform_matrix).permute(1, 0, 2)


def discretisation_RK4(x_prev, A, tau, I):
    """
    Runge-Kutta 4th order (RK4) discretisation.
    Uses the classical RK4 method to approximate x_next.
    """
    batch_size = tau.shape[0]
    tau = tau.view(batch_size, 1, 1)

    x_prev = x_prev.squeeze(0).unsqueeze(1)
    A = A.expand(batch_size, -1, -1)

    #print(x_prev.shape) # torch.Size([128, 1, 2])
    #print(A.shape)      # torch.Size([128, 2, 2])
    #print(tau.shape)    # torch.Size([128, 1, 1])

    k1 = torch.bmm(x_prev, A)
    k2 = torch.bmm(x_prev + 0.5 * tau * k1, A)
    k3 = torch.bmm(x_prev + 0.5 * tau * k2, A)
    k4 = torch.bmm(x_prev + tau * k3, A)

    x_next = x_prev + (tau / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next.permute(1, 0, 2)


def discretisation_exact(x_prev, A, tau, I):
    """
    Exact discretisation: x_next = exp(tau * A) * x_prev
    Uses the matrix exponential function from PyTorch.
    """
    batch_size = tau.shape[0]
    tau = tau.view(batch_size, 1, 1)

    exp_matrix = torch.matrix_exp(tau * A)

    x_prev = x_prev.squeeze(0).unsqueeze(1)
    return torch.bmm(x_prev, exp_matrix).permute(1, 0, 2)


# ------------------------- x_update_mode ------------------------- #

def x_update_mode__alpha(x_mid, h, alpha_gate, W__h_to_x):
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

def x_update_mode__beta(x_mid, h, alpha_gate, W__h_to_x):
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

def x_update_mode__lamda(x_mid, h, alpha_gate, W__h_to_x):
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