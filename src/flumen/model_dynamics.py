import torch
import numpy as np
from pprint import pprint
from scipy.optimize import fsolve
from torch.autograd.functional import jacobian

# Constants
c_m = 0.5
v_k, v_na, v_l, v_t = -90., 50., -70., -56.2
g_k, g_na, g_l = 10., 56., 1.5e-2
v_scale = 100.
time_scale = 100.

# Define full dynamics in PyTorch
def hhfs_dynamics(xu):
    x = xu[:4]
    u = xu[4]
    v, n, m, h = x
    v_real = v * v_scale

    dv = (u - g_k * n**4 * (v_real - v_k) - g_na * m**3 * h * (v_real - v_na) - g_l * (v_real - v_l)) / (100 * c_m)

    a_n = -0.032 * (v_real - v_t - 15.) / (torch.exp(-(v_real - v_t - 15.) / 5.) - 1)
    b_n = 0.5 * torch.exp(-(v_real - v_t - 10.) / 40.)
    dn = a_n * (1 - n) - b_n * n

    a_m = -0.32 * (v_real - v_t - 13.) / (torch.exp(-(v_real - v_t - 13.) / 4.) - 1)
    b_m = 0.28 * (v_real - v_t - 40.) / (torch.exp((v_real - v_t - 40.) / 5.) - 1)
    dm = a_m * (1 - m) - b_m * m

    a_h = 0.128 * torch.exp(-(v_real - v_t - 17.) / 18.)
    b_h = 4. / (1 + torch.exp(-(v_real - v_t - 40.) / 5.))
    dh = a_h * (1 - h) - b_h * h

    return time_scale * torch.stack([dv, dn, dm, dh])

# Wrapper for fsolve (expects NumPy input/output)
def dynamics_numpy_for_fsolve(x_np):
    x_torch = torch.tensor(list(x_np) + [0.0], dtype=torch.float64)
    return hhfs_dynamics(x_torch).detach().numpy()

# Compute equilibrium point
x0 = [-0.7, 0.2, 0.05, 0.6]
x_star_np = fsolve(dynamics_numpy_for_fsolve, x0)
x_star = torch.tensor(list(x_star_np) + [0.0], dtype=torch.float64, requires_grad=True)


# Compute Jacobian at equilibrium
J = jacobian(hhfs_dynamics, x_star)
A = J[:, :4]
B = J[:, 4].unsqueeze(1)


print("\neq_point:")
pprint(x_star)
print("\n\nA:")
pprint(A)
print("\n\nB:")
pprint(B)
