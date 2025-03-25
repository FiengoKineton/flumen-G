import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# -------------------------------
# Funzione per generare un segnale randomico a gradini
def generate_random_input(t_eval, step_size=2.0, low=-1.0, high=1.0, seed=42):
    np.random.seed(seed)
    t_min, t_max = t_eval[0], t_eval[-1]
    steps = np.arange(t_min, t_max, step_size)
    values = np.random.uniform(low, high, size=len(steps))
    
    u_t = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        index = int((t - t_min) // step_size)
        u_t[i] = values[min(index, len(values) - 1)]
    
    return u_t


t_span = (0, 150)
t_eval = np.linspace(*t_span, 1000)
u_array = generate_random_input(t_eval, step_size=2.0)


# -------------------------------
# Dinamica Van der Pol (non lineare)
def van_der_pol(t, x, mu):
    p, v = x
    u = np.interp(t, t_eval, u_array)

    dp = v
    dv = -p + mu * (1 - p**2) * v + u
    return [dp, dv]

# -------------------------------
# Dinamica FitzHugh-Nagumo (non lineare)
def fitzhugh_nagumo(t, x, tau, a, b, fact):
    v, w = x
    u = np.interp(t, t_eval, u_array)

    dv = fact * (v - v**3 - w) + fact * u
    dw = (v - a - b * w) / tau
    return [dv, dw]

# -------------------------------
# Dinamica Linearizzata (generica)
def linear_system(t, x, A, B, eq):
    u = np.interp(t, t_eval, u_array)
    return A @ (x - eq) + B * u

# -------------------------------
# Parametri
mu = -0.2   # 1.0

tau = 0.8
a = -0.3
b = 1.4
v_fact = 2.0

control_delta = 0.2

# -------------------------------
# Equilibrio FitzHugh-Nagumo
def fhn_equilibrium(v):
    return v - v**3 - (v - a) / b

v_star = fsolve(fhn_equilibrium, 0)[0]
w_star = (v_star - a) / b
eq_fhn_np = np.array([v_star, w_star])

# -------------------------------
# Matrici Linearizzate
A_vdp_np = np.array([[0.0, 1.0], [-1.0, mu]])
B_vdp_np = np.array([0, 1])
eq_vdp_np = np.array([0.0, 0.0])

A_fhn_np = np.array([[v_fact * (1 - 3 * v_star**2), -v_fact], [1 / tau, -b / tau]])
B_fhn_np = np.array([v_fact, 0])
# -------------------------------
# Simulazione

# Iniziali (piccola perturbazione attorno al punto di equilibrio)
x0_vdp = eq_vdp_np + np.array([0.1, 0.0])
x0_fhn = eq_fhn_np + np.array([0.1, 0.0])

# Simulazioni non lineari
sol_vdp = solve_ivp(van_der_pol, t_span, x0_vdp, args=(mu,), t_eval=t_eval)
sol_fhn = solve_ivp(fitzhugh_nagumo, t_span, x0_fhn, args=(tau, a, b, v_fact), t_eval=t_eval)

# Simulazioni linearizzate
sol_lin_vdp = solve_ivp(linear_system, t_span, x0_vdp, args=(A_vdp_np, B_vdp_np, eq_vdp_np), t_eval=t_eval)
sol_lin_fhn = solve_ivp(linear_system, t_span, x0_fhn, args=(A_fhn_np, B_fhn_np, eq_fhn_np), t_eval=t_eval)

# -------------------------------
# Grafici
plt.figure(figsize=(14, 8))

# Van der Pol
plt.subplot(3, 1, 1)
plt.plot(sol_vdp.t, sol_vdp.y[0], label='Nonlinear: p')
plt.plot(sol_lin_vdp.t, sol_lin_vdp.y[0], '--', label='Linearized: p')
plt.title('Van der Pol Oscillator')
plt.xlabel('Time')
plt.ylabel('Position p')
plt.legend()
plt.grid()

# FitzHugh-Nagumo
plt.subplot(3, 1, 2)
plt.plot(sol_fhn.t, sol_fhn.y[0], label='Nonlinear: v')
plt.plot(sol_lin_fhn.t, sol_lin_fhn.y[0], '--', label='Linearized: v')
plt.title('FitzHugh-Nagumo Model')
plt.xlabel('Time')
plt.ylabel('Excitation v')
plt.legend()
plt.grid()

# Ingresso
plt.subplot(3, 1, 3)
plt.plot(t_eval, u_array, label='u(t)')
plt.title('Ingresso randomico a gradini')
plt.ylabel('u(t)')
plt.grid()

plt.tight_layout()
plt.show()
