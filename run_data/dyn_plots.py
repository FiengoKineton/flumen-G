import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import argparse


class Dynamics: 
    def __init__(self, both=False):
        self.t_span = (0, 150)
        self.t_eval = np.linspace(*self.t_span, 1000)
        self.u_array = self.generate_random_input(self.t_eval, step_size=2.0)

        # -------------------------------
        # Parametri
        self.mu = -0.001   # 1.0

        self.tau = 0.8
        self.a = -0.3
        self.b = 1.4
        self.v_fact = 2.0

        self.control_delta = 0.2

        self.v_star = fsolve(self.fhn_equilibrium, 0)[0]
        self.w_star = (self.v_star - self.a) / self.b
        self.eq_fhn_np = np.array([self.v_star, self.w_star])

        # -------------------------------
        # Matrici Linearizzate
        self.A_vdp_np = self.control_delta*np.array([[0.0, 1.0], [-1.0, self.mu]])
        self.B_vdp_np = self.control_delta*np.array([0, 1])
        self.eq_vdp_np = np.array([0.0, 0.0])

        self.A_fhn_np = self.control_delta*np.array([[self.v_fact * (1 - 3 * self.v_star**2), -self.v_fact], [1 / self.tau, -self.b / self.tau]])
        self.B_fhn_np = self.control_delta*np.array([self.v_fact, 0])
        # -------------------------------
        # Simulazione

        # Iniziali (piccola perturbazione attorno al punto di equilibrio)
        x0_vdp = self.eq_vdp_np + np.array([0.1, 0.0])
        x0_fhn = self.eq_fhn_np + np.array([0.1, 0.0])

        # Simulazioni non lineari
        self.sol_vdp = solve_ivp(self.van_der_pol, self.t_span, x0_vdp, args=(self.mu,), t_eval=self.t_eval)
        self.sol_fhn = solve_ivp(self.fitzhugh_nagumo, self.t_span, x0_fhn, args=(self.tau, self.a, self.b, self.v_fact), t_eval=self.t_eval)

        # Simulazioni linearizzate
        self.sol_lin_vdp = solve_ivp(self.linear_system, self.t_span, x0_vdp, args=(self.A_vdp_np, self.B_vdp_np, self.eq_vdp_np), t_eval=self.t_eval)
        self.sol_lin_fhn = solve_ivp(self.linear_system, self.t_span, x0_fhn, args=(self.A_fhn_np, self.B_fhn_np, self.eq_fhn_np), t_eval=self.t_eval)


        self.plot(both)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Run results analysis with optional display and plotting.")
        parser.add_argument("--both", action="store_true", help="Select all metrics across datasets.")
        args = parser.parse_args()
        return args


    # -------------------------------
    # Grafici
    def plot(self, both):
        plt.figure(figsize=(14, 8))

        # Van der Pol
        plt.subplot(3, 1, 1)
        plt.plot(self.sol_vdp.t, self.sol_vdp.y[0], label='Nonlinear: p')
        if both: plt.plot(self.sol_lin_vdp.t, self.sol_lin_vdp.y[0], '--', label='Linearized: p')
        plt.title('Van der Pol Oscillator')
        plt.xlabel('Time')
        plt.ylabel('Position p')
        plt.legend()
        plt.grid()

        # FitzHugh-Nagumo
        plt.subplot(3, 1, 2)
        plt.plot(self.sol_fhn.t, self.sol_fhn.y[0], label='Nonlinear: v')
        if both: plt.plot(self.sol_lin_fhn.t, self.sol_lin_fhn.y[0], '--', label='Linearized: v')
        plt.title('FitzHugh-Nagumo Model')
        plt.xlabel('Time')
        plt.ylabel('Excitation v')
        plt.legend()
        plt.grid()

        # Ingresso
        plt.subplot(3, 1, 3)
        plt.plot(self.t_eval, self.u_array, label='u(t)')
        plt.title('Ingresso randomico a gradini')
        plt.ylabel('u(t)')
        plt.grid()

        plt.tight_layout()
        plt.show()


    # -------------------------------
    # Funzione per generare un segnale randomico a gradini
    def generate_random_input(self, t_eval, step_size=2.0, low=-1.0, high=1.0, seed=42):
        np.random.seed(seed)
        t_min, t_max = t_eval[0], t_eval[-1]
        steps = np.arange(t_min, t_max, step_size)
        values = np.random.uniform(low, high, size=len(steps))
        
        u_t = np.zeros_like(t_eval)
        for i, t in enumerate(t_eval):
            index = int((t - t_min) // step_size)
            u_t[i] = values[min(index, len(values) - 1)]
        
        return u_t

    # -------------------------------
    # Dinamica Van der Pol (non lineare)
    def van_der_pol(self, t, x, mu):
        p, v = x
        u = np.interp(t, self.t_eval, self.u_array)

        dp = v
        dv = -p + mu * (1 - p**2) * v + u
        return [dp, dv]

    # -------------------------------
    # Dinamica FitzHugh-Nagumo (non lineare)
    def fitzhugh_nagumo(self, t, x, tau, a, b, fact):
        v, w = x
        u = np.interp(t, self.t_eval, self.u_array)

        dv = fact * (v - v**3 - w) + fact * u
        dw = (v - a - b * w) / tau
        return [dv, dw]

    # -------------------------------
    # Dinamica Linearizzata (generica)
    def linear_system(self, t, x, A, B, eq):
        u = np.interp(t, self.t_eval, self.u_array)
        return A @ (x - eq) + B * u

    # -------------------------------
    # Equilibrio FitzHugh-Nagumo
    def fhn_equilibrium(self, v):
        return v - v**3 - (v - self.a) / self.b



if __name__ == "__main__":
    args = Dynamics.parse_arguments()
    Dynamics(both=args.both)
