import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PhasePlot: 
    def __init__(self, which):

        # Time span and initial condition
        self.t_span = (0, 40)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 5000)
        self.y0 = [2.0, 0.0]  # Try different values to see convergence to the limit cycle

        # Create grid of initial conditions (polar coordinates → Cartesian)
        self.r_vals = np.linspace(0.05, 2.5, 20)     # distances from origin
        self.angles = np.linspace(0, 2*np.pi, 30)    # directions
        self.initial_conds = [
            [r * np.cos(theta), r * np.sin(theta)]
            for r in self.r_vals for theta in self.angles
        ]


        if which:   self.sol = solve_ivp(self.vdp, self.t_span, self.y0, t_eval=self.t_eval)
        else:       self.sol = solve_ivp(self.fhn, self.t_span, self.y0, t_eval=self.t_eval)
        
        self.plot_limit_cicle()
        self.plot_phase_plot(which)


    # Dinamica VanDerPol
    def vdp(self, t, y, mu=1):
        x1, x2 = y
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        return [dx1, dx2]

    # Dinamica FitzHugh-Nagumo
    def fhn(self, t, x):
        v, w = x
        tau, a, b, fact = 0.8, -0.3, 1.4, 50

        dv = fact * (v - (v**3) / 3 - w)
        dw = (v + a - b * w) / tau
        return [dv, dw]

    
    def plot_limit_cicle(self): 
        # Extract results
        x1 = self.sol.y[0]
        x2 = self.sol.y[1]

        # Plot phase portrait
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x1, x2, color='royalblue')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('Phase Portrait (Limit Cycle)')
        plt.grid(True)
        plt.axis('equal')

        # Plot x1 over time
        plt.subplot(1, 2, 2)
        plt.plot(self.sol.t, x1, label='$x_1(t)$', color='crimson')
        plt.xlabel('Time')
        plt.ylabel('$x_1$')
        plt.title('Time Evolution')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_phase_plot(self, which):
        # Plot
        plt.figure(figsize=(8, 8))
        for y0 in self.initial_conds:
            if which: sol = solve_ivp(self.vdp, self.t_span, y0, t_eval=self.t_eval)
            else:     sol = solve_ivp(self.fhn, self.t_span, y0, t_eval=self.t_eval)

            plt.plot(sol.y[0], sol.y[1], color='royalblue', alpha=0.4, linewidth=0.8)

        # Highlight the equilibrium point
        plt.plot(0, 0, 'ro', label='Equilibrium (Unstable)')

        plt.title("Phase Portrait & ROA Insight")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__": 
    which = True            # if True then VanDerPol
    PhasePlot(which=which)