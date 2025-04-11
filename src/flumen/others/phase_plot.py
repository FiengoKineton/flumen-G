import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from control.phaseplot import phase_plot
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations



class PhasePlot: 
    def __init__(self, which):

        # Time span and initial condition
        self.t_span = (0, 40)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 5000)

        # Create grid of initial conditions (polar coordinates → Cartesian)
        self.r_vals = np.linspace(0.05, 2.5, 20)     # distances from origin
        self.angles = np.linspace(0, 2*np.pi, 30)    # directions

        if which=='vdp': 
            self.y0 = [2.0, 0.0]
            print("VanDerPol")
            self.eq = np.array([0.0, 0.0])  
            print("equilibrium:", self.eq)
            self.sol = solve_ivp(self.vdp, self.t_span, self.y0, t_eval=self.t_eval)
            self.i, self.j = 0, 1
        elif which=='fhn':   
            self.y0 = [2.0, 0.0]    
            print("FitzHugh-Nagumo")
            self.eq = fsolve(lambda p: self.fhn(0, p), [0.0, 0.0])  # np.array([0.3087, 0.4348])
            print("equilibrium:", self.eq)
            self.sol = solve_ivp(self.fhn, self.t_span, self.y0, t_eval=self.t_eval)
            self.i, self.j = 0, 1
        elif which=='nad':
            self.y0 = [2.0, 0.0, 0.0, 0.0, 0.0]
            print("NonlinearActivationDynamics")
            self.eq = np.array([0.42599762, 0.42599282, 0.42591131, 0.4245284,  0.40105814])  
            print("equilibrium:", self.eq)
            self.sol = solve_ivp(self.nad, self.t_span, self.y0, t_eval=self.t_eval)
            self.i, self.j = 0, 4


        self.initial_conds = []
        if which in ['vdp', 'fhn']:
            self.initial_conds = [
                [self.eq[self.i] + r * np.cos(theta), self.eq[self.j] + r * np.sin(theta)]
                for r in self.r_vals for theta in self.angles
            ]
        elif which == 'nad':
            # Per il sistema NAD (a 5 dimensioni), proiettiamo solo le prime 2 componenti
            self.initial_conds = [
                [self.eq[self.i] + r * np.cos(theta), self.eq[self.j] + r * np.sin(theta)]
                for r in np.linspace(0.05, 2.5, 20)
                for theta in np.linspace(0, 2*np.pi, 30)
            ]

        #self.animate_2d()
        #self.plot_3d_phase()
        #self.plot_3d_phase_grid()

        #self.plot_limit_cicle()
        #self.animate_2d()
        #self.plot_phase_plot(which)

    # -------------------------------------------------------
    # Dinamiche del sistema
    def vdp(self, t, y, mu=1):
        x1, x2 = y
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        return [dx1, dx2]

    def fhn(self, t, x, k=50):
        v, w = x
        tau, a, b = 0.8, -0.3, 1.4

        dv = k * (v - (v**3) - w)
        dw = (v - a - b * w) / tau
        return [dv, dw]

    def nad(self, t, x, mode='stable'):
        a_s = np.array([[-1.0,  0.3,  0.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.3,  0.0,  0.0],
                [ 0.0,  0.0, -1.0,  0.3,  0.0],
                [ 0.0,  0.0,  0.0, -1.0,  0.3],
                [ 0.0,  0.0,  0.0,  0.0, -1.0]])
        a_m = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],  
                [1.0, 0.0, 0.0, 0.0, 0.0]])
        
        A = a_s if mode == 'stable' else a_m
        # B = np.array([[1.0], [0.0], [0.0], [0.0], [0.0]])

        from scipy.special import expit
        dx = -x + expit(A @ x)
        return dx
    
    # -------------------------------------------------------
    # Funzioni di plotting
    def plot_limit_cicle(self): 
        # Extract results
        x1 = self.sol.y[self.i]
        x2 = self.sol.y[self.j]

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
        plt.figure(figsize=(8, 8))

        """for y0 in self.initial_conds:
            if which=='vdp': 
                sol = solve_ivp(self.vdp, self.t_span, y0, t_eval=self.t_eval)
            elif which=='fhn':
                sol = solve_ivp(self.fhn, self.t_span, y0, t_eval=self.t_eval)
            elif which=='nad':
                sol = solve_ivp(self.nad, self.t_span, y0, t_eval=self.t_eval)

            #sol = solve_ivp(self.vdp if which else self.fhn, self.t_span, y0, t_eval=self.t_eval)
            plt.plot(sol.y[0], sol.y[1], color='royalblue', alpha=0.4, linewidth=0.8)"""
        
        for y0 in self.initial_conds:
            if which == 'vdp':
                sol = solve_ivp(self.vdp, self.t_span, y0, t_eval=self.t_eval)
                plt.plot(sol.y[self.i], sol.y[self.j], color='royalblue', alpha=0.4, linewidth=0.8)
            elif which == 'fhn':
                sol = solve_ivp(self.fhn, self.t_span, y0, t_eval=self.t_eval)
                plt.plot(sol.y[self.i], sol.y[self.j], color='royalblue', alpha=0.4, linewidth=0.8)
            elif which == 'nad':
                y0_full = self.check_ij(y0)

                sol = solve_ivp(self.nad, self.t_span, y0_full, t_eval=self.t_eval)
                plt.plot(sol.y[self.i], sol.y[self.j], color='royalblue', alpha=0.4, linewidth=0.8)

        # Plot the correct equilibrium
        plt.plot(self.eq[self.i], self.eq[self.j], 'ro', label='Equilibrium')

        plt.title("Phase Portrait & ROA Insight")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    def check_ij(self, y0): 
        if self.i == 0 and self.j == 1:
            y0_full = [y0[0], y0[1], *self.eq[2:]]
        elif self.i == 0 and self.j == 2:
            y0_full = [y0[0], self.eq[1], y0[1], *self.eq[3:]]
        elif self.i == 0 and self.j == 3:
            y0_full = [y0[0], self.eq[1], self.eq[2], y0[1], self.eq[4]]
        elif self.i == 0 and self.j == 4:
            y0_full = [y0[0], *self.eq[1:4], y0[1]]
        elif self.i == 1 and self.j == 0:
            y0_full = [y0[1], y0[0], *self.eq[2:]]
        elif self.i == 1 and self.j == 2:
            y0_full = [self.eq[0], y0[0], y0[1], *self.eq[3:]]
        elif self.i == 1 and self.j == 3:
            y0_full = [self.eq[0], y0[0], self.eq[2], y0[1], self.eq[4]]
        elif self.i == 1 and self.j == 4:
            y0_full = [self.eq[0], y0[0], *self.eq[2:4], y0[1]]
        elif self.i == 2 and self.j == 0:
            y0_full = [y0[1], self.eq[1], y0[0], *self.eq[3:]]
        elif self.i == 2 and self.j == 1:
            y0_full = [self.eq[0], y0[1], y0[0], *self.eq[3:]]
        elif self.i == 2 and self.j == 3:
            y0_full = [*self.eq[:2], y0[0], y0[1], self.eq[4]]
        elif self.i == 2 and self.j == 4:
            y0_full = [*self.eq[:2], y0[0], self.eq[3], y0[1]]
        elif self.i == 3 and self.j == 0:
            y0_full = [y0[1], self.eq[1], self.eq[2], y0[0], self.eq[4]]
        elif self.i == 3 and self.j == 1:
            y0_full = [self.eq[0], y0[1], self.eq[2], y0[0], self.eq[4]]
        elif self.i == 3 and self.j == 2:
            y0_full = [*self.eq[:2], y0[1], y0[0], self.eq[4]]
        elif self.i == 3 and self.j == 4:
            y0_full = [*self.eq[:3], y0[0], y0[1]]
        elif self.i == 4 and self.j == 0:
            y0_full = [y0[1], *self.eq[1:4], y0[0]]
        elif self.i == 4 and self.j == 1:
            y0_full = [self.eq[0], y0[1], *self.eq[2:4], y0[0]]
        elif self.i == 4 and self.j == 2:
            y0_full = [*self.eq[:2], y0[1], self.eq[3], y0[0]]
        elif self.i == 4 and self.j == 3:
            y0_full = [*self.eq[:3], y0[1], y0[0]]

        return y0_full

    def animate_2d(self):
        i, j = self.i, self.j

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-2.5, 3)
        ax.set_ylim(-2.5, 3)
        ax.set_xlabel(f"$x_{i}$")
        ax.set_ylabel(f"$x_{j}$")
        ax.set_title("Animated Phase Plot")
        ax.grid(True)
        ax.plot(self.eq[i], self.eq[j], 'ro', label="Equilibrium")
        ax.legend()

        # Pre-calcola tutte le soluzioni
        all_sols = []
        for y0 in self.initial_conds:
            y0_full = self.eq.copy()
            y0_full[self.i] = y0[0]
            y0_full[self.j] = y0[1]
            sol = solve_ivp(self.nad, self.t_span, y0_full, t_eval=self.t_eval)
            all_sols.append(sol)

        # Crea una linea per ogni traiettoria
        traces = [ax.plot([], [], lw=1, alpha=0.5)[0] for _ in all_sols]
        points = [ax.plot([], [], 'o', markersize=3, alpha=0.5)[0] for _ in all_sols]

        def update(frame):
            for sol, trace, point in zip(all_sols, traces, points):
                x = sol.y[i][:frame]
                y = sol.y[j][:frame]
                trace.set_data(x, y)
                if frame < len(x):
                    point.set_data([x[-1]], [y[-1]])
            return traces + points

        ani = FuncAnimation(fig, update, frames=len(self.t_eval), interval=20, blit=True)
        plt.show()

    def plot_3d_phase(self, i=0, j=1, k=2):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Phase Portrait")

        # Loop su tutte le condizioni iniziali
        for y0 in self.initial_conds:
            # Crea un vettore iniziale completo di 5 componenti
            y0_full = self.eq.copy()
            y0_full[i] = y0[0]
            y0_full[j] = y0[1]

            # Risolvi l'ODE
            sol = solve_ivp(self.nad, self.t_span, y0_full, t_eval=self.t_eval)

            # Plot della traiettoria proiettata sui tre assi
            ax.plot(sol.y[i], sol.y[j], sol.y[k], alpha=0.5, lw=1)

        # Etichette assi
        ax.set_xlabel(f'$x_{i}$')
        ax.set_ylabel(f'$x_{j}$')
        ax.set_zlabel(f'$x_{k}$')
        ax.view_init(elev=30, azim=135)  # opzionale: cambia angolo 3D
        ax.set_title(f"3D Phase Portrait (i={i}, j={j}, k={k})")
        plt.tight_layout()
        plt.show()

    def plot_3d_phase_grid(self):
        fig = plt.figure(figsize=(20, 12))

        # Tutte le 15 permutazioni uniche di 3 variabili su 5 (senza ripetizioni)
        combos = list(permutations(range(5), 3))[:15]  # 5*4*3 = 60, ma ne prendiamo solo 15

        for idx, (i, j, k) in enumerate(combos):
            ax = fig.add_subplot(3, 5, idx + 1, projection='3d')
            ax.set_title(f'$x_{i}$-$x_{j}$-$x_{k}$', fontsize=10)

            for y0 in self.initial_conds:
                y0_full = self.eq.copy()
                y0_full[i] = y0[0]
                y0_full[j] = y0[1]
                sol = solve_ivp(self.nad, self.t_span, y0_full, t_eval=self.t_eval)
                ax.plot(sol.y[i], sol.y[j], sol.y[k], lw=0.7, alpha=0.6)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=25, azim=135)

        plt.tight_layout()
        plt.show()



if __name__ == "__main__": 
    which = 'nad'            # if True then VanDerPol, elif False then FitzHughNagumo
    p = PhasePlot(which=which)

    if which in ['vdp', 'fhn']: 
        p.plot_limit_cicle()
        p.plot_phase_plot(which)
        p.animate_2d()

    if which == 'nad': 
        #p.animate_2d(0, 1, 2)

        for i in range(5):
            for j in range(5):
                for k in range(5):
                    p.plot_3d_phase(i, j, k)
        
        """
        ombrello basso
        - 0, 1, 3
        - 0, 2, 3
        - 1, 2, 3

        ombrello alto: 
        - 0, 1, 2
        - 0, 1, 4
        - 0, 2, 4
        - 1, 2, 4
        - 1, 3, 4
        - 2, 3, 4
        """

        #p.plot_3d_phase_grid()