import numpy as np
import matplotlib.pyplot as plt

# Metodi di discretizzazione
def forward_euler(s, T):
    return 1 + s * T

def backward_euler(s, T):
    return 1 / (1 - s * T)

def tustin(s, T):
    return (2 + s * T) / (2 - s * T)

def plot_s_to_z_mapping_ultra(T=0.2, N=2000):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Griglia nel piano s
    x = np.linspace(-25 / T, 5 / T, N)
    y = np.linspace(-25 / T, 25 / T, N)
    S_real, S_imag = np.meshgrid(x, y)
    S = S_real + 1j * S_imag

    # Circonferenza unitaria
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.exp(1j * theta)

    # Poli nel piano s
    mhu = 1 # -1
    b, tau, v0 = 1.4, 0.8, 0.3087
    k = 50  # 1

    c1, c2 = -0.5*(b/tau-k*(1-3*v0**2)), k/tau*(1-b*(1-3*v0**2))

    print("VDP |", -1+mhu**2/4, "if pos then Real poles")
    print("FHN |", c1**2-c2, "if pos then Real poles")

    vdp_real, fhn_real = mhu/2, c1
    vdp_img = np.sqrt(np.abs(1-mhu**2/4))*1j if mhu**2/4-1<0 else np.sqrt(mhu**2/4-1)
    fhn_img = np.sqrt(np.abs(c2-c1**2))*1j if c1**2-c2<0 else np.sqrt(c1**2-c2)

    poli_s = np.array([vdp_real+vdp_img, vdp_real-vdp_img,
                       fhn_real+fhn_img, fhn_real-fhn_img])

    # Esegui la mappatura per ogni metodo
    for ax, method_func, title, formula in zip(
        [axes[0, 1], axes[1, 0], axes[1, 1]],
        [forward_euler, backward_euler, tustin],
        ["Forward Euler", "Backward Euler", "Tustin"], 
        ["z → 1+sT", "z → 1/(1-sT)", "z → (2+sT)/(2-sT)"]
    ):
        Z = method_func(S, T)
        z_imm = method_func(1j * np.linspace(-100, 100, N), T)
        poli_z = method_func(poli_s, T)

        # Plot mappatura del metodo
        ax.set_title(f"{title} Mapping {formula}")
        ax.axvline(0, color='black', lw=0.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel("Re(z)")
        ax.set_ylabel("Im(z)")
        ax.grid(True)

        # Plot circonferenza unitaria
        ax.plot(np.real(unit_circle), np.imag(unit_circle), 'k--', label='Unit Circle')

        # Plot asse immaginario
        ax.plot(np.real(z_imm), np.imag(z_imm), 'r', lw=1.5, label='Im(s)-axis')

        # Plot poli mappati
        ax.plot(np.real(poli_z), np.imag(poli_z), 'rx', markersize=8, label='Mapped Poles')

        # Mappatura stabile
        ax.contourf(np.real(Z), np.imag(Z), np.real(S) < 0, levels=[0.5, 1], colors=['lightblue'], alpha=0.4)
        ax.legend()

    # Plot piano s (continuo)
    ax_s = axes[0, 0]
    ax_s.set_title("Continuous-Time System (s-plane)")
    ax_s.axvline(0, color='black', lw=0.5)
    ax_s.axhline(0, color='black', lw=0.5)
    ax_s.set_xlim([-5, 35])
    ax_s.set_ylim([-1, 1])
    ax_s.set_xlabel("Re(s)")
    ax_s.set_ylabel("Im(s)")
    ax_s.grid(True)

    # Plot poli nel piano s
    ax_s.fill_betweenx(y, x[0], 0, color='lightblue', alpha=0.4)
    ax_s.plot(np.real(poli_s), np.imag(poli_s), 'rx', markersize=8, label='Poles')
    ax_s.plot([x[0], 0], [0, 0], 'r', lw=2, label='Re(s)<0')
    ax_s.legend()

    plt.tight_layout()
    plt.show()

# Esegui il plot
plot_s_to_z_mapping_ultra()
