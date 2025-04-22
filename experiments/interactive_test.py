import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from flumen import CausalFlowModel
from flumen.utils import pack_model_inputs
from generate_data import make_trajectory_sampler

from argparse import ArgumentParser

import yaml
from pathlib import Path
import sys
from pprint import pprint
from time import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file" "(or, if run with --wandb, path to a Weights & Biases artifact)")
    ap.add_argument('--print_info', action='store_true', help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--time_horizon', type=float, default=None)
    ap.add_argument('--more', action='store_true')

    return ap.parse_args()


def main():
    plt.ion()  # Enable interactive mode
    args = parse_args()

    num_times = 2 if args.time_horizon is None else 1          # default 2 | multiplied by the time_span
    more = args.more


    if args.wandb:
        import wandb
        api = wandb.Api()
        model_artifact = api.artifact(args.path)        
        model_path = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
        print(model_run.summary)
    else:
        model_path = Path(args.path)

    with open(model_path / "state_dict.pth", 'rb') as f:
        state_dict = torch.load(f, weights_only=True, map_location='cpu')
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
    
    if "dyn_matrix" in metadata["args"] and not isinstance(metadata["args"]["dyn_matrix"], torch.Tensor):
        metadata["args"]["dyn_matrix"] = torch.tensor(metadata["args"]["dyn_matrix"])

    #metadata["args"]["mode_rnn"] = 'old'
    pprint(metadata)

    mode_rnn = metadata.get("args", {}).get("mode_rnn", "old")  # ["args"]["mode_rnn"]

    if args.print_info:
        return

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    if args.continuous_state:
        xx = np.linspace(0., 1., model.state_dim)
        n_plots = 2
    else:
        n_plots = model.output_dim

    # First Figure: y_true vs y_pred + Input
    fig1, ax1 = plt.subplots(n_plots+1, 1, sharex=True)
    fig1.canvas.mpl_connect('close_event', on_close_window)

    # Second Figure: Delta Plots
    fig2, ax2 = plt.subplots(n_plots, 1, sharex=True) # 2
    fig2.canvas.mpl_connect('close_event', on_close_window)

    # Third Figure: Coefficient Evolution (for alpha, beta, lambda) ###############
    if mode_rnn!="old":
        fig3, ax3 = plt.subplots(2, 1, sharex=True)
        fig3.canvas.mpl_connect('close_event', on_close_window)

    others = True if more and mode_rnn!="old" else False
    if others:
        fig4, ax4 = plt.subplots(2, 1, sharex=True)
        fig4.canvas.mpl_connect('close_event', on_close_window)
                                
        fig5, ax5 = plt.subplots(1, 2)
        fig5.canvas.mpl_connect('close_event', on_close_window)

        fig6, ax6 = plt.subplots(1, 1)
        fig6.canvas.mpl_connect('close_event', on_close_window)


    time_horizon = args.time_horizon if args.time_horizon else num_times * metadata["data_args"]["time_horizon"]

    # Store insets and connection lines
    prev_insets = []
    prev_markings = []

    while True:
        time_integrate = time()
        x0, t, y, u = sampler.get_example(time_horizon=time_horizon, n_samples=int(1 + 100 * time_horizon))
        time_integrate = time() - time_integrate

        time_predict = time()
        x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(x0, t, u, delta)

        with torch.no_grad():
            y_pred, coeffs, matrices = model(x0_feed, u_feed, deltas_feed)
            # model 2 

        y_pred = np.flip(y_pred.numpy(), 0)
        coeffs = np.flip(coeffs.numpy(), 0) 
        matrices = np.flip(matrices.numpy(), 0)     # torch.size([150, 2, 2])
        if mode_rnn!="old": coeffs = coeffs[:, -1, :] 
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]
        sq_error = model.output_dim * np.mean(np.square(y - y_pred))
        print("MSE (mean square error):", sq_error, "\n")

        # **Clear previous plots and remove insets & connections**
        for ax_ in ax1:
            ax_.cla()
        for ax_ in ax2:
            ax_.cla()
        if mode_rnn!="old": 
            for ax_ in ax3: ax_.cla()  ###############
        if others: 
            for ax_ in ax4: ax_.cla()
            for ax_ in ax5: ax_.cla()
            ax6.cla()
            

        # **Remove previous insets and connection lines**
        for inset in prev_insets:
            inset.remove()
        prev_insets.clear()

        for line in prev_markings:
            line.remove()
        prev_markings.clear()


        if args.continuous_state:
            ax[0].pcolormesh(t.squeeze(), xx, y.T)
            ax[1].pcolormesh(t.squeeze(), xx, y_pred.T)
        else:
            # **Plot y_true vs y_pred**
            n = model.output_dim ### model.state_dim if model.state_dim==2 else 1
            for k, ax_ in enumerate(ax1[:n]):
                ax_.plot(t, y_pred[:, k], c='orange', label='Model output')
                ax_.plot(t, y[:, k], 'b--', label='True state')
                ax_.set_ylabel(f"$x_{k+1}$")
                ax_.legend()

        # **Plot input u**
        ax1[-1].step(np.arange(0., time_horizon, delta), u[:-1], where='post')
        ax1[-1].set_ylabel("$u$")
        ax1[-1].set_xlabel("$t$")

        # **Plot delta (Error)**
        for k, ax_ in enumerate(ax2[:n]):
            ax_.plot(t, y[:, 0] - y_pred[:, k], c='blue') #, label=f'Advanced ({sq_error:.3f})')
            ax_.set_ylabel(f"$Δx_{k+1}$")
            ax_.legend()


        if n==2:
            # **Plot Coefficients Evolution**   ###############
            if mode_rnn!="old":
                ax3[0].plot(t, coeffs[:, 0], label=r"$\alpha_1$", color='purple')
                ax3[0].set_ylabel("Coefficient 1")
                ax3[0].legend()
                ax3[0].grid()

                ax3[1].plot(t, coeffs[:, 1], label=r"$\alpha_2$", color='green')
                ax3[1].set_ylabel("Coefficient 2")
                ax3[1].set_xlabel("$t$")
                ax3[1].legend()
                ax3[1].grid()


            # **Zoomed-in Insets for Initial Conditions (first 5% of data)**
            for i, ax in enumerate(ax2):
                inset = inset_axes(ax, width="30%", height="30%", loc="lower right", borderpad=1)
                inset.plot(t, y[:, i] - y_pred[:, i], color='black')

                # Focus the inset on the first part of the curve
                inset_xlim = (t[0], t[int(len(t) * 0.05)])
                inset_ylim = (np.min(y[:int(len(t) * 0.05), i] - y_pred[:int(len(t) * 0.05), i]) * 1.1,
                            np.max(y[:int(len(t) * 0.05), i] - y_pred[:int(len(t) * 0.05), i]) * 1.1)

                inset.set_xlim(inset_xlim)
                inset.set_ylim(inset_ylim)
                inset.grid()

                # Connect inset to the main plot and store references
                lines = mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="0.5")
                
                prev_insets.append(inset)  # Store inset reference
                prev_markings.extend(lines)  # Store connection line references
            #"""

            if others: 
                eigvals = np.linalg.eigvals(matrices)  # shape: [seq_len, state_dim]
                eig_real, eig_imag = eigvals.real, eigvals.imag
                t_vals = np.linspace(0, delta * (eigvals.shape[0] - 1), eigvals.shape[0])

                # Autovalore λ₁
                sc1 = ax4[0].scatter(eig_real[:, 0], eig_imag[:, 0], c=t_vals, cmap='viridis', s=20)
                ax4[0].set_xlabel("Re(λ₁)")
                ax4[0].set_ylabel("Im(λ₁)")
                ax4[0].set_title("Autovalore λ₁ nel tempo")
                ax4[0].grid()

                # Posizionamento colorbar a destra di ax4[0]
                divider1 = make_axes_locatable(ax4[0])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                fig4.colorbar(sc1, cax=cax1, label=r"seq_len")

                # Autovalore λ₂
                sc2 = ax4[1].scatter(eig_real[:, 1], eig_imag[:, 1], c=t_vals, cmap='plasma', s=20)
                ax4[1].set_xlabel("Re(λ₂)")
                ax4[1].set_ylabel("Im(λ₂)")
                ax4[1].set_title("Autovalore λ₂ nel tempo")
                ax4[1].grid()

                # Posizionamento colorbar a destra di ax4[1]
                divider2 = make_axes_locatable(ax4[1])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                fig4.colorbar(sc2, cax=cax2, label=r"seq_len")


                ax5[0].hist(coeffs[:, 0], bins=30, color="purple", alpha=0.7)
                ax5[0].set_title("γ₁ distribution")
                ax5[1].hist(coeffs[:, 1], bins=30, color="green", alpha=0.7)
                ax5[1].set_title("γ₂ distribution")


                sc = ax6.scatter(coeffs[:, 0], coeffs[:, 1], c=t, cmap="viridis", s=10)
                ax6.set_xlabel("γ₁")
                ax6.set_ylabel("γ₂")
                ax6.set_title("Phase plot: γ₁ vs γ₂")
                divider3 = make_axes_locatable(ax6)
                cax3 = divider3.append_axes("right", size="5%", pad=0.05)
                fig6.colorbar(sc, cax=cax3, label="Time")


        fig1.tight_layout()
        fig1.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)

        fig2.tight_layout()
        if mode_rnn!="old": 
            fig3.tight_layout()

        if others: 
            fig4.tight_layout()
            fig5.tight_layout()
            fig6.tight_layout()

        plt.show(block=False)       # or plt.draw(block=False)
        plt.pause(0.1)

        """
        print("Press SPACE to update the trajectory...")
        while True:
            key = plt.waitforbuttonpress()
            if key and plt.gcf().canvas.key_press_event.key == " ":
                break
        """
        # Wait for key press
        skip = False
        while not skip:
            skip = plt.waitforbuttonpress()
        #"""


def on_close_window(ev):
    plt.close('all')  # Close figures before exiting
    sys.exit(0)


if __name__ == '__main__':
    main()
