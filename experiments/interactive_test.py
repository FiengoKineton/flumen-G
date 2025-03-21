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


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--print_info', action='store_true', help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')

    return ap.parse_args()


def main():
    plt.ion()  # Enable interactive mode
    args = parse_args()

    num_times = 2           # default 2

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
        state_dict = torch.load(f, weights_only=True)
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)
    
    if "dyn_matrix" in metadata["args"] and not isinstance(metadata["args"]["dyn_matrix"], torch.Tensor):
        metadata["args"]["dyn_matrix"] = torch.tensor(metadata["args"]["dyn_matrix"])

    pprint(metadata)

    mode_rnn = metadata.get("args", {}).get("mode_rnn", "new")  # ["args"]["mode_rnn"]

    if args.print_info:
        return

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    # First Figure: y_true vs y_pred + Input
    fig1, ax1 = plt.subplots(3, 1, sharex=True)
    fig1.canvas.mpl_connect('close_event', on_close_window)

    # Second Figure: Delta Plots
    fig2, ax2 = plt.subplots(2, 1, sharex=True)
    fig2.canvas.mpl_connect('close_event', on_close_window)

    # Third Figure: Coefficient Evolution (for alpha, beta, lambda) ###############
    if mode_rnn!="old":
        fig3, ax3 = plt.subplots(2, 1, sharex=True)
        fig3.canvas.mpl_connect('close_event', on_close_window)

    xx = np.linspace(0., 1., model.output_dim)
    time_horizon = num_times * metadata["data_args"]["time_horizon"]

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
            y_pred, coeffs = model(x0_feed, u_feed, deltas_feed)    ###############

        y_pred = np.flip(y_pred.numpy(), 0)
        coeffs = np.flip(coeffs.numpy(), 0) ###############
        if mode_rnn!="old": coeffs = coeffs[:, -1, :]   ###############
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]
        sq_error = np.square(y - y_pred)
        print(np.mean(sq_error))

        # **Clear previous plots and remove insets & connections**
        for ax_ in ax1:
            ax_.cla()
        for ax_ in ax2:
            ax_.cla()
        if mode_rnn!="old": 
            for ax_ in ax3: ax_.cla()  ###############

        # **Remove previous insets and connection lines**
        for inset in prev_insets:
            inset.remove()
        prev_insets.clear()

        for line in prev_markings:
            line.remove()
        prev_markings.clear()

        # **Plot y_true vs y_pred**
        for k, ax_ in enumerate(ax1[:model.state_dim]):
            ax_.plot(t, y_pred[:, k], c='orange', label='Model output')
            ax_.plot(t, y[:, k], 'b--', label='True state')
            ax_.set_ylabel(f"$x_{k+1}$")
            ax_.legend()

        # **Plot input u**
        ax1[-1].step(np.arange(0., time_horizon, delta), u[:-1], where='post')
        ax1[-1].set_ylabel("$u$")
        ax1[-1].set_xlabel("$t$")

        # **Plot delta (Error)**
        ax2[0].plot(t, y[:, 0] - y_pred[:, 0], label=r"$\Delta x_1$", color='red')
        ax2[0].set_ylabel("Delta x1")
        ax2[0].legend()
        ax2[0].grid()

        ax2[1].plot(t, y[:, 1] - y_pred[:, 1], label=r"$\Delta x_2$", color='blue')
        ax2[1].set_ylabel("Delta x2")
        ax2[1].set_xlabel("$t$")
        ax2[1].legend()
        ax2[1].grid()

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

        fig1.tight_layout()
        fig2.tight_layout()
        if mode_rnn!="old": fig3.tight_layout() ###############

        plt.show(block=False)
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
