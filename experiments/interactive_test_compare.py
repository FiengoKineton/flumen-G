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


"""
026___default-code-same-dim / --wandb_2
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flow_model-vdp_test_data-vdp_test-790qo5ei:v30

vdp_TEST-r_3 / --wandb
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flow_model-vdp_test_data3-vdp_TEST-r_3-pv6yc6ip:v1



TEMP-without (dnn) / --wandb
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data3-xc0zpee1:v0

TEMP-without-2 (dnn) / --wandb (GOOD)
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data3-awyl5oha:v1

TEMP-with (dnn) / --wandb (good)
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data3-hjfz1nwh:v0

TEMP-with-2 (dnn) / --wandb
aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data3-nkxyl2cu:v0

------------------------------



"""


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file" "(or, if run with --wandb, path to a Weights & Biases artifact)")
    ap.add_argument('path_2', type=str, help="Path_2 to .pth file" "(or, if run with --wandb_2, path to a Weights & Biases artifact)")
    ap.add_argument('--print_info', action='store_true', help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb_2', action='store_true')
    ap.add_argument('--time_horizon', type=float, default=None)

    return ap.parse_args()


def get_metadata(wandb, path):
    if wandb:
        import wandb
        api = wandb.Api()
        model_artifact = api.artifact(path)      
        model_path = Path(model_artifact.download())

        model_run = model_artifact.logged_by()
        print(model_run.summary)
    else:
        model_path = Path(path)

    with open(model_path / "state_dict.pth", 'rb') as f:
        state_dict = torch.load(f, weights_only=True, map_location='cpu')
    with open(model_path / "metadata.yaml", 'r') as f:
        metadata: dict = yaml.load(f, Loader=yaml.FullLoader)

    return metadata, state_dict


def main():
    plt.ion()  # Enable interactive mode
    args = parse_args()

    num_times = 2 if args.time_horizon is None else 1          # default 2 | multiplied by the time_span

    metadata, state_dict = get_metadata(args.wandb, args.path)
    if args.wandb_2: metadata_2, state_dict_2 = get_metadata(args.wandb_2, args.path_2)
    
    #metadata["args"]["mode_rnn"] = 'old'
    print("metadata:")
    pprint(metadata)

    if args.wandb_2: 
        print("\nmetadata_2:")
        pprint(metadata_2)

    #mode_rnn = metadata.get("args", {}).get("mode_rnn", "old")

    if args.print_info:
        return

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    if args.wandb_2: 
        model_2 = CausalFlowModel(**metadata_2["args"])
        model_2.load_state_dict(state_dict_2)
        model_2.eval()

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

    time_horizon = args.time_horizon if args.time_horizon else num_times * metadata["data_args"]["time_horizon"]

    while True:
        time_integrate = time()
        x0, t, y, u = sampler.get_example(time_horizon=time_horizon, n_samples=int(1 + 100 * time_horizon))
        time_integrate = time() - time_integrate

        time_predict = time()
        x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(x0, t, u, delta)

        with torch.no_grad():
            y_pred, _, _ = model(x0_feed, u_feed, deltas_feed)
            if args.wandb_2: y_pred_2, _, _ = model_2(x0_feed, u_feed, deltas_feed)
            # model 2 

        y_pred = np.flip(y_pred.numpy(), 0)
        if args.wandb_2: y_pred_2 = np.flip(y_pred_2.numpy(), 0)

        
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]
        sq_error = model.output_dim * np.mean(np.square(y - y_pred))
        sq_error_2 = model.output_dim * np.mean(np.square(y - y_pred_2))
        print("MSE (advance prediction):", sq_error)
        print("MSE (default prediction):", sq_error_2, "\n")

        # **Clear previous plots and remove insets & connections**
        for ax_ in ax1:
            ax_.cla()            

        if args.continuous_state:
            ax1[0].pcolormesh(t.squeeze(), xx, y.T)
            ax1[1].pcolormesh(t.squeeze(), xx, y_pred.T)
            if args.wandb_2: ax1[1].pcolormesh(t.squeeze(), xx, y_pred_2.T)
        else:
            # **Plot y_true vs y_pred**
            n = model.output_dim if model.state_dim==2 else 1
            colors = ['red', 'blue', 'green', 'red']  # extend if needed
            linestyles = ['-', '-.', ':', '--']  # different line styles

            for k, ax_ in enumerate(ax1[:n]):
                # Predicted (Advanced)
                ax_.plot(t, y_pred[:, k], color=colors[0], linestyle=linestyles[0], 
                        label=f'Advanced ({sq_error:.3f})')

                # Predicted (Default), optional
                if args.wandb_2:
                    ax_.plot(t, y_pred_2[:, k], color=colors[1], linestyle=linestyles[1],
                            label=f'Default ({sq_error_2:.3f})')

                # True state trajectory
                ax_.plot(t, y[:, k], color=colors[2], linestyle=linestyles[2], 
                        label='True state')

                ax_.set_ylabel(f"$x_{k+1}$")
                ax_.legend()


        # **Plot input u**
        ax1[-1].step(np.arange(0., time_horizon, delta), u[:-1], where='post')
        ax1[-1].set_ylabel("$u$")
        ax1[-1].set_xlabel("$t$")

        fig1.tight_layout()
        fig1.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig1.axes[:-1]], visible=False)

        plt.show(block=False)       # or plt.draw(block=False)
        plt.pause(0.1)

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
