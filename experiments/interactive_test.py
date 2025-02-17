import torch
import matplotlib.pyplot as plt
import numpy as np

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
    ap.add_argument(
        'path',
        type=str,
        help="Path to .pth file "
        "(or, if run with --wandb, path to a Weights & Biases artifact)")
    ap.add_argument('--print_info',
                    action='store_true',
                    help="Print training metadata and quit")
    ap.add_argument('--continuous_state', action='store_true')
    ap.add_argument('--wandb', action='store_true')

    return ap.parse_args()


def main():
    args = parse_args()

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

    pprint(metadata)

    if args.print_info:
        return

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    fig, ax = plt.subplots(3, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)
    xx = np.linspace(0., 1., model.output_dim)

    time_horizon = metadata["data_args"]["time_horizon"]

    while True:
        time_integrate = time()
        x0, t, y, u = sampler.get_example(time_horizon=time_horizon,
                                          n_samples=int(1 +
                                                        100 * time_horizon))
        time_integrate = time() - time_integrate

        time_predict = time()

        x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(
            x0, t, u, delta)

        with torch.no_grad():
            y_pred = model(x0_feed, u_feed, deltas_feed).numpy()

        y_pred = np.flip(y_pred, 0)
        time_predict = time() - time_predict

        print(f"Timings: {time_integrate}, {time_predict}")

        y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]

        sq_error = np.square(y - y_pred)
        print(model.state_dim * np.mean(sq_error))

        if args.continuous_state:
            ax[0].pcolormesh(t.squeeze(), xx, y.T)
            ax[1].pcolormesh(t.squeeze(), xx, y_pred.T)
        else:
            for k, ax_ in enumerate(ax[:model.state_dim]):
                ax_.plot(t, y_pred[:, k], c='orange', label='Model output')
                ax_.plot(t, y[:, k], 'b--', label='True state')
                ax_.set_ylabel(f"$x_{k+1}$")

        ax[-1].step(np.arange(0., time_horizon, delta), u[:-1], where='post')
        ax[-1].set_ylabel("$u$")
        ax[-1].set_xlabel("$t$")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        plt.draw()

        # Wait for key press
        skip = False
        while not skip:
            skip = plt.waitforbuttonpress()

        for ax_ in ax:
            ax_.clear()


def on_close_window(ev):
    sys.exit(0)


if __name__ == '__main__':
    main()
