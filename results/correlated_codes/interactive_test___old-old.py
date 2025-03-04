import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from flumen import Experiment, pack_model_inputs    # before: from flow_model
from semble import TrajectorySampler                # before: from trajectory_sampler
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('path', type=str, help="Path to .pth file")
    ap.add_argument('--print_info',
                    action='store_true',
                    help="Print training metadata and quit")
    ap.add_argument('--plot_loss',
                    action='store_true',
                    help="Plot loss and quit")

    return ap.parse_args()


def main():
    args = parse_args()

    experiment: Experiment = torch.load(args.path,
                                        map_location=torch.device('cpu'))

    print(experiment)

    if args.print_info:
        print(experiment.args_str())
        print(vars(experiment.generator._seq_gen))
        return

    model = experiment.load_model()

    fig, ax = plt.subplots()
    ax.semilogy(experiment.train_loss[1:], label='Train loss')
    ax.semilogy(experiment.val_loss[1:], label='Validation loss')
    ax.semilogy(experiment.test_loss[1:], label='Test loss')
    ax.legend()
    ax.vlines(np.argmin(experiment.val_loss[1:]),
              ymin=0,
              ymax=np.max(experiment.test_loss[1:]),
              colors='r',
              linestyles='dashed')
    ax.set_xlabel('Training epochs')
    fig.tight_layout()
    plt.show()

    if args.plot_loss:
        return

    model.eval()

    sampler: TrajectorySampler = experiment.generator.sampler
    sampler.reset_rngs()
    delta = sampler._delta

    fig, ax = plt.subplots(1 + model.output_dim, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)
    plt.ion()

    with torch.no_grad():
        while True:
            time_horizon = 4 * experiment.generator.time_horizon

            x0, t, y, u = sampler.get_example(
                time_horizon=time_horizon,
                n_samples=int(1 + 100 * time_horizon))

            x0_feed, t_feed, u_feed, deltas_feed = pack_model_inputs(
                x0, t, u, delta)

            y_pred = experiment.predict(model, x0_feed, u_feed, deltas_feed)

            y = y[:, tuple(bool(v) for v in sampler._dyn.mask)]

            sq_error = np.square(y - np.flip(y_pred, 0))
            print(np.mean(sq_error))

            for k, ax_ in enumerate(ax[:-1]):
                ax_.plot(t, y[:, k], 'b', label='True state')
                ax_.plot(t_feed, y_pred[:, k], c='orange', label='Prediction')
                ax_.set_ylabel(f"$x_{k+1}$")

            ax[0].legend()

            ax[-1].step(np.arange(0., time_horizon, delta),
                        u[:-1],
                        where='post')
            ax[-1].set_ylabel("$u$")
            ax[-1].set_xlabel("$t$")

            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]],
                     visible=False)

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
