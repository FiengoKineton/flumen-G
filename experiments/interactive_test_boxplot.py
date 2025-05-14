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
RUNS: 

VDP----------------
(lpv) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lbz1tnpu:v3 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0 --note lpv
(static) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-lwqp2l3z:v3 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-vdp_test_data-x3zk3ip4:v0 --note static
(upd_new) python .\experiments\interactive_test_boxplot.py --wandb _ --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v10 --note upd_new

FHN----------------
(stat) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-04y8vw0k:v4 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v6 --note static
(upd_new) python.exe .\experiments\interactive_test_boxplot.py --wandb _ --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-fhn_test_data-w52hqxkd:v6 --note upd_new

NAD----------------
(big) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-3dxiz9gf:v2 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-jwlwuqmw:v10
(big_upd_new) python.exe .\experiments\interactive_test_boxplot.py --wandb _ --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-jwlwuqmw:v10 --note upd_new
(small) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-zshs5333:v0 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v10
(small_upd_new) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-siq3pyyf:v8 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-nad_test_data-mg4z6swx:v10 --note upd_new

R3D12--------------
(r3d12_static) python.exe .\experiments\interactive_test_boxplot.py --wandb aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-r3d12_test_data-ku1s1thr:v1 --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-r3d12_test_data-gaciweh0:v0
(r3d12_upd_new) python.exe .\experiments\interactive_test_boxplot.py --wandb _ --wandb_2 aguiar-kth-royal-institute-of-technology/g7-fiengo-msc-thesis/flumen-r3d12_test_data-gaciweh0:v0

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
    ap.add_argument('--test_noise_std', type=float, default=0.0, help="Add Gaussian noise to test inputs during evaluation.")
    ap.add_argument('--note', type=str, default=None)
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
    NOTE = args.note
    NOISE_STD = args.test_noise_std
    WANDB_1, WANDB_2 = 'new_architecture', 'default_architecture'   # 'old_architecture', 'default' 
    test_sin = True
    N, i = 100, 0

    num_times = 2 if args.time_horizon is None else 1          # default 2 | multiplied by the time_span

    metadata, state_dict = get_metadata(args.wandb, args.path)
    metadata_2, state_dict_2 = get_metadata(args.wandb_2, args.path_2)
    
    #print("metadata:"), pprint(metadata)
    #print("\nmetadata_2:"), pprint(metadata_2)

    if args.print_info:
        return

    model = CausalFlowModel(**metadata["args"])
    model.load_state_dict(state_dict)
    model.eval()

    model_2 = CausalFlowModel(**metadata_2["args"])
    model_2.load_state_dict(state_dict_2)
    model_2.eval()

    if test_sin: 
        metadata["data_settings"]["sequence_generator"] = {
            "name": "SinusoidalSequence",
            "args": {
                "max_freq": 0.2
            }
    }
        
    sampler = make_trajectory_sampler(metadata["data_settings"])
    sampler.reset_rngs()
    delta = sampler._delta

    n_plots = model.output_dim
    model_name = metadata['data_settings']['dynamics']['name']


    err_list_1, err_list_2 = [], []

    if n_plots==2:      n = n_plots
    elif n_plots==12:   n = 3
    else:               n = 1

    if args.time_horizon: 
        time_horizon = args.time_horizon 
    else: 
        time_horizon = num_times * metadata["data_args"]["time_horizon"] if n!=3 else metadata["data_args"]["time_horizon"]

    print(f'output dimension for {model_name}: {n}')
    stop = False
    while not stop:
        i = i+1
        time_integrate = time()
        x0, t, y, u = sampler.get_example(time_horizon=time_horizon, n_samples=int(1 + 100 * time_horizon))
        time_integrate = time() - time_integrate

        time_predict = time()
        u = u + np.random.randn(*u.shape) * NOISE_STD
        x0_feed, _, u_feed, deltas_feed = pack_model_inputs(x0, t, u, delta)

        with torch.no_grad():
            y_pred, _, _ = model(x0_feed, u_feed, deltas_feed)
            if args.wandb_2: y_pred_2, _, _ = model_2(x0_feed, u_feed, deltas_feed)
            # model 2 

        y_pred = np.flip(y_pred.numpy(), 0)
        y_pred_2 = np.flip(y_pred_2.numpy(), 0)

        
        time_predict = time() - time_predict

        err, err_2 = np.zeros(n), np.zeros(n)
        for k in range(n):
            err[n-1-k] = model.output_dim * np.mean(np.square(y[:, n-1-k] - y_pred[:, n-1-k]))
            err_2[n-1-k] = model.output_dim * np.mean(np.square(y[:, n-1-k] - y_pred_2[:, n-1-k]))

        if n==2:
            err_list_1.append([err[n-2], err[n-1]])
            err_list_2.append([err_2[n-2], err_2[n-1]])
        elif n==3:
            err_list_1.append([err[n-3], err[n-2], err[n-1]])
            err_list_2.append([err_2[n-3], err_2[n-2], err_2[n-1]])
        else: 
            err_list_1.append([err[n-1]])
            err_list_2.append([err_2[n-1]])

        if i >= N: stop = True
        print(f'\tcount over {N} = {i}')


    err_array_1 = np.array(err_list_1) 
    err_array_2 = np.array(err_list_2) 

    if n==2:
        data_to_plot = [
            err_array_1[:, 0],  # x1 errors - method 1
            err_array_2[:, 0],  # x1 errors - method 2
            err_array_1[:, 1],  # x2 errors - method 1
            err_array_2[:, 1],  # x2 errors - method 2
        ]

        # Labels for x-axis
        labels = [f'{WANDB_1} $x_1$', f'{WANDB_2} $x_1$', f'{WANDB_1} $x_2$', f'{WANDB_2} $x_2$']
    elif n==3:
        data_to_plot = [
            err_array_1[:, 0],  # x10 errors - method 1
            err_array_2[:, 0],  # x10 errors - method 2
            err_array_1[:, 1],  # x11 errors - method 1
            err_array_2[:, 1],  # x11 errors - method 2
            err_array_1[:, 2],  # x12 errors - method 1
            err_array_2[:, 2],  # x12 errors - method 2
        ]

        # Labels for x-axis
        m = model.output_dim
        labels = [f'{WANDB_1} $x_{{{m-2}}}$', f'{WANDB_2} $x_{{{m-2}}}$', f'{WANDB_1} $x_{{{m-1}}}$', f'{WANDB_2} $x_{{{m-1}}}$', f'{WANDB_1} $x_{{{m}}}$', f'{WANDB_2} $x_{{{m}}}$']
    else:
        data_to_plot = [
            err_array_1[:, 0],  # x_15 errors - method 1
            err_array_2[:, 0],  # x_15 errors - method 2
        ]

        m = model.output_dim
        labels = [f'{WANDB_1} $x_{{{m}}}$', f'{WANDB_2} $x_{{{m}}}$']

    fig, ax = plt.subplots(1, 1, sharex=True)
    fig.canvas.mpl_connect('close_event', on_close_window)

    # Create boxplot
    ax.boxplot(data_to_plot, patch_artist=True, showfliers=True)

    # Set x-axis labels
    if n==2: ax.set_xticks([1, 2, 3, 4])
    elif n==3: ax.set_xticks([1, 2, 3, 4, 5, 6])
    else: ax.set_xticks([1, 2])

    ax.set_xticklabels(labels, rotation=15)

    # Optional: log scale for y-axis if the values span multiple orders
    ax.set_yscale('log')
    ax.set_ylabel('MSE')
    if NOTE is None:    ax.set_title(f'{model_name} error distributions over {N} runs')
    else:               ax.set_title(f'{model_name} error distributions over {N} runs ({NOTE})')

    plt.tight_layout()
    plt.show(block=True)


def on_close_window(ev):
    plt.close('all')  # Close figures before exiting
    sys.exit(0)


if __name__ == '__main__':
    main()
