import torch

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from scipy.signal import find_peaks

from generate_data import parse_args, generate, make_trajectory_sampler


def main():
    args = parse_args()

    with open(args.settings, 'r') as f:
        settings: dict = yaml.load(f, Loader=yaml.FullLoader)

    sampler = make_trajectory_sampler(settings)
    postprocess = get_postprocess(settings["dynamics"]["name"])

    train_data, val_data, test_data = generate(args,
                                               sampler,
                                               postprocess=postprocess)

    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "settings": settings,
        "args": vars(args),
    }

    output_dir = Path("./data/")
    output_dir.mkdir(exist_ok=True)

    # Write to disk
    with open(output_dir.joinpath(args.output_name + ".pkl"), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_postprocess(dynamics: str):
    if dynamics.startswith("HodgkinHuxley"):
        if dynamics.endswith(("FS", "RSA", "IB")):
            return [
                rejection_sampling_single_neuron,
            ]
        elif dynamics.endswith(("FFE", "FBE")):
            return [
                rejection_sampling_two_neuron,
            ]
    return []


def rejection_sampling_single_neuron(data):
    for (k, y) in enumerate(data.state):
        p = y[:, 0].flatten()
        p_min = p.min()
        p = 1e-4 + ((p - p_min) / (p.max() - p_min))

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio), ))
        keep_idxs = (u <= (likelihood_ratio / lr_bound))
        keep_idxs[0] = True

        peaks, _ = find_peaks(y[:, 0])
        keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


def rejection_sampling_two_neuron(data):
    for (k, y) in enumerate(data.state):
        p_1 = y[:, 0].flatten()
        p_min = p_1.min()
        p_1 = ((p_1 - p_min) / (p_1.max() - p_min))

        p_2 = y[:, 5].flatten()
        p_min = p_2.min()
        p_2 = ((p_2 - p_min) / (p_2.max() - p_min))

        p = torch.maximum(p_1, p_2)

        likelihood_ratio = p / torch.mean(p)
        lr_bound = torch.max(likelihood_ratio)

        u = torch.rand((len(likelihood_ratio), ))
        keep_idxs = (u <= (likelihood_ratio / lr_bound))
        keep_idxs[0] = True

        for y_spiking in (y[:, 0], y[:, 5]):
            peaks, _ = find_peaks(y_spiking)
            keep_idxs[peaks] = True

        data.state[k] = y[keep_idxs, :]
        data.state_noise[k] = data.state_noise[k][keep_idxs, :]
        data.time[k] = data.time[k][keep_idxs, :]


if __name__ == '__main__':
    main()
