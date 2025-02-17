# A neural architecture for approximating flows of dynamical systems with inputs

See our paper [Miguel Aguiar, Amritam Das and Karl H. Johansson, _Learning Flow Functions from Data with Applications to Nonlinear Oscillators_ (2023)](https://www.sciencedirect.com/science/article/pii/S240589632302147X) for a description of the architecture.

The `flumen` package provides the PyTorch module implementing the architecture,
and some auxiliary functions which can be used for training.

This repository also contains scripts to train a model using synthetic data.
To generate the data, the [`semble`](https://github.com/mcpca/semble) package is required.
To learn e.g. the Van der Pol dynamics, first create a data file as follows:
```shell
  python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 data_generation/vdp.yaml vdp_test_data
```
This will create a data file in `./data/vdp_test_data.pkl`.

The script `experiments/train_wandb.py` provides an example of training the model using PyTorch.
The [`wandb`](https://pypi.org/project/wandb/) package is required to run the script.
Set the environment variable `WANDB_DISABLED=true` if you do not have a Weights & Biases account or do not want to log the results.
Then, run the script as follows:
```shell
  python experiments/train_wandb.py data/vdp_test_data.pkl vdp_test
```
This will create a directory in `./outputs/vdp_standard` containing the model parameters and some metadata.
To simulate the trained model, you can use the help script `experiments/interactive_test.py`.
