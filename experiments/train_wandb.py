import torch
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from flumen import CausalFlowModel, print_gpu_info, TrajectoryDataset
from flumen.train import EarlyStopping, train_step, validate

from argparse import ArgumentParser
import time

import wandb

# --------------------------------------------------------------------------- #
import os
import pandas as pd
# --------------------------------------------------------------------------- #


hyperparams = {
    'control_rnn_size': 8,          ### default 12 | try 20 | 8 if self.mode_rnn="true" else 10
    'control_rnn_depth': 1,         ### maybe try 2? (num_layer == control_rnn_depth) --- Nope!
    'encoder_size': 1,
    'encoder_depth': 2,
    'decoder_size': 1,
    'decoder_depth': 2,
    'batch_size': 128,
    'lr': 0.001,                    ### try 5e-4 to increase stability
    'n_epochs': 1000,
    'es_patience': 20,              ### default 20
    'es_delta': 1e-7,
    'sched_patience': 10,
    'sched_factor': 2,
    'loss': "mse",
    'discretisation_mode': "TU",    #-- {TU, FE, BE}
    'optimiser_mode': "adam",       #-- {adam, tbptt, nesterov, newton}
    'x_update_mode': "beta",       #-- {alpha, beta, lamda}
}


def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss {which}.")


def main():
    ap = ArgumentParser()

    ap.add_argument('load_path', type=str, help="Path to trajectory dataset")

    ap.add_argument('name', type=str, help="Name of the experiment.")

    ap.add_argument('--reset_noise',
                    action='store_true',
                    help="Regenerate the measurement noise.")

    ap.add_argument('--noise_std',
                    type=float,
                    default=None,
                    help="If reset_noise is set, set standard deviation ' \
                            'of the measurement noise to this value.")

    sys_args = ap.parse_args()
    data_path = Path(sys_args.load_path)

    run = wandb.init(entity='aguiar-kth-royal-institute-of-technology', project='g7-fiengo-msc-thesis', name=sys_args.name, config=hyperparams)

    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])

    mhu = data["settings"]["dynamics"]["args"]["damping"]
    dyn_factor = data["settings"]["control_delta"]
    A = dyn_factor * torch.tensor([[mhu, -mhu], [1/mhu, 0]])

    model_args = {
        'state_dim': train_data.state_dim,
        'control_dim': train_data.control_dim,
        'output_dim': train_data.output_dim,
        'control_rnn_size': wandb.config['control_rnn_size'],
        'control_rnn_depth': wandb.config['control_rnn_depth'],
        'encoder_size': wandb.config['encoder_size'],
        'encoder_depth': wandb.config['encoder_depth'],
        'decoder_size': wandb.config['decoder_size'],
        'decoder_depth': wandb.config['decoder_depth'],
        'discretisation_mode': wandb.config['discretisation_mode'],
        'x_update_mode': wandb.config['x_update_mode'],
        'dyn_matrix': A,
        'use_batch_norm': False,
    }

    model_metadata = {
        'args': model_args,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }
    model_name = f"flow_model-{data_path.stem}-{sys_args.name}-{run.id}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}_{run.id}")
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", 'w') as f:
        yaml.dump(model_metadata, f)

    model = CausalFlowModel(**model_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    #"""
    # --------------------------------------------------------------------------- #
    optimiser_mode = wandb.config['optimiser_mode']

    if optimiser_mode == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    elif optimiser_mode == "tbptt":
        optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    elif optimiser_mode == "nesterov":
        optimiser = torch.optim.SGD(model.parameters(), lr=wandb.config['lr'], momentum=0.9, nesterov=True)
    elif optimiser_mode == "newton":
        optimiser = torch.optim.LBFGS(model.parameters())
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
        raise ValueError(f"Unknown optimizer mode: {optimiser_mode}. Choose from: adam, sgd_nesterov, lbfgs.")
    # --------------------------------------------------------------------------- #
    #"""


    ###optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=wandb.config['sched_patience'],
        cooldown=0,
        factor=1. / wandb.config['sched_factor'])

    loss = get_loss(wandb.config["loss"]).to(device)

    early_stop = EarlyStopping(es_patience=wandb.config['es_patience'],
                               es_delta=wandb.config['es_delta'])

    bs = wandb.config['batch_size']
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)

    header_msg = f"{'Epoch':>5} :: {'Loss (Train)':>16} :: " \
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss = validate(train_dl, loss, model, device)   
    val_loss = validate(val_dl, loss, model, device)       
    test_loss = validate(test_dl, loss, model, device)     

    early_stop.step(val_loss)
    print(
        f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
        f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
    )

    start = time.time()

    for epoch in range(wandb.config['n_epochs']):
        model.train()
        for example in train_dl:        # for i in range 190
            ###print("check")
            train_step(example, loss, model, optimiser, device)
    
            """   
        # --------------------------------------------------------------------------- #
            ...
        # --------------------------------------------------------------------------- #
            #"""

        model.eval()
        train_loss = validate(train_dl, loss, model, device)   
        val_loss = validate(val_dl, loss, model, device)       
        test_loss = validate(test_dl, loss, model, device)     

        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}"
        )

        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")
            run.log_model(model_save_dir.as_posix(), name=model_name)

            run.summary["best_train"] = train_loss
            run.summary["best_val"] = val_loss
            run.summary["best_test"] = test_loss
            run.summary["best_epoch"] = epoch + 1

        wandb.log({
            'time': time.time() - start,
            'epoch': epoch + 1,
            'lr': optimiser.param_groups[0]['lr'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")

    
    """
    # --------------------------------------------------------------------------- #
    ...
    # --------------------------------------------------------------------------- #
    #"""

if __name__ == '__main__':
    print_gpu_info()
    main()
