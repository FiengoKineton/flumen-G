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
import pprint


from hyperparams import Hyperparams  
hp_manager = Hyperparams()
hyperparams = hp_manager.get_hyperparams('hyperparams___swift_sweep_1')


SWEEP = False
# --------------------------------------------------------------------------- #


sweep_config = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'control_rnn_size': {'values': [8, 12, 20]}, 
        'control_rnn_depth': {'values': [1]}, 
        'encoder_size': {'values': [1, 2]},  
        'encoder_depth': {'values': [1, 2]},  
        'decoder_size': {'values': [1, 2]},  
        'decoder_depth': {'values': [1, 2]},  
        'batch_size': {'values': [64, 128, 256]},
        'lr': {'values': [0.001, 0.0005, 0.0001, 0.002]},
        'n_epochs': {'values': [500, 1000]},
        'es_patience': {'values': [10, 20]}, 
        'es_delta': {'values': [1e-7, 1e-5]}, 
        'sched_patience': {'values': [10]},
        'sched_factor': {'values': [2]},
        'loss': {'values': ["mse", "l1"]},  
        'optimiser_mode': {'values': ["adam", "nesterov", "newton"]},
        'discretisation_mode': {'values': ["TU", "FE"]},
        'x_update_mode': {'values': ["alpha", "beta"]},
    }
}




def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()
    elif which == "l1":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss {which}.")


def main(sweep):
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

    if sweep:
        run = wandb.init(entity='aguiar-kth-royal-institute-of-technology', 
                        project='g7-fiengo-msc-thesis', 
                        name=sys_args.name)                                
                        ###, config=hyperparams)
        config = wandb.config
    else:
        run = wandb.init(entity='aguiar-kth-royal-institute-of-technology', 
                        project='g7-fiengo-msc-thesis', 
                        name=sys_args.name,                             
                        config=hyperparams)   

    with data_path.open('rb') as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])

    mhu = data["settings"]["dynamics"]["args"]["damping"]
    dyn_factor = data["settings"]["control_delta"]


    # normally wandb.config["param_name"], config.param_name when sweep
    if sweep:
        __control_rnn_size = config.control_rnn_size
        __control_rnn_depth = config.control_rnn_depth
        __encoder_size = config.encoder_size
        __encoder_depth = config.encoder_depth
        __decoder_size = config.decoder_size
        __decoder_depth = config.decoder_depth
        __batch_size = config.batch_size
        __lr = config.lr
        __n_epochs = config.n_epochs
        __es_patience = config.es_patience
        __es_delta = config.es_delta
        __sched_patience = config.sched_patience
        __sched_factor = config.sched_factor
        __loss = config.loss
        __discretisation_mode = config.discretisation_mode
        __optimiser_mode = config.optimiser_mode
        __x_update_mode = config.x_update_mode
    else:
        __control_rnn_size = wandb.config["control_rnn_size"]
        __control_rnn_depth = wandb.config["control_rnn_depth"]
        __encoder_size = wandb.config["encoder_size"]
        __encoder_depth = wandb.config["encoder_depth"]
        __decoder_size = wandb.config["decoder_size"]
        __decoder_depth = wandb.config["decoder_depth"]
        __batch_size = wandb.config["batch_size"]
        __lr = wandb.config["lr"]
        __n_epochs = wandb.config["n_epochs"]
        __es_patience = wandb.config["es_patience"]
        __es_delta = wandb.config["es_delta"]
        __sched_patience = wandb.config["sched_patience"]
        __sched_factor = wandb.config["sched_factor"]
        __loss = wandb.config["loss"]
        __discretisation_mode = wandb.config["discretisation_mode"]
        __optimiser_mode = wandb.config["optimiser_mode"]
        __x_update_mode = wandb.config["x_update_mode"]


    model_args = {
        'state_dim': train_data.state_dim,
        'control_dim': train_data.control_dim,
        'output_dim': train_data.output_dim,
        'control_rnn_size': __control_rnn_size,
        'control_rnn_depth': __control_rnn_depth,
        'encoder_size': __encoder_size,
        'encoder_depth': __encoder_depth,
        'decoder_size': __decoder_size,
        'decoder_depth': __decoder_depth,
        'discretisation_mode': __discretisation_mode,
        'x_update_mode': __x_update_mode,
        'mhu': mhu,
        'dyn_factor': dyn_factor,
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
    optimiser_mode = __optimiser_mode  # wandb.config['optimiser_mode']

    if optimiser_mode == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=__lr)
    elif optimiser_mode == "tbptt":
        optimiser = torch.optim.Adam(model.parameters(), lr=__lr)
    elif optimiser_mode == "nesterov":
        optimiser = torch.optim.SGD(model.parameters(), lr=__lr, momentum=0.9, nesterov=True)
    elif optimiser_mode == "newton":
        optimiser = torch.optim.LBFGS(model.parameters())
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=__lr)
        raise ValueError(f"Unknown optimizer mode: {optimiser_mode}. Choose from: adam, sgd_nesterov, lbfgs.")
    # --------------------------------------------------------------------------- #
    #"""

    wandb.log({
        'batch_size': __batch_size,
        'lr': __lr,
        'n_epochs': __n_epochs
    })

    ###optimiser = torch.optim.Adam(model.parameters(), lr=wandb.config['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        patience=__sched_patience,
        cooldown=0,
        factor=1. / __sched_factor)

    loss = get_loss(__loss).to(device)

    early_stop = EarlyStopping(es_patience=__es_patience,
                               es_delta=__es_delta)

    bs = __batch_size
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

    for epoch in range(__n_epochs):
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

            if sweep: 
                artifact = wandb.Artifact("model_checkpoint", type="model")
                artifact.add_file(str(model_save_dir / "state_dict.pth"))
                wandb.log_artifact(artifact)                
            else: 
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


def train_sweep():
    with wandb.init(entity='aguiar-kth-royal-institute-of-technology', 
                     project='g7-fiengo-msc-thesis'):
        main(sweep=True)

if __name__ == '__main__':
    print_gpu_info()

    if SWEEP: 
        sweep_id = wandb.sweep(sweep_config, 
                            entity='aguiar-kth-royal-institute-of-technology', 
                            project='g7-fiengo-msc-thesis')
        wandb.agent(sweep_id, train_sweep, count=10)

    else: main(SWEEP)
