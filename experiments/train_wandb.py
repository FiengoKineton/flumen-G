import torch
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float32)

import pickle, yaml
from pathlib import Path

from flumen import CausalFlowModel, print_gpu_info, TrajectoryDataset
from flumen.train import EarlyStopping, train_step, validate

from argparse import ArgumentParser
import time
import re

import wandb
import psutil

import random
import numpy as np
import sys

# --------------------------------------------------------------------------- #
"""
COMMANDs:

--VDP-------------------------------    vdp_test3-data3 / vdp_TEST-r_3 / 026___default-code-same-dim / vdp_fin-old
python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/vdp.yaml vdp_test_data
python experiments/train_wandb.py data/vdp_test_data.pkl vdp_test 

--FHN-------------------------------    fhn_data2 / fhn_001--set_4-old
python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/fhn.yaml fhn_test_data
python experiments/train_wandb.py data/fhn_test_data.pkl fhn_test 

--nad-------------------------------    nad_old / nad_FE_stat / nad_fin-01
python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/nad.yaml nad_test_data
python.exe experiments/train_wandb.py data/nad_test_data.pkl nad

--linsys----------------------------
python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/linsys.yaml linsys_test_data
python experiments/train_wandb.py data/linsys_test_data.pkl linsys_test 

--HHfs------------------------------
python experiments/semble_generate.py --n_trajectories 200 --n_samples 200 --time_horizon 15 data_generation/hhfs.yaml hhfs_test_data
python experiments/train_wandb.py data/hhfs_test_data.pkl hhfs_test 


--general---------------------------
python experiments/train_wandb.py data/{model_name}_test_data.pkl {run_name} 
{model_name}: [vdp, fhn, twotank, linsys, hhfs, hhffe, greenshields]

Note: to test with smaller DS, use --n_trajectories 150
"""

import os
import pandas as pd
from pprint import pprint
import torch_optimizer as optim


from hyperparams import Hyperparams  

os.environ["WANDB_MODE"] = "online"  # Set to "online" for real-time logging, "offline" for local logging

# ------ Current Run Settings ----------------------------------------------- #
hp_manager = Hyperparams()  

sets = {
    'init': 'hyperparams___init',
    'vdp': 'hyperparams___vdp',             # use TU and lpv!
    'nad': 'hyperparams___nad',             # use FE and static!
    'fhn': 'hyperparams___fhn',             # in progress
    'vdp_old': 'hyperparams___vdp_old', 
    'fhn_old': 'hyperparams___fhn_old', 
    'nad_old': 'hyperparams___nad_old'
}

sweeps = {
    'init': 'sweep_config_init', 
    'test1': 'sweep_config_test_1',
    'test2': 'sweep_config_test_2',
    'test3': 'sweep_config_test_3',
    'new': 'sweep_config_test_4_new', 
    'old': 'sweep_config_test_4_old',
    'vdp': 'sweep_config_vdp',
    'fhn': 'sweep_config_fhn',
}


name_set = sets['fhn']     # vdp, fhn, nad
hyperparams = hp_manager.get_hyperparams(name_set)

name_sweep = sweeps['fhn']
sweep_config = hp_manager.get_sweep(name_sweep)
num_sweeps = 8
SWEEP = False


if SWEEP:
    print("\nSWEEP: ", SWEEP, "\n\n", f"{name_sweep} --- num_sweeps: {num_sweeps}")
    pprint(sweep_config)
    print("\n\n")
else:
    print("\nSWEEP: ", SWEEP, "\n\n", f"{name_set}:")
    pprint(hyperparams)
    print("\n\n")
# --------------------------------------------------------------------------- #


# ------ Execution Performance Summary -------------------------------------- #
def get_initial_metrics():
    """ Capture initial system metrics before execution starts. """
    process = psutil.Process()
    return {
        "start_time": time.time(),
        "start_memory": process.memory_info().rss,  # Bytes
        "start_cpu": process.cpu_percent(interval=None),  # No wait, first read
        "start_disk": psutil.disk_usage('/').used,  # Bytes
        "start_net": psutil.net_io_counters(),
        "system_mem_start": psutil.virtual_memory()  # Full system memory stats
    }

def format_time(seconds):
    """ Convert seconds into hours, minutes, and seconds. """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = seconds % 60  # Keep decimals for milliseconds precision
    return f"{hours}h {minutes}min {sec:.1f}sec"

def print_system_performance(initial_metrics):
    """ Print system resource usage after execution and compare with initial. """
    process = psutil.Process()

    # Capture final metrics
    end_time = time.time()
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    cpu_percent = process.cpu_percent(interval=1)  # Final CPU measurement
    disk_usage = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()

    # Calculate differences
    time_elapsed = end_time - initial_metrics["start_time"]
    formatted_time = format_time(time_elapsed)
    memory_used = (mem_info.rss - initial_metrics["start_memory"]) / (1024 ** 2)  # MB
    disk_used = (disk_usage.used - initial_metrics["start_disk"]) / (1024 ** 3)  # GB
    net_sent = (net_io.bytes_sent - initial_metrics["start_net"].bytes_sent) / (1024 ** 2)  # MB
    net_recv = (net_io.bytes_recv - initial_metrics["start_net"].bytes_recv) / (1024 ** 2)  # MB

    # Print summary
    print("\n===== SYSTEM PERFORMANCE SUMMARY =====")
    print(f"Total Execution Time: {formatted_time}")
    print(f"Total Memory Allocation: {system_mem.total / (1024 ** 3):.2f} GB")
    print(f"Process Memory In Use: {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"Memory Used During Execution: {memory_used:.2f} MB")
    print(f"System Memory Utilization: {system_mem.percent}%")
    print(f"Process CPU Utilization: {cpu_percent}%")
    print(f"Disk Utilization: {disk_usage.used / (1024 ** 3):.2f} GB / {disk_usage.total / (1024 ** 3):.2f} GB")
    print(f"Disk Space Used During Execution: {disk_used:.2f} GB")
    print(f"Network Sent: {net_io.bytes_sent / (1024 ** 2):.2f} MB")
    print(f"Network Received: {net_io.bytes_recv / (1024 ** 2):.2f} MB")
    print(f"Network Traffic Sent During Execution: {net_sent:.2f} MB")
    print(f"Network Traffic Received During Execution: {net_recv:.2f} MB")
    print("======================================\n")
# --------------------------------------------------------------------------- #


# ------ Loss Function ------------------------------------------------------ #
def get_loss(which):
    if which == "mse":
        return torch.nn.MSELoss()   #########################################################
    elif which == "l1":
        return torch.nn.L1Loss()
    elif which == "huber": 
        return torch.nn.HuberLoss()        
    else:
        raise ValueError(f"Unknown loss {which}.")
# --------------------------------------------------------------------------- #


# ------ GLOBAL SEED SETTER FOR REPRODUCIBILITY ----------------------------- #
def set_seed(seed=42):
    """
    Sets global random seeds across Python, NumPy, PyTorch, and CUDA to ensure 
    full reproducibility of training runs. This controls all sources of 
    randomness such as weight initialization, data shuffling, dropout, 
    and GPU behavior.

    Args:
        seed (int): The seed value to use (default is 42).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# --------------------------------------------------------------------------- #


# ------ Run Processing ----------------------------------------------------- #
def main(sweep):
    initial_metrics = get_initial_metrics()     # Execution Performance Summary
    #set_seed(42)                                # usefull?

    ap = ArgumentParser()

    ap.add_argument('load_path', type=str, help="Path to trajectory dataset")

    ap.add_argument('name', type=str, help="Name of the experiment.")       # type=str, nargs='+',

    ap.add_argument('--reset_noise',
                    action='store_true',
                    help="Regenerate the measurement noise.")

    ap.add_argument('--noise_std',
                    type=float,
                    default=None,
                    help="If reset_noise is set, set standard deviation ' \
                            'of the measurement noise to this value.")

    ap.add_argument('--model_log_rate', type=int, default=15)                       # not pulled from flumen
    
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


        # CHNAGED
        first_name = sys_args.name[0]
        full_name = '_'.join(sys_args.name)
        full_name = re.sub("[^a-zA-Z0-9_-]", '_', full_name)
    
        # CHECK THIS OUT !!
        #run = wandb.init(project='flumen', config=hyperparams, name=full_name)

    with data_path.open('rb') as f:
        data = pickle.load(f)
    
    model_name = data["settings"]["dynamics"]["name"]
    print("\nmodel name:", model_name, "\n")
    pprint(data)
    print("\n\n")

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])

    """
    # ----------------------------------------------------------------------- #
    from torch.nn.utils.rnn import pad_packed_sequence
    #from sippy import system_identification as sysid

    train_dl = DataLoader(train_data, batch_size=wandb.config["batch_size"], shuffle=True)
    sd = train_data.state_dim
    all_u, all_y = [], []

    def prep_inputs(x0, y, u, lengths):
        sort_idxs = torch.argsort(lengths, descending=True)

        x0 = x0[sort_idxs]
        y = y[sort_idxs]
        u = u[sort_idxs]
        lengths = lengths[sort_idxs]

        deltas = u[:, :lengths[0], -1].unsqueeze(-1)

        u = torch.nn.utils.rnn.pack_padded_sequence(u,
                                                    lengths,
                                                    batch_first=True,
                                                    enforce_sorted=True)
        return x0, y, u, deltas

    def estimate_bla(train_dl, n_order=2):
        for example in train_dl:
            _, y, u, _ = prep_inputs(*example)
            u, lengths = pad_packed_sequence(u, batch_first=True)
            u = u[:, -1, :]

            #print(y.shape)          # torch.Size([96, 2])
            #print(u.shape)          # torch.Size([96, 75, 2])
            #print(lengths.shape)    # torch.Size([96])

            for i in range(y.shape[0]):
                seq_len = lengths[i]
                u_i = u[i, :seq_len]  # (seq_len, control_dim)
                y_i = y[i, :seq_len]  # (seq_len, output_dim)

                all_u.append(u_i)
                all_y.append(y_i)

        u_all = torch.cat(all_u, dim=0)  # (N, control_dim)
        y_all = torch.cat(all_y, dim=0)  # (N, output_dim)

        u_tilde = u_all - u_all.mean(dim=0, keepdim=True)
        y_tilde = y_all - y_all.mean(dim=0, keepdim=True)

        X = u_tilde.unfold(0, n_order, 1)  # shape: (N - n_order + 1, n_order, control_dim)
        Y = y_tilde[n_order-1:]  # shape: (N - n_order, output_dim) 

        G_bla = torch.linalg.lstsq(X, Y).solution
        #model = sysid.N4SID(Y, X, dt=1.0, SS_fixed_order=sd)
        #model.show()
        #sys.exit()
        return G_bla, 0 #model
    
    G_bla, model = estimate_bla(train_dl, n_order=sd)
    print("G_bla shape:", G_bla.shape)
    print("G_bla:", G_bla)
    #print(model.A, model.B, model.C, model.D)
    sys.exit()
    # ----------------------------------------------------------------------- #
    # """


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
        __mode_rnn = config.mode_rnn
        __mode_dnn = config.mode_dnn
        __linearisation_mode = config.linearisation_mode
        __decoder_mode = config.decoder_mode
        __radius = config.radius
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
        __mode_rnn = wandb.config["mode_rnn"]
        __mode_dnn = wandb.config["mode_dnn"]
        __linearisation_mode = wandb.config["linearisation_mode"]
        __decoder_mode = wandb.config["decoder_mode"]
        __radius = wandb.config["radius"]


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
        'model_name': model_name,     #------------#
        'mode_rnn': __mode_rnn,
        'mode_dnn': __mode_dnn,
        'use_batch_norm': False,
        'linearisation_mode': __linearisation_mode, 
        'batch_size': __batch_size,
        'decoder_mode': __decoder_mode,
        'radius': __radius,
    }

    model_metadata = {
        'args': model_args,
        'data_path': data_path.absolute().as_posix(),
        'data_settings': data["settings"],
        'data_args': data["args"]
    }
    ###model_name = f"flow_model-{data_path.stem}-{sys_args.name}-{run.id}"
    model_name = f"flumen-{data_path.stem}-{run.id}"

    # Prepare for saving the model
    model_save_dir = Path(
        f"./outputs/{sys_args.name}/{sys_args.name}_{run.id}")
        ###f"./outputs/{first_name}/{full_name}_{run.id}")
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
        optimiser = torch.optim.Adam(model.parameters(), lr=__lr)#, weight_decay=1e-4)   #########################################################
    elif optimiser_mode == "tbptt":
        optimiser = torch.optim.Adam(model.parameters(), lr=__lr)
    elif optimiser_mode == "nesterov":
        optimiser = torch.optim.SGD(model.parameters(), lr=__lr, momentum=0.9, nesterov=True)
    elif optimiser_mode == "newton":
        optimiser = torch.optim.LBFGS(model.parameters())
    elif optimiser_mode == "lamb":      
        optimiser = optim.Lamb(model.parameters(), lr=__lr)
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
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}" ### :: {'Coeff(Train)':>16}"  ###############

    print(header_msg)
    print('=' * len(header_msg))

    # Evaluate initial loss
    model.eval()
    train_loss = validate(train_dl, loss, model, device)   ###
    val_loss = validate(val_dl, loss, model, device)       
    test_loss = validate(test_dl, loss, model, device)     

    early_stop.step(val_loss)
    print(
        f"{0:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
        f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}" ###:: {coeff:>16e}"  ###############
    )

    last_save_epoch = 0
    start = time.time()

    for epoch in range(__n_epochs):
        model.train()
        #num_batches = len(train_dl)
        #tot_coeff = 0

        for example in train_dl:        # for i in range 190
            ###print("check")
            #_, mean_coeff = 
            train_step(example, loss, model, optimiser, device)

            #tot_coeff += mean_coeff
    
            """   
        # --------------------------------------------------------------------------- #
            ...
        # --------------------------------------------------------------------------- #
            #"""

        #avg_train_coeff = tot_coeff / num_batches
        model.eval()
        train_loss = validate(train_dl, loss, model, device) 
        val_loss = validate(val_dl, loss, model, device) 
        test_loss = validate(test_dl, loss, model, device)  

        sched.step(val_loss)
        early_stop.step(val_loss)

        print(
            f"{epoch + 1:>5d} :: {train_loss:>16e} :: {val_loss:>16e} :: " \
            f"{test_loss:>16e} :: {early_stop.best_val_loss:>16e}" ### :: {coeff:>16f}"   ###############
        )

        if early_stop.best_model:
            torch.save(model.state_dict(), model_save_dir / "state_dict.pth")

            if epoch > last_save_epoch + sys_args.model_log_rate:                   # not pulled from flumen
                if sweep: 
                    artifact = wandb.Artifact("model_checkpoint", type="model")
                    artifact.add_file(str(model_save_dir / "state_dict.pth"))
                    wandb.log_artifact(artifact)                
                else: 
                    run.log_model(model_save_dir.as_posix(), name=model_name)
                last_save_epoch = epoch

            run.summary["best_train"] = train_loss
            run.summary["best_val"] = val_loss
            run.summary["best_test"] = test_loss
            run.summary["best_epoch"] = epoch + 1
            ###run.summary["coeff_train"] = coeff  ###############

        wandb.log({
            'time': time.time() - start,
            'epoch': epoch + 1,
            'lr': optimiser.param_groups[0]['lr'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            ###'coeff_train': coeff,   ###############
        })

        if early_stop.early_stop:
            print(f"{epoch + 1:>5d} :: --- Early stop ---")
            break

    train_time = time.time() - start

    print(f"Training took {train_time:.2f} seconds.")

    run.log_model(model_save_dir.as_posix(), name=model_name, aliases=["best"])     # not pulled from flumen

    print_system_performance(initial_metrics)       # Execution Performance Summary
    
    """
    # --------------------------------------------------------------------------- #
    ...
    # --------------------------------------------------------------------------- #
    #"""
# --------------------------------------------------------------------------- #


# ------ Run Initialization ------------------------------------------------- #
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
        wandb.agent(sweep_id, train_sweep, count=num_sweeps)

    else: main(SWEEP)
# --------------------------------------------------------------------------- #