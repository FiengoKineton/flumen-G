import torch
from torch.utils.data import DataLoader

from . import CausalFlowModel
from .train import EarlyStopping

from .experiment import Experiment, instantiate_model

import numpy as np
from scipy.linalg import sqrtm, inv


# ------------------------------------------------- #
#   Normalizza i target dei dati d'input.           #
#   Calcola la media e la deviazione standard,      #
#   quindi applica la trasformazione di whitening.  #
# ------------------------------------------------- #

def whiten_targets(data):
    mean = data[0].state.mean(axis=0)
    std = sqrtm(np.cov(data[0].state.T))
    istd = inv(std)

    for d in data:
        d.state[:] = ((d.state - mean) @ istd).type(torch.get_default_dtype())
        d.init_state[:] = ((d.init_state - mean) @ istd).type(
            torch.get_default_dtype())

    return mean, std, istd


# ------------------------------------------------- #
#   Prepara un esperimento per l'addestramento.     #
#   Divide i dati in train, validation e test,      #
#   applica il whitening se richiesto e            #
#   inizializza il modello e gli ottimizzatori.     #
# ------------------------------------------------- #

def prepare_experiment(data, args):
    train_data, val_data, test_data = data.get_datasets(
        args.max_seq_len, args.samples_per_state)

    if args.whiten_data:
        train_mean, train_std, train_istd = whiten_targets(
            (train_data, val_data, test_data))
    else:
        train_mean = 0.
        train_std = np.eye(train_data.state_dim)
        train_istd = np.eye(train_data.state_dim)

    experiment = Experiment(args,
                            data.dims(), (train_mean, train_std, train_istd),
                            save_root=args.write_dir)

    model: CausalFlowModel = instantiate_model(args, train_data.state_dim,
                                               train_data.control_dim,
                                               train_data.output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Sposta il modello sulla GPU se disponibile

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.sched_patience,
        cooldown=args.sched_cooldown,
        factor=1. / args.sched_factor,      # ---ADDED!
        threshold=args.es_delta,
        verbose=True)

    mse_loss = torch.nn.MSELoss().to(device)

    early_stop = EarlyStopping(es_patience=args.es_patience,
                               es_delta=args.es_delta)

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    return experiment, (model, mse_loss, optimizer, sched, early_stop,
                        train_dl, val_dl, test_dl, device, args.n_epochs)
