import torch

import sys, os, subprocess
from uuid import uuid4
from time import time
from datetime import datetime
from inspect import cleandoc
from shlex import quote
from copy import deepcopy

from .model___old import CausalFlowModel


def timestamp_str(timestamp):
    return datetime.fromtimestamp(timestamp).strftime(
        '%Y%m%d_%H%M%S') if timestamp else "N/A"


def save_path(root, dir, timestamp, train_id):
    file_name = dir + '_' + timestamp_str(timestamp) + '_' + str(train_id.hex)
    path = os.path.join(root, dir)

    return path, file_name


def instantiate_model(args, state_dim, control_dim, output_dim):
    return CausalFlowModel(state_dim=state_dim,
                           control_dim=control_dim,
                           output_dim=output_dim,
                           control_rnn_size=args.control_rnn_size,
                           control_rnn_depth=args.control_rnn_depth,
                           encoder_size=args.encoder_size,
                           encoder_depth=args.encoder_depth,
                           decoder_size=args.decoder_size,
                           decoder_depth=args.decoder_depth,
                           use_batch_norm=args.use_batch_norm)


class Experiment:

    def __init__(self, args, dims, train_stats, save_root):
        self.model = None

        self.train_id = uuid4()

        self.command_line = sys.argv.copy()
        self.args = args

        self.td_mean, self.td_std, self.td_std_inv = train_stats

        self.dims = dims

        self.save_timestamp = None

        try:
            self.git_head = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            self.git_status = 'clean' if (subprocess.call(
                ['git', 'diff-index', '--quiet', 'HEAD']) == 0) else 'dirty'
        except:
            self.git_head = None

        if args.experiment_id:
            self.creation_timestamp = time()
            self.save_path, self.file_name = save_path(save_root,
                                                       args.experiment_id,
                                                       self.creation_timestamp,
                                                       self.train_id)
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = None
            self.file_name = None

        self.n_epochs = 0
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.train_loss_best = None
        self.val_loss_best = None
        self.test_loss_best = None

        self.train_time = 0

        self.model_state = None

    def register_progress(self, train, val, test, best):
        self.n_epochs += 1
        self.train_loss.append(train)
        self.val_loss.append(val)
        self.test_loss.append(test)

        if best:
            self.train_loss_best = train
            self.val_loss_best = val
            self.test_loss_best = test

    def save_model(self, model):
        self.model_state = deepcopy(model.state_dict())

    def load_model(self, device='cpu'):
        model = instantiate_model(self.args, *self.dims)
        model.load_state_dict(self.model_state)

        return model

    def save(self, train_time):
        self.train_time = train_time

        if self.save_path:
            self.save_timestamp = time()
            torch.save(self,
                       os.path.join(self.save_path, self.file_name + '.pth'))

    def predict(self, model, x0, u, deltas):
        x0[:] = (x0 - self.td_mean) @ self.td_std_inv
        with torch.no_grad():
            y_pred = model(x0, u, deltas).numpy()
        # y_pred[:] = self.td_mean + y_pred @ self.td_std

        return y_pred

    def args_str(self):
        out_str = ""

        for k, v in vars(self.args).items():
            out_str += f"{k}: {v}\n"

        return out_str

    def __str__(self):
        return cleandoc(f'''\
            --- Trained model   {self.file_name}
                Timestamp:      {timestamp_str(self.save_timestamp)}
                Git hash:       {self.git_head + ' (' + self.git_status + ')' if self.git_head else 'N/A'}
                Command line:   {' '.join((quote(arg) for arg in self.command_line))}
                Train time:     {self.train_time:.2f}
                Loss:           tr={self.train_loss_best:.3e} // vl={self.val_loss_best:.3e} // ts={self.test_loss_best:.3e}
        ''')
