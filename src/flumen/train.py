import torch
import torch.optim as optim
import torch.optim.lbfgs as lbfgs


# ------------------------------------------------- #
#   Prepara gli input per il modello.               #
#   Riordina i dati per lunghezza decrescente,      #
#   impacchetta le sequenze per l'RNN e le invia    #
#   al dispositivo corretto.                        #
# ------------------------------------------------- #

def prep_inputs(x0, y, u, lengths, device):
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

    x0 = x0.to(device)
    y = y.to(device)
    u = u.to(device)
    deltas = deltas.to(device)

    return x0, y, u, deltas



# ------------------------------------------------- #
#   Valida il modello sul dataset di validazione.   #
#   Calcola la loss media sui dati di validazione   #
#   senza aggiornare i pesi del modello.            #
# ------------------------------------------------- #

def validate(data, loss_fn, model, device):
    vl = 0.

    with torch.no_grad():
        for example in data:
            x0, y, u, deltas = prep_inputs(*example, device)

            y_pred = model(x0, u, deltas)
            vl += loss_fn(y, y_pred).item()

    return model.state_dim * vl / len(data)



# ------------------------------------------------- #
#   Esegue un singolo passo di addestramento.       #
#   Calcola la predizione, la loss, esegue il       #
#   backpropagation e aggiorna i pesi del modello.  #
# ------------------------------------------------- #

# Enable-anomaly-detection --------------------------------------- #
torch.autograd.set_detect_anomaly(True)     # --- ADDED for LSTM_my!
# ---------------------------------------------------------------- #


def train_step(example, loss_fn, model, optimizer, device):
    mode="tbptt"
    function_name = f"train_step_{mode}"
    train_step_function = globals().get(function_name)

    if train_step_function is None:
        raise ValueError(f"Unknown training mode: {mode}. Available modes: default, tbptt, nesterov, newton, adam")

    ###print("\n\nGradient Propagation mode: ", function_name, "\n\n")   # --- ADDED!
    return train_step_function(example, loss_fn, model, optimizer, device)




def train_step_default(example, loss_fn, model, optimizer, device):
    """
    Standard training step using basic gradient descent optimization.
    This function:
    1. Prepares the inputs and moves them to the device.
    2. Performs a forward pass to compute predictions.
    3. Computes the loss using the given loss function.
    4. Performs backpropagation to compute gradients.
    5. Updates the model parameters using the optimizer.
    
    Suitable for standard training scenarios without specialized optimization techniques.
    """
    x0, y, u, deltas = prep_inputs(*example, device)

    optimizer.zero_grad()

    y_pred = model(x0, u, deltas)
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_step_tbptt(example, loss_fn, model, optimizer, device, tbptt_steps=5):
    """
    Truncated Backpropagation Through Time (TBPTT) training step.
    Used for training recurrent models by breaking the sequence into smaller chunks.
    
    This function:
    1. Prepares the inputs and initializes hidden states.
    2. Iterates through the sequence in chunks of size `tbptt_steps`.
    3. Updates gradients in a rolling manner (retaining computation graph).
    4. Performs multiple partial backward passes before updating the model.

    Suitable for training RNNs efficiently by reducing memory usage compared to full BPTT.
    """
    x0, y, u, deltas = prep_inputs(*example, device)
    optimizer.zero_grad()
    h, c = None, None
    loss_total = 0
    u_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(u, batch_first=True)
    
    for t in range(0, u_unpacked.shape[1], tbptt_steps):
        u_t = u_unpacked[:, t:t+tbptt_steps, :]
        if u_t.shape[1] == 0:
            continue
        u_packed = torch.nn.utils.rnn.pack_padded_sequence(u_t, lengths, batch_first=True)
        y_pred, (h, c) = model(x0, u_packed, deltas, h, c)
        loss = model.state_dim * loss_fn(y[:, t:t+tbptt_steps, :], y_pred)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        loss_total += loss.item()
    return loss_total / (u_unpacked.shape[1] // tbptt_steps)


def train_step_nesterov(example, loss_fn, optimizer, model, device):
    """
    Training step using Nesterov Accelerated Gradient (NAG).
    
    This function:
    1. Uses an SGD optimizer with Nesterov momentum.
    2. Performs forward propagation, computes the loss, and backpropagates.
    3. Applies an update step using the optimizer.

    Nesterov momentum looks ahead to the future position before computing gradients,
    which can lead to faster convergence in convex optimization problems.
    """
    x0, y, u, deltas = prep_inputs(*example, device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    optimizer.zero_grad()
    y_pred = model(x0, u, deltas)
    loss = model.state_dim * loss_fn(y, y_pred)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_step_newton(example, loss_fn, optimizer, model, device):
    """
    Training step using the Newton method via L-BFGS optimizer.
    
    This function:
    1. Uses L-BFGS, a quasi-Newton optimization method that approximates the Hessian.
    2. Defines a closure function to recompute the forward and backward pass for optimization.
    3. Calls the optimizer step, which iteratively refines the parameter update.

    L-BFGS is useful for small-scale problems and second-order optimization,
    but can be computationally expensive in high-dimensional settings.
    """
    x0, y, u, deltas = prep_inputs(*example, device)
    optimizer = lbfgs.LBFGS(model.parameters())
    def closure():
        optimizer.zero_grad()
        y_pred = model(x0, u, deltas)
        loss = model.state_dim * loss_fn(y, y_pred)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    return loss.item()


def train_step_adam(example, loss_fn, optimizer, model, device):
    """
    Training step using the Adam optimizer.
    
    This function:
    1. Uses Adam (Adaptive Moment Estimation), an adaptive gradient-based optimizer.
    2. Computes gradients and updates parameters with an adaptive learning rate.
    3. Balances first-order momentum (similar to momentum-based SGD) and second-order moment estimation.

    Adam is widely used due to its adaptive learning rates, making it effective in many deep learning tasks.
    """
    x0, y, u, deltas = prep_inputs(*example, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    y_pred = model(x0, u, deltas)
    loss = model.state_dim * loss_fn(y, y_pred)
    loss.backward()
    optimizer.step()
    return loss.item()



# ------------------------------------------------- #
#   Implementa l'early stopping.                    #
#   Ferma l'addestramento se la loss di validazione #
#   non migliora per un numero definito di epoche.  #
# ------------------------------------------------- #

class EarlyStopping:

    def __init__(self, es_patience, es_delta=0.):
        self.patience = es_patience
        self.delta = es_delta

        self.best_val_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model = False

    def step(self, val_loss):
        self.best_model = False

        if self.best_val_loss - val_loss > self.delta:
            self.best_val_loss = val_loss
            self.best_model = True
            self.counter = 0
        else:
            print("\tEarlyStopping counter: ", self.counter, " - must be <", self.patience, "\n")   # --- ADDED!
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
