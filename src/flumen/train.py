import torch, wandb


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

def train_step(example, loss_fn, model, optimiser, device):
    x0, y, u, deltas = prep_inputs(*example, device)

    optimiser.zero_grad()

    y_pred = model(x0, u, deltas)  
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    print("check1")
    ###torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        ### added for lamda mode
    print("check2")
    optimiser.step()

    return loss.item()



def train_step_(example, loss_fn, model, optimiser, device):
    optimizer_mode = wandb.config['optimiser_mode'] 
    function_name = f"train_step_{optimizer_mode}"
    train_step_function = globals().get(function_name)

    ###if train_step_function is None: raise ValueError(f"Unknown training mode: {optimizer_mode}. Available modes: adam, tbptt, nesterov, newton")

    print("\n\nGradient Propagation mode: ", function_name, "\n\n")   
    return train_step_function(example, loss_fn, model, optimiser, device)


# --------------------------------------------------------------------------- #

def train_step_adam(example, loss_fn, model, optimiser, device):
    """
    DEFAULT!
    --------------------------------------------------------------------------------------
    Standard training step using basic gradient descent optimization.
    Uses Adam (Adaptive Moment Estimation), an adaptive gradient-based optimiser.

    This function:
    1. Prepares the inputs and moves them to the device.
    2. Performs a forward pass to compute predictions.
    3. Computes the loss using the given loss function.
    4. Performs backpropagation to compute gradients.
    5. Updates the model parameters using the optimiser.
    
    Suitable for standard training scenarios without specialized optimization techniques.
    """
    x0, y, u, deltas = prep_inputs(*example, device)

    optimiser.zero_grad()

    y_pred = model(x0, u, deltas)  
    loss = model.state_dim * loss_fn(y, y_pred)

    loss.backward()
    optimiser.step()

    #print("\ttrain_step_adam")
    return loss.item(), y_pred  


def train_step_tbptt(example, loss_fn, model, optimiser, device, tbptt_steps=5):
    """
    DOES NOT WORK!
    same optimiser as default case!
    --------------------------------------------------------------------------------------
    Truncated Backpropagation Through Time (TBPTT) training step.
    Used for training recurrent models by breaking the sequence into smaller chunks.
    
    This function:
    1. Prepares the inputs and initializes hidden states.
    2. Iterates through the sequence in chunks of size `tbptt_steps`.
    3. Updates gradients in a rolling manner (retaining computation graph).
    4. Performs multiple partial backward passes before updating the model.

    Suitable for training RNNs efficiently by reducing memory usage compared to full BPTT.
    """
    x0, y, u, deltas = prep_inputs(*example, device)    ###
    optimiser.zero_grad()                               ###

    loss_total = 0
    u_unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(u, batch_first=True)
    print("\nu_unpacked.shape:", u_unpacked.shape)      # output | torch.Size([128, 75, 2])
    print("lengths.shape:", lengths.shape)              # output | torch.Size([128])
    print("deltas.shape:", deltas.shape)                # output | torch.Size([128, 75, 1])
    
    for t in range(0, u_unpacked.shape[1], tbptt_steps):
        u_t = u_unpacked[:, t:t+tbptt_steps, :]
        lengths_t = torch.clamp(lengths - t, min=0, max=tbptt_steps)
        deltas_t = deltas[:, t:t+tbptt_steps, :]
        if u_t.shape[1] == 0:
            Warning(f"Skipping step at t={t} due to zero-length sequence")
            continue

        print("\t--------------------------------\n\tt:", t)
        print("\tu_t.shape:", u_t.shape)                # output | torch.Size([128, 5, 2])
        print("\tlengths_t.shape:", lengths_t.shape)    # output | torch.Size([128])
        print("\tdeltas_t.shape:", deltas_t.shape)      # output | torch.Size([128, 5, 1])
        u_packed = torch.nn.utils.rnn.pack_padded_sequence(u_t, lengths_t, batch_first=True)
        y_pred = model(x0, u_packed, deltas_t)

        loss = model.state_dim * loss_fn(y[:, t:t+tbptt_steps, :], y_pred)
        loss.backward(retain_graph=True)                ###

        optimiser.step()                                ### 
        optimiser.zero_grad()                           ###
        loss_total += loss.item()
    
    loss_item = loss_total / (u_unpacked.shape[1] // tbptt_steps)
    #print("\ttrain_step_tbptt")
    return loss_item, y_pred


def train_step_nesterov(example, loss_fn, model, optimiser, device):
    """
    SEEMS TO WORK FIN£!
    different optimiser then the default case!
    --------------------------------------------------------------------------------------
    Training step using Nesterov Accelerated Gradient (NAG).
    
    This function:
    1. Uses an SGD optimiser with Nesterov momentum.
    2. Performs forward propagation, computes the loss, and backpropagates.
    3. Applies an update step using the optimiser.

    Nesterov momentum looks ahead to the future position before computing gradients,
    which can lead to faster convergence in convex optimization problems.
    """
    x0, y, u, deltas = prep_inputs(*example, device)    ###

    ###optimiser_new = optim.SGD(model.parameters(), lr=wandb.config['lr'], momentum=0.9, nesterov=True)
    optimiser.zero_grad()                               ###

    y_pred = model(x0, u, deltas)  ###
    loss = model.state_dim * loss_fn(y, y_pred)         ###

    loss.backward()                                     ###
    optimiser.step()                                    ###     

    #print("\ttrain_step_nesterov")
    return loss.item(), y_pred                          ###


def train_step_newton(example, loss_fn, model, optimiser, device):
    """
    WORKS BUT TOO SLOW!
    different optimiser then the default case!
    --------------------------------------------------------------------------------------
    Training step using the Newton method via L-BFGS optimiser.
    
    This function:
    1. Uses L-BFGS, a quasi-Newton optimization method that approximates the Hessian.
    2. Defines a closure function to recompute the forward and backward pass for optimization.
    3. Calls the optimiser step, which iteratively refines the parameter update.

    L-BFGS is useful for small-scale problems and second-order optimization,
    but can be computationally expensive in high-dimensional settings.
    """
    x0, y, u, deltas = prep_inputs(*example, device)        ###
    ###optimiser_new = lbfgs.LBFGS(model.parameters())

    def closure():
        optimiser.zero_grad()                               ###
        y_pred = model(x0, u, deltas)                       ###
        loss = model.state_dim * loss_fn(y, y_pred)         ###
        loss.backward()                                     ###
        return loss, y_pred
        
    loss, y_pred = optimiser.step(closure)
    #print("\ttrain_step_newton")
    return loss.item(), y_pred                              ###

# --------------------------------------------------------------------------- #


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
            print("----EarlyStopping counter: ", self.counter, " - must be <", self.patience)   # --- ADDED!
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
