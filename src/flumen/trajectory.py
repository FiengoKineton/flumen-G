import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------- #
#   Dataset per memorizzare traiettorie raw.        #
#   Contiene stati iniziali, sequenze di controllo  #
#   e variabili temporali associate.                #
# ------------------------------------------------- #

class RawTrajectoryDataset(Dataset):

    def __init__(self,
                 data,
                 state_dim,
                 control_dim,
                 output_dim,
                 delta,
                 output_mask,
                 noise_std=0.,
                 **kwargs):
        self.__dict__.update(kwargs)

        n_traj = len(data)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.output_dim = output_dim
        self.delta = delta
        self.mask = output_mask

        self.init_state = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())
        self.init_state_noise = torch.empty(
            (n_traj, self.state_dim)).type(torch.get_default_dtype())

        self.time = []
        self.state = []
        self.state_noise = []
        self.control_seq = []

        for k, sample in enumerate(data):
            self.init_state[k] = torch.from_numpy(sample["init_state"].reshape(
                (1, self.state_dim)))
            self.init_state_noise[k] = 0.
            self.time.append(
                torch.from_numpy(sample["time"]).type(
                    torch.get_default_dtype()).reshape((-1, 1)))

            self.state.append(
                torch.from_numpy(sample["state"]).type(
                    torch.get_default_dtype()).reshape((-1, self.state_dim)))

            self.state_noise.append(
                torch.normal(mean=0.,
                             std=noise_std,
                             size=self.state[-1].size()))

            self.control_seq.append(
                torch.from_numpy(sample["control"]).type(
                    torch.get_default_dtype()).reshape((-1, self.control_dim)))


    
    # ------------------------------------------------- #
    #   Genera un dataset a partire da un generatore.   #
    #   Crea traiettorie casuali basate sui parametri   #
    #   specificati.                                    #
    # ------------------------------------------------- #

    @classmethod
    def generate(cls, generator, time_horizon, n_trajectories, n_samples,
                 noise_std):

        def get_example():
            x0, t, y, u = generator.get_example(time_horizon, n_samples)
            return {
                "init_state": x0,
                "time": t,
                "state": y,
                "control": u,
            }

        data = [get_example() for _ in range(n_trajectories)]

        return cls(data,
                   *generator.dims(),
                   delta=generator._delta,
                   output_mask=generator._dyn.mask,
                   generator=generator,
                   noise_std=noise_std)

    def __len__(self):
        return len(self.init_state)

    def __getitem__(self, index):
        return (self.init_state[index], self.init_state_noise[index],
                self.time[index], self.state[index], self.state_noise[index],
                self.control_seq[index])



# ------------------------------------------------- #
#   Dataset per la preparazione delle sequenze RNN. #
#   Converte le traiettorie raw in input            #
#   strutturati per modelli basati su RNN.          #
# ------------------------------------------------- #

class TrajectoryDataset(Dataset):

    def __init__(self,
                 raw_data: RawTrajectoryDataset,
                 max_seq_len=-1,
                 n_samples=1):

        self.state_dim = raw_data.state_dim
        self.control_dim = raw_data.control_dim
        self.output_dim = raw_data.output_dim
        self.delta = raw_data.delta

        mask = tuple(bool(v) for v in raw_data.mask)

        init_state = []
        state = []
        rnn_input_data = []
        seq_len_data = []

        rng = np.random.default_rng()

        k_tr = 0        # --- what does it do?

        for (x0, x0_n, t, y, y_n, u) in raw_data:
            y += y_n
            x0 += x0_n

            if max_seq_len == -1:
                for k_s, y_s in enumerate(y):
                    rnn_input, rnn_input_len = self.process_example(
                        0, k_s, t, u, self.delta)

                    s = y_s.view(1, -1)[:, mask].reshape(-1)

                    init_state.append(x0)
                    state.append(s)
                    seq_len_data.append(rnn_input_len)
                    rnn_input_data.append(rnn_input)

            else:
                for k_s, y_s in enumerate(y):
                    # find index of last relevant state sample
                    times = (t - t[k_s] - max_seq_len * self.delta)
                    times[times > 0] = 0.
                    k_l = times.argmax().item()

                    if k_l == k_s:
                        end_idxs = (0, )
                    else:
                        end_idxs = rng.choice(k_l - k_s,
                                              size=min(n_samples, k_l - k_s),
                                              replace=False)

                    for k_e in end_idxs:
                        rnn_input, rnn_input_len = self.process_example(
                            k_s, k_s + k_e, t, u, self.delta)

                        init_state.append(y_s)
                        state.append(y[k_s + k_e, mask])
                        seq_len_data.append(rnn_input_len)
                        rnn_input_data.append(rnn_input)

        self.init_state = torch.stack(init_state).type(
            torch.get_default_dtype())
        self.state = torch.stack(state).type(torch.get_default_dtype())
        self.rnn_input = torch.stack(rnn_input_data).type(
            torch.get_default_dtype())
        self.seq_lens = torch.tensor(seq_len_data, dtype=torch.long)

        self.len = len(init_state)


    # ------------------------------------------------- #
    #   Prepara un esempio per l'input RNN.             #
    #   Determina gli indici e costruisce la sequenza.  #
    # ------------------------------------------------- #  

    @staticmethod
    def process_example(start_idx, end_idx, t, u, delta):
        init_time = 0.

        u_start_idx = int(np.floor((t[start_idx] - init_time) / delta))
        u_end_idx = int(np.floor((t[end_idx] - init_time) / delta))
        u_sz = 1 + u_end_idx - u_start_idx
        # print("\nu_sz: ", u_sz)         # output | 75

        u_seq = torch.zeros_like(u)
        u_seq[0:u_sz] = u[u_start_idx:(u_end_idx + 1)]  # extract a segment from u and put it at the beginning of u_seq, while padding the rest of u_seq with zeros

        deltas = torch.ones((u_seq.shape[0], 1))
        t_u_end = init_time + delta * u_end_idx
        t_u_start = init_time + delta * u_start_idx

        if u_sz > 1:
            deltas[0] = (1. - (t[start_idx] - t_u_start) / delta).item()
            deltas[u_sz - 1] = ((t[end_idx] - t_u_end) / delta).item()
        else:
            deltas[0] = ((t[end_idx] - t[start_idx]) / delta).item()

        deltas[u_sz:] = 0.

        # print("\nu_seq shape: ", u_seq.shape)         # , output | torch.Size([76, 1])
        # print("\ndeltas shape: ", deltas.shape)       # , output | torch.Size([76, 1])
        rnn_input = torch.hstack((u_seq, deltas))
        
        return rnn_input, u_sz

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.init_state[index], self.state[index],
                self.rnn_input[index], self.seq_lens[index])
