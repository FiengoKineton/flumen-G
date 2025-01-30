from scipy.integrate import solve_ivp
import numpy as np

from .dynamics import Dynamics
from .sequence_generators import SequenceGenerator
from .initial_state import InitialStateGenerator


class TrajectorySampler:

    def __init__(self,
                 dynamics: Dynamics,
                 control_delta,
                 control_generator: SequenceGenerator,
                 method=None,
                 initial_state_generator: InitialStateGenerator = None):
        self._n = dynamics.n
        self._ode_method = dynamics.default_method if not method else method
        self._dyn = dynamics
        self._delta = control_delta  # control sampling time
        self._seq_gen = control_generator

        self.state_generator = (initial_state_generator
                                if initial_state_generator else
                                dynamics.default_initial_state())

        self._rng = np.random.default_rng()

        self._init_time = 0.

    def dims(self):
        return self._dyn.dims()

    def reset_rngs(self):
        self._seq_gen._rng = np.random.default_rng()
        self.state_generator.rng = np.random.default_rng()
        self._rng = np.random.default_rng()

    def get_example(self, time_horizon, n_samples):
        y0 = self.state_generator.sample()

        control_seq = self._seq_gen.sample(time_range=(self._init_time,
                                                       time_horizon),
                                           delta=self._delta)

        def f(t, y):
            n_control = int(np.floor((t - self._init_time) / self._delta))
            u = control_seq[n_control]  # get u(t)

            return self._dyn(y, u)

        t_samples = self._init_time + (time_horizon - self._init_time) * lhs(
            n_samples, self._rng)
        t_samples = np.sort(np.append(t_samples, [self._init_time]))

        traj = solve_ivp(
            f,
            (self._init_time, time_horizon),
            y0,
            t_eval=t_samples,
            method=self._ode_method,
        )

        y = traj.y.T
        t = traj.t.reshape(-1, 1)

        return y0, t, y, control_seq


def lhs(n_samples, rng):
    '''Performs Latin Hypercube sampling on the unit interval.'''
    bins_start_val = np.linspace(0., 1., n_samples + 1)[:-1]
    samples = rng.uniform(
        size=(n_samples, )) / n_samples  # sample a delta for each bin
    return bins_start_val + samples
