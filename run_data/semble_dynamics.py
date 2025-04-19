import numpy as np
from . import initial_state
from scipy.special import expit


class Dynamics:

    def __init__(self, state_dim, control_dim, mask=None):
        self.n = state_dim
        self.m = control_dim

        self.mask = mask if mask is not None else self.n * (1, )
        self.p = sum(self.mask)

        self._method = "RK45"

    def __call__(self, x, u):
        return self._dx(x, u)

    def default_initial_state(self) -> initial_state.InitialStateGenerator:
        """Returns an instance of the default initial state sampler."""
        return initial_state.GaussianInitialState(self.n)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def default_method(self) -> str:
        return self._method

    def dims(self):
        return (self.n, self.m, self.p)


class LinearSys(Dynamics):

    def __init__(self, a, b):
        a = np.array(a)
        b = np.array(b)

        super().__init__(a.shape[0], b.shape[1])

        self.a = a
        self.b = b
        print(a)

    def _dx(self, x, u):
        return self.a @ x + self.b @ u


class VanDerPol(Dynamics):

    def __init__(self, damping):
        super().__init__(2, 1)

        self.damping = damping
        print("vdp mhu:", self.damping)

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -p + self.damping * (1 - p**2) * v + u.item()

        return (dp, dv)


class FitzHughNagumo(Dynamics):

    def __init__(self, tau, a, b):
        super().__init__(2, 1)
        self._method = "BDF"

        self.tau = tau
        self.a = a
        self.b = b
        self.fact = 50  #1
        print("fhn fact:", self.fact)

    def _dx(self, x, u):
        v, w = x

        dv = self.fact * (v - v**3 - w + u.item())
        dw = (v - self.a - self.b * w) / self.tau

        return (dv, dw)


class Pendulum(Dynamics):

    def __init__(self, damping, freq=2 * np.pi):
        super().__init__(2, 1)
        self.damping = damping
        self.freq2 = freq**2

    def _dx(self, x, u):
        p, v = x

        dp = v
        dv = -self.freq2 * np.sin(p) - self.damping * v + u.item()

        return (dp, dv)


class HodgkinHuxleyFS(Dynamics):

    def __init__(self):
        super().__init__(4, 1, (1, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 0.5
        self.v_k = -90.
        self.v_na = 50.
        self.v_l = -70.
        self.v_t = -56.2
        self.g_k = 10.
        self.g_na = 56.
        self.g_l = 1.5e-2

    def default_initial_state(self):
        return initial_state.HHFSInitialState()

    def _dx(self, x, u):
        v, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - self.g_k * n**4 *
              (v - self.v_k) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx for dx in (dv, dn, dm, dh))


class HodgkinHuxleyRSA(Dynamics):

    def __init__(self):
        super().__init__(5, 1, (1, 0, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 1.0
        self.v_k = -90.
        self.v_na = 56.
        self.v_l = -70.3
        self.v_t = -56.2
        self.g_k = 6.
        self.g_m = 0.075
        self.g_na = 56.
        self.g_l = 2.05e-2
        self.t_max = 608.

    def default_initial_state(self):
        return initial_state.HHRSAInitialState()

    def _dx(self, x, u):
        v, p, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - (self.g_k * n**4 + self.g_m * p) *
              (v - self.v_k) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        t_p = self.t_max / (3.3 * np.exp(
            (v + 35.) / 20.) + np.exp(-(v + 35.) / 20.))

        dp = (1. / (1 + np.exp(-(v + 35) / 10.)) - p) / t_p

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx for dx in (dv, dp, dn, dm, dh))


class HodgkinHuxleyIB(Dynamics):

    def __init__(self):
        super().__init__(7, 1, (1, 0, 0, 0, 0, 0, 0))
        self._method = "BDF"

        self.time_scale = 100.
        self.v_scale = 100.

        # Parameters follow
        #   A. G. Giannari and A. Astolfi, ‘Model design for networks of
        #   heterogeneous Hodgkin–Huxley neurons’,
        #   Neurocomputing, vol. 496, pp. 147–157, Jul. 2022,
        #   doi: 10.1016/j.neucom.2022.04.115.
        self.c_m = 1.0
        self.v_k = -90.
        self.v_ca = 120.
        self.v_na = 56.
        self.v_l = -70
        self.v_t = -56.2
        self.g_k = 5.
        self.g_m = 0.03
        self.g_ca = 0.2
        self.g_na = 50.
        self.g_l = 0.01
        self.t_max = 608.

    def default_initial_state(self):
        return initial_state.HHIBInitialState()

    def _dx(self, x, u):
        v, p, q, s, n, m, h = x

        # denormalise first state variable
        v *= self.v_scale

        dv = (u.item() - (self.g_k * n**4 + self.g_m * p) *
              (v - self.v_k) - self.g_ca * q**2 * s *
              (v - self.v_ca) - self.g_na * m**3 * h *
              (v - self.v_na) - self.g_l * (v - self.v_l)) / (100. * self.c_m)

        t_p = self.t_max / (3.3 * np.exp(
            (v + 35.) / 20.) + np.exp(-(v + 35.) / 20.))

        dp = (1. / (1 + np.exp(-(v + 35) / 10.)) - p) / t_p

        a_q = 0.055 * (-27. - v) / (np.exp((-27. - v) / 3.8) - 1.)
        b_q = 0.94 * np.exp((-75. - v) / 17.)
        dq = a_q * (1. - q) - b_q * q

        a_s = 0.000457 * np.exp((-13. - v) / 50.)
        b_s = 0.0065 / (np.exp((-15. - v) / 28.) + 1.)
        ds = a_s * (1. - s) - b_s * s

        a_n = -0.032 * (v - self.v_t -
                        15.) / (np.exp(-(v - self.v_t - 15.) / 5.) - 1)
        b_n = 0.5 * np.exp(-(v - self.v_t - 10.) / 40.)
        dn = a_n * (1. - n) - b_n * n

        a_m = -0.32 * (v - self.v_t -
                       13.) / (np.exp(-(v - self.v_t - 13.) / 4.) - 1)
        b_m = 0.28 * (v - self.v_t - 40.) / (np.exp(
            (v - self.v_t - 40.) / 5.) - 1)
        dm = a_m * (1. - m) - b_m * m

        a_h = 0.128 * np.exp(-(v - self.v_t - 17.) / 18.)
        b_h = 4. / (1. + np.exp(-(v - self.v_t - 40.) / 5.))
        dh = a_h * (1. - h) - b_h * h

        return tuple(self.time_scale * dx
                     for dx in (dv, dp, dq, ds, dn, dm, dh))


class HodgkinHuxleyFFE(Dynamics):
    """
    Two RSA neurons coupled in feedforward with an electrical synapse.
    """

    def __init__(self):
        super().__init__(10, 1, (1, 0, 0, 0, 0, 1, 0, 0, 0, 0))
        self._method = "BDF"

        self.rsa = HodgkinHuxleyRSA()
        self.v_scale = self.rsa.v_scale
        self.time_scale = self.rsa.time_scale
        self.eps = 0.1

    def default_initial_state(self):
        return initial_state.HHFFEInitialState()

    def _dx(self, x, u):
        x_in = x[:5]
        x_out = x[5:]
        delta = self.v_scale * (x_in[0] - x_out[0])

        dx_in = self.rsa._dx(x_in, u)
        dx_out = self.rsa._dx(x_out, self.eps * delta)

        return (*dx_in, *dx_out)


class HodgkinHuxleyFBE(Dynamics):
    """
    Two RSA neurons coupled with an electrical synapse in feedfoward and
    a chemical synapse in feedback.
    """

    def __init__(self):
        super().__init__(11, 1, (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0))
        self._method = "BDF"

        self.rsa = HodgkinHuxleyRSA()
        self.v_scale = self.rsa.v_scale
        self.time_scale = self.rsa.time_scale

        self.eps_el = 0.1
        self.eps_ch = 0.5

        self.v_syn = 20.
        self.tau_r = 0.5
        self.tau_d = 8.
        self.v0 = -20.

    def default_initial_state(self):
        return initial_state.HHFBEInitialState()

    def _dx(self, x, u):
        x_in = x[:5]
        x_out = x[5:-1]
        r = x[-1]

        v_in = self.v_scale * x_in[0]
        v_out = self.v_scale * x_out[0]

        delta_el = v_in - v_out
        delta_ch = self.v_syn - v_out

        dx_in = self.rsa._dx(x_in, u + r * self.eps_ch * delta_ch)
        dx_out = self.rsa._dx(x_out, self.eps_el * delta_el)

        dr = (1 / self.tau_r - 1 / self.tau_d) * (1. - r) / (
            1. + np.exp(-v_out + self.v0)) - r / self.tau_d

        return (*dx_in, *dx_out, self.time_scale * dr)


class GreenshieldsTraffic(Dynamics):

    def __init__(self, n, v0, dx=None):
        super().__init__(n, 1)

        self.inv_step = self.n if not dx else 1. / dx
        self.v0 = v0

    def flux(self, x):
        return self.v0 * x * (1. - x)

    def _dx(self, x, u):
        q_out = self.flux(x)
        q0_in = self.flux(u.item())

        q_in = np.roll(q_out, 1)
        q_in[0] = q0_in

        dx = self.inv_step * (q_in - q_out)

        return dx


class TwoTank(Dynamics):
    ''' Two tank dynamics with overflow.
        Source: https://apmonitor.com/do/index.php/Main/LevelControl
    '''

    def __init__(self):
        super().__init__(2, 2)
        self._method = "BDF"

        self.c1 = 0.08  # inlet valve coefficient
        self.c2 = 0.04  # outlet valve coefficient

    def default_initial_state(self):
        return initial_state.UniformInitialState(self.n)

    def _dx(self, x, u):
        h1, h2 = x

        pump = u[0]
        valve = u[1]

        dh1 = self.c1 * (1.0 - valve) * pump - self.c2 * np.sqrt(np.abs(h1))
        dh2 = self.c1 * valve * pump + self.c2 * \
            np.sqrt(np.abs(h1)) - self.c2 * np.sqrt(np.abs(h2))

        if (h1 >= 1. and dh1 > 0.) or (h1 <= 1e-10 and dh1 < 0.):
            dh1 = 0.

        if (h2 >= 1. and dh2 > 0.) or (h2 <= 1e-10 and dh2 < 0.):
            dh2 = 0.

        return (dh1, dh2)


class NonlinearActivationDynamics(Dynamics):
    def __init__(self, a_s, a_m, a_b, b_b, b, state_dim=5, control_dim=1, activation="sigmoid", mode="stable"):
        super().__init__(state_dim, control_dim)

        self.activation = activation
        self.mode = mode

        if mode=="stable": 
            self.A = a_s
            self.B = b
        elif mode=="big": 
            self.A = a_b
            self.B = b_b
        else: 
            self.A = a_m #if mode == "stable" else a_m
            self.B = b

        print(f"NonlinearActivationDynamics initialized with activation: {activation}\nMode: {mode}\nSize: {state_dim}")
        print("Matrix A:", self.A)
        print("Matrix B:", self.B)

    def _sigma(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return expit(x)  # sigmoid
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    

    def _dx(self, x, u):
        return -x + self._sigma(self.A @ x + self.B @ u)



_dynamics_names = {
    "LinearSys": LinearSys,
    "VanDerPol": VanDerPol,
    "FitzHughNagumo": FitzHughNagumo,
    "Pendulum": Pendulum,
    "HodgkinHuxleyFS": HodgkinHuxleyFS,
    "HodgkinHuxleyRSA": HodgkinHuxleyRSA,
    "HodgkinHuxleyIB": HodgkinHuxleyIB,
    "HodgkinHuxleyFFE": HodgkinHuxleyFFE,
    "HodgkinHuxleyFBE": HodgkinHuxleyFBE,
    "GreenshieldsTraffic": GreenshieldsTraffic,
    "TwoTank": TwoTank,
    "NonlinearActivationDynamics": NonlinearActivationDynamics,
}


def get_dynamics(name, args):
    return _dynamics_names[name](**args)
