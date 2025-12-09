# guidance/drift.py
import jax
import jax.numpy as np
from coefficients import Coefficient

def eps_fn(t: float) -> float:
    # global / shared guidance schedule
    return 1.0

def make_phi_fn(net, params, class_label: int):
    def phi_single(x, s, t):
        return net.apply(
            params,
            s,
            t,
            x,
            class_label,
            train=False,
            method=net.calc_phi,
        )
    return phi_single

def X_t1_single(phi_single, x, t):
    v_t1 = phi_single(x, t, 1.0)
    return x + (1.0 - t) * v_t1

def grad_r_of_Xt1_wrt_x_single(phi_single, dap_reward, x, t, class_label):
    def f(x):
        return X_t1_single(phi_single, x, t)
    y = f(x)
    g_y = dap_reward.avg_grad(y, label=class_label)
    _, pullback = jax.vjp(f, x)
    grad_x, = pullback(g_y)
    return grad_x

def guided_drift_single(phi_single, dap_reward, x, t, class_label):
    v_tt = phi_single(x, t, t)
    denom = 1.0 - t + 1e-6
    s_t_val = (t * v_tt - x) / denom
    grad_r_chain = grad_r_of_Xt1_wrt_x_single(phi_single, dap_reward, x, t, class_label)
    eps_t = eps_fn(t)
    return v_tt + eps_t * (s_t_val + t * grad_r_chain)

class GuidedLinearDrift(Coefficient):
    def __init__(self, net, params, dap_reward, class_label: int):
        super().__init__()
        self.class_label = int(class_label)
        self.dap_reward = dap_reward
        self.phi_single = make_phi_fn(net, params, self.class_label)

    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        single = lambda x: guided_drift_single(self.phi_single, self.dap_reward, x, t, self.class_label)
        return jax.vmap(single)(X)
