import torch
import torch.jit
import math

from norse.torch.module.snn import SNNCell
from typing import NamedTuple, Tuple
from norse.torch.functional.threshold import threshold
try:
    import norse_op
except ModuleNotFoundError:  # pragma: no cover
    pass

class DWMTJParameters(NamedTuple):
    
    """Parametrization of a DWMTJ neuron

    Parameters:
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-10)
    w1: torch.Tensor = torch.as_tensor(25e-9)
    w2: torch.Tensor = torch.as_tensor(25e-9)
    d: torch.Tensor = torch.as_tensor(1.5e-9)
    P: torch.Tensor = torch.as_tensor(0.7)
    Ms: torch.Tensor = torch.as_tensor(8e5)
    a: torch.Tensor = torch.as_tensor(0.05)
    H: torch.Tensor = torch.as_tensor(0.0)
    I: torch.Tensor = torch.as_tensor(80e-6)
    x_th: torch.Tensor = torch.as_tensor(200e-9)
    x_reset: torch.Tensor = torch.as_tensor(0.0e-9)
    method: str = "super"
    alpha: torch.Tensor = torch.as_tensor(150.0)

class DWMTJLIFState(NamedTuple):
    """State of a feed forward DWMTJ

    Parameters:
        z (torch.Tensor): recurrent spikes
        x (torch.Tensor): DW position
        i (torch.Tensor): synaptic input current
    """

    z: torch.Tensor
    x: torch.Tensor
    i: torch.Tensor

class DWMTJRecurrentCell(SNNCell):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        p: DWMTJParameters = DWMTJParameters(),
        **kwargs
    ):
        super().__init__(
            activation=_lif_step_jit,
            activation_sparse=lif_step_sparse,
            state_fallback=self.initial_state,
            p=p,
            input_size=input_size,
            hidden_size=hidden_size,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> DWMTJLIFState:
        dims = (*input_tensor.shape[:-1], self.hidden_size)
        state = DWMTJLIFState(
            z=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ).to_sparse()
            if input_tensor.is_sparse
            else torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            v=torch.full(
                dims,
                torch.as_tensor(self.p.x_reset).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *dims,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state

class DWMTJParametersJIT(NamedTuple):
    """Parametrization of a DW-MTJ neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (torch.Tensor): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    d: torch.Tensor
    P: torch.Tensor
    Ms: torch.Tensor
    a: torch.Tensor
    H: torch.Tensor
    I: torch.Tensor
    x_th: torch.Tensor
    x_reset: torch.Tensor
    method: str
    alpha: torch.Tensor

@torch.jit.script
def _lif_step_jit(
    input_tensor: torch.Tensor,
    state: DWMTJLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DWMTJParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DWMTJLIFState]:  # pragma: no cover

    # constants
    Keff = 5e5 - (0.5*0.9*4e-7*math.pi*(p.Ms**2)) # effective anisotropy
    Delt = math.sqrt(1.3e-11/Keff) # DW width param (Bloch)
    Hslope = (6.000171568956471e4*((p.w2-p.w1)/p.x_th)**2 + 1.084982104400131e+04*((p.w2-p.w1)/p.x_th)
         + 1.228766907095338e+03)
    mu_Hfield = 4e-7*math.pi*1.7595e11*Delt/p.a  # DW mobility Neel
    mu_Hslope = 4e-7*math.pi*1.7595e11*Delt*p.a  # DW mobility Walker

    # compute voltage updates
    # dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    # v_decayed = state.v + dv
    w = state.x*((p.w2-p.w1)/p.x_th) + p.w1
    dx = dt * (((state.i * p.I)/(w * p.d)) * 
        (2*9.274e-24*p.P)/(2*1.602e-19*p.Ms*(1+p.a**2)) - mu_Hfield*p.H - mu_Hslope*Hslope)
    x_next = state.x + dx

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(x_next - p.x_th, p.method, p.alpha)

    # compute reset
    x_new = (1 - z_new.detach()) * x_next + z_new.detach() * p.x_reset
    x_new = torch.where(x_new > 0, x_new, torch.tensor(0,dtype=torch.float32,device=torch.device("cuda")))

    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    return z_new, DWMTJLIFState(z=z_new, x=x_new, i=i_new)


def lif_step_sparse(
    input_tensor: torch.Tensor,
    state: DWMTJLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DWMTJParameters = DWMTJParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DWMTJLIFState]:  # pragma: no cover

    # constants
    Keff = 5e5 - (0.5*0.9*4e-7*math.pi*(p.Ms**2)) # effective anisotropy
    Delt = math.sqrt(1.3e-11/Keff) # DW width param (Bloch)
    Hslope = (6.000171568956471e4*((p.w2-p.w1)/p.x_th)**2 + 
            1.084982104400131e+04*((p.w2-p.w1)/p.x_th) + 1.228766907095338e+03)
    mu_Hfield = 4e-7*math.pi*1.7595e11*Delt/p.a  # DW mobility Neel
    mu_Hslope = 4e-7*math.pi*1.7595e11*Delt*p.a  # DW mobility Walker

    # compute voltage updates
    # dv = dt * p.tau_mem_inv * ((p.v_leak - state.v) + state.i)
    # v_decayed = state.v + dv
    w = state.x*((p.w2-p.w1)/p.x_th) + p.w1
    dx = dt * (((state.i * p.I)/(w * p.d)) * (2*9.274e-24*p.P)/(2*1.602e-19*p.Ms*(1+p.a**2)) 
                - mu_Hfield*p.H - mu_Hslope*Hslope)
    x_next = state.x + dx

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(x_next - p.x_th, p.method, p.alpha)

    # compute reset
    x_new = (1 - z_new) * x_next + z_new * p.x_reset
    x_new = torch.where(x_new > 0, x_new, 0.0)
    
    # compute current jumps
    i_new = (
        i_decayed
        + torch.sparse.mm(input_tensor, input_weights.t())
        + torch.sparse.mm(state.z, recurrent_weights.t())
    )

    z_sparse = z_new.to_sparse()
    return z_sparse, DWMTJLIFState(z_sparse, x_new, i_new)


def lif_step(
    input_tensor: torch.Tensor,
    state: DWMTJLIFState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: DWMTJParameters = DWMTJParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, DWMTJLIFState]:
    r"""Computes a single euler-integration step of a LIF neuron-model. More
    specifically it implements one integration step of the following ODE
    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} (v_{\text{leak}} - v + i) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}
    together with the jump condition
    .. math::
        z = \Theta(v - v_{\text{th}})
    and transition equations
    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}
    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.
    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    z, v, i = norse_op.lif_super_step(
        input_tensor, state, input_weights, recurrent_weights, p, dt
    )
    return z, DWMTJLIFState(z=z, v=v, i=i)