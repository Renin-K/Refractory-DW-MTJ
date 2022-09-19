import torch
import torch.jit
import math
import norse

from norse.torch.module.snn import SNNCell
from typing import NamedTuple, Optional, Tuple
from norse.torch.functional.threshold import threshold

class StochParameters(NamedTuple):
    """Parametrization of a LIF neuron

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
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    beta: float = torch.as_tensor(1)
    method: str = "super"
    alpha: float = torch.as_tensor(100.0)

class StochFeedForwardState(NamedTuple):
    """State of a feed forward LIF neuron

    Parameters:
        i (torch.Tensor): synaptic input current
    """
    v: torch.Tensor
    i: torch.Tensor

class StochCell(SNNCell):
    """Module that computes a single euler-integration step of a
    leaky integrate-and-fire (LIF) neuron-model *without* recurrence and *without* time.

    More specifically it implements one integration step
    of the following ODE

    .. math::
        \\begin{align*}
            \\dot{v} &= 1/\\tau_{\\text{mem}} (v_{\\text{leak}} - v + i) \\
            \\dot{i} &= -1/\\tau_{\\text{syn}} i
        \\end{align*}

    together with the jump condition

    .. math::
        z = \\Theta(v - v_{\\text{th}})

    and transition equations

    .. math::
        \\begin{align*}
            v &= (1-z) v + z v_{\\text{reset}}
        \\end{align*}

    Example:
        >>> data = torch.zeros(5, 2) # 5 batches, 2 neurons
        >>> l = LIFCell(2, 4)
        >>> l(data) # Returns tuple of (Tensor(5, 4), LIFState)

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        sparse (bool): Whether to apply sparse activation functions (True) or not (False). Defaults to False.
        dt (float): Time step to use. Defaults to 0.001.
    """

    def __init__(self, p: StochParameters = StochParameters(), **kwargs):
        super().__init__(
            activation=stoch_feed_forward_step,
            activation_sparse= stoch_feed_forward_step_sparse,
            state_fallback=self.initial_state,
            p=StochParameters(
                torch.as_tensor(p.beta),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> StochFeedForwardState:
        state = StochFeedForwardState(
            v=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state

class StochParametersJIT(NamedTuple):
    """Parametrization of a LIF neuron

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

    beta: torch.Tensor
    method: str
    alpha: torch.Tensor

@torch.jit.script
def _stoch_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: StochFeedForwardState,
    p: StochParametersJIT,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:  # pragma: no cover
    # compute voltage updates

    prob = torch.sigmoid(p.beta*input_tensor-10)
    sample = torch.rand(prob.shape,device=torch.device("cuda"))
    z_new = torch.where(prob > sample, torch.ceil(prob), torch.floor(prob))

    return z_new, StochFeedForwardState(v=state.v,i=state.i)

def stoch_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[StochFeedForwardState],
    p: StochParameters = StochParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:

    jit_params = StochParametersJIT(
        beta = p.beta,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    return _stoch_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=dt)

def stoch_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: StochFeedForwardState,
    p: StochParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:  # pragma: no cover

    prob = torch.sigmoid(p.beta*input_tensor-10)
    sample = torch.rand(prob.shape,device=torch.device("cuda"))
    z_new = torch.where(prob > sample, torch.ceil(prob), torch.floor(prob))

    return z_new.to_sparse, StochFeedForwardState(v=state.v,i=state.i)