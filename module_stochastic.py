import torch
import torch.jit
import math
import norse
from torch_interp1d import Interp1d

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
    reset: float = torch.as_tensor(0.0)

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
                torch.as_tensor(p.reset),
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

x = torch.tensor([0.5,1.000,1.100,1.250,1.400,1.500,1.600,1.750,1.825,1.900,2.000,2.200],device=torch.device("cuda"))
y = torch.tensor([0.001,0.025974025974025976, 0.023391812865497075, 0.02527075812274368, 0.017241379310344827, 0.09042553191489362, 
    0.07692307692307693, 0.09433962264150944, 0.17391304347826086, 0.6521739130434783, 0.9333333333333333, 0.999],device=torch.device("cuda"))

def stoch_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[StochFeedForwardState],
    p: StochParameters = StochParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:

    scale = 1
    new_in = input_tensor/(p.beta*scale) + 1.88
    prob = Interp1d()(x,y,new_in)
    # prob = torch.sigmoid(input_tensor)
    sample = torch.rand(prob.shape,device=torch.device("cuda"))
    z_new = torch.where(prob > sample, torch.ceil(prob), torch.floor(prob))

    return z_new, StochFeedForwardState(v=state.v,i=state.i)

def stoch_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: StochFeedForwardState,
    p: StochParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:  # pragma: no cover

    scale = 1
    new_in = input_tensor/(p.beta*scale) + 1.88
    prob = Interp1d()(x,y,new_in)
    # prob = torch.sigmoid(input_tensor)
    sample = torch.rand(prob.shape,device=torch.device("cuda"))
    z_new = torch.where(prob > sample, torch.ceil(prob), torch.floor(prob))

    return z_new.to_sparse, StochFeedForwardState(v=state.v,i=state.i)

class StochMWCell(SNNCell):
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
            activation=stochmw_feed_forward_step,
            activation_sparse= stochmw_feed_forward_step_sparse,
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

a = torch.tensor([
       0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3,
       1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
       2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
       4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2,
       5.3, 5.4, 5.5, 5.6, 5.7],
    device=torch.device("cuda"))
d1 = torch.tensor([
       1. , 1. , 0.7, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2,
       0.2, 0.1, 0.1, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. ],
    device=torch.device("cuda"))
d2 = torch.tensor([
       0. , 0. , 0.3, 0.6, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0. , 0.1, 0.1,
       0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. ],
    device=torch.device("cuda"))
d3 = torch.tensor([
       0. , 0. , 0. , 0. , 0.1, 0.5, 0.5, 0.5, 0.4, 0.5, 0.5, 0.6, 0.6, 0.5,
       0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5,
       0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
       0.3, 0.3, 0.3, 0.2, 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. ],
    device=torch.device("cuda"))
d4 = torch.tensor([
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.1, 0.1, 0.1, 0.1, 0.2,
       0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
       0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5,
       0.4, 0.4, 0.4, 0.5, 0.5, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.1, 0.1,
       0.1, 0.1, 0. , 0. , 0. ],
    device=torch.device("cuda"))
d5 = torch.tensor([
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
       0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
       0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2,
       0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.7, 0.9, 0.9,
       0.9, 0.9, 1. , 1. , 1. 
    ],device=torch.device("cuda"))

def stochmw_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[StochFeedForwardState],
    p: StochParameters = StochParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:
    
    new_in = torch.abs(input_tensor*p.beta)
    prob1 = Interp1d()(a,d1,new_in)
    prob2 = Interp1d()(a,d2,new_in)
    prob3 = Interp1d()(a,d3,new_in)
    prob4 = Interp1d()(a,d4,new_in)
    sample = torch.rand(input_tensor.shape,device=torch.device("cuda"))

    zero_tens = input_tensor*0
    dx = torch.where(torch.logical_and(sample>prob1,sample<=prob1+prob2),0.25+zero_tens,zero_tens)
    dx = torch.where(torch.logical_and(sample>prob1+prob2,sample<=prob1+prob2+prob3),zero_tens+0.50,dx)
    dx = torch.where(torch.logical_and(sample>prob1+prob2+prob3,sample<=prob1+prob2+prob3+prob4),zero_tens+0.75,dx)
    dx = torch.where(sample>prob1+prob2+prob3+prob4,zero_tens+1.0,dx)
    dx = torch.where(input_tensor<0,-dx,dx)

    v_new = state.v+dx
    v_new = torch.where(v_new<0,zero_tens,v_new)
    z_new = threshold(v_new-0.9,p.method,p.alpha)
    
    v_new = (1-z_new)*v_new + z_new*p.reset

    return z_new, StochFeedForwardState(v=v_new,i=state.i)

def stochmw_feed_forward_step_sparse(
    input_tensor: torch.Tensor,
    state: StochFeedForwardState,
    p: StochParameters,
    dt: float = 0.001,
) -> Tuple[torch.Tensor, StochFeedForwardState]:  # pragma: no cover

    scale = 1
    new_in = input_tensor/(p.beta*scale) + 1.88
    prob = Interp1d()(x,y,new_in)
    sample = torch.rand(prob.shape,device=torch.device("cuda"))
    z_new = torch.where(prob > sample, torch.ceil(prob), torch.floor(prob))

    return z_new.to_sparse, StochFeedForwardState(v=state.v,i=state.i)