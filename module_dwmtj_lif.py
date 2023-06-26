import torch
import torch.jit
import math

from norse.torch.module.snn import SNNCell
from typing import NamedTuple, Optional, Tuple
from norse.torch.functional.threshold import threshold

class DWMTJParameters(NamedTuple):
    
    """Parametrization of a DWMTJ neuron

    Parameters:
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0/5e-10)
    #Width 1
    w1: torch.Tensor = torch.as_tensor(25e-9) #tested 25 to 50
    #Width 2
    w2: torch.Tensor = torch.as_tensor(25e-9) #tested 50 to 75
    d: torch.Tensor = torch.as_tensor(1.5e-9)
    P: torch.Tensor = torch.as_tensor(0.7)
    Ms: torch.Tensor = torch.as_tensor(8e5)
    a: torch.Tensor = torch.as_tensor(0.05)
    #External Magnetic Field
    H: torch.Tensor = torch.as_tensor(0.0) #tested 
    #Input current
    I: torch.Tensor = torch.as_tensor(80e-6) #tested 10 to 80
    #Threshold Location
    x_th: torch.Tensor = torch.as_tensor(200e-9) #tested 125 to 200
    #Reset Location
    x_reset: torch.Tensor = torch.as_tensor(0.0e-9) #tested 0 to 180
    method: str = "super"
    alpha: torch.Tensor = torch.as_tensor(100.0)

class DWMTJFeedForwardState(NamedTuple):
    """State of a feed forward DWMTJ

    Parameters:
        x (torch.Tensor): DW position
        i (torch.Tensor): synaptic input current
    """

    x: torch.Tensor
    i: torch.Tensor

class DWMTJCell(SNNCell):
    """Module that computes a single euler-integration step of a
    DW-MTJ neuron-model *without* recurrence and *without* time.

    Arguments:
        p (LIFParameters): Parameters of the LIF neuron model.
        dt (float): Time step to use. Defaults to 1e-10.
    """

    def __init__(self, p: DWMTJParameters = DWMTJParameters(), **kwargs):
        super().__init__(
            dwmtj_feed_forward_step,
            self.initial_state,
            p=p,
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor) -> DWMTJFeedForwardState:
        state = DWMTJFeedForwardState(
            x=torch.full(
                input_tensor.shape,
                self.p.x_reset.detach(),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            ),
        )
        state.x.requires_grad = True
        return state

class DWMTJParametersJIT(NamedTuple):
    
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
def _dwmtj_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: DWMTJFeedForwardState,
    p: DWMTJParametersJIT,
    dt: float = 1e-10,
) -> Tuple[torch.Tensor, DWMTJFeedForwardState]:  # pragma: no cover

    # constants
    Keff = 5e5 - (0.5*0.9*4e-7*math.pi*(p.Ms**2)) # effective anisotropy
    Delt = math.sqrt(1.3e-11/Keff) # DW width param (Bloch)
    Hslope = (6.000171568956471e4*((p.w2-p.w1)/p.x_th)**2 + 1.084982104400131e+04*((p.w2-p.w1)/p.x_th)
         + 1.228766907095338e+03)
    mu_Hfield = 4e-7*math.pi*1.7595e11*Delt/p.a  # DW mobility Neel
    mu_Hslope = 4e-7*math.pi*1.7595e11*Delt*p.a  # DW mobility Walker

    # compute DW position updates
    w = state.x*((p.w2-p.w1)/p.x_th) + p.w1
    dx = dt * (((state.i * p.I)/(w * p.d)) * 
        (2*9.274e-24*p.P)/(2*1.602e-19*p.Ms*(1+p.a**2)) - mu_Hfield*p.H - mu_Hslope*Hslope)
    x_next = state.x + dx

    # compute new spikes
    z_new = threshold(x_next - p.x_th, p.method, p.alpha)

    # compute reset
    x_new = (1 - z_new) * x_next + z_new * p.x_reset
    x_new = torch.where(x_new > 0, x_new, torch.tensor(0,dtype=torch.float32,device=torch.device("cuda")))

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di
    i_new = input_tensor + i_decayed

    return z_new, DWMTJFeedForwardState(x=x_new, i=i_new)

def dwmtj_feed_forward_step(
    input_tensor: torch.Tensor,
    state: Optional[DWMTJFeedForwardState] = None,
    p: DWMTJParameters = DWMTJParameters(),
    dt: float = 1e-10,
) -> Tuple[torch.Tensor, DWMTJFeedForwardState]:
    jit_params = DWMTJParametersJIT(
        tau_syn_inv=p.tau_syn_inv,
        w1=p.w1,
        w2 = p.w2,
        d=p.d,
        P=p.P,
        Ms=p.Ms,
        a=p.a,
        H = p.H,
        I = p.I,
        x_th=p.x_th,
        x_reset=p.x_reset,
        method=p.method,
        alpha=torch.as_tensor(p.alpha),
    )
    # Because input tensors are not directly used in the first pass (no
    # broadcasting takes place) we need to set the state values to the
    # same shape as the input.
    if state is None:
        state = DWMTJFeedForwardState(
            x=torch.full_like(input_tensor, jit_params.x_reset),
            i=torch.zeros_like(input_tensor),
        )
    return _dwmtj_feed_forward_step_jit(input_tensor, state=state, p=jit_params, dt=1e-10)