import torch
import torch.jit
import math

from norse.torch.module.snn import SNNCell
from typing import NamedTuple, Optional, Tuple
from norse.torch.functional.threshold import threshold
from module_dwmtj_lif import DWMTJParameters

class ConstantCurrentDWMTJEncoder(torch.nn.Module):
    def __init__(
        self,
        seq_length: int,
        p: DWMTJParameters = DWMTJParameters(),
        dt: float = 1e-10,
    ):
        super(ConstantCurrentDWMTJEncoder, self).__init__()
        self.seq_length = seq_length
        self.p = p
        self.dt = dt

    def forward(self, input_currents):
        return constant_current_dwmtj_encode(
            input_currents,
            seq_length=self.seq_length,
            p=self.p,
            dt=self.dt,
        )

def constant_current_dwmtj_encode(
    input_current: torch.Tensor,
    seq_length: int,
    p: DWMTJParameters = DWMTJParameters(),
    dt: float = 1e-10,
) -> torch.Tensor:

    x = torch.zeros(*input_current.shape, device=input_current.device)
    z = torch.zeros(*input_current.shape, device=input_current.device)
    spikes = torch.zeros(seq_length, *input_current.shape, device=input_current.device)

    for ts in range(seq_length):
        z, x = dwmtj_current_encoder(input_current=input_current, x = x, p=p, dt=dt)
        spikes[ts] = z
    return spikes

def dwmtj_current_encoder(
    input_current: torch.Tensor,
    x: torch.Tensor,
    p: DWMTJParameters = DWMTJParameters(),
    dt: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # constants
    e = 1.602e-19  # electron charge
    mu_B = 9.274e-24  # Bohr magneton
    mu_0 = 4e-7*math.pi  # vacuum permeability
    GammaE = 1.7595e11  # electron gyromagnetic ratio
    GammaLL = mu_0*GammaE  # Gilbert gyromagnetic ratio
    Aex = 1.3e-11  # exchange parameter
    Ku = 5e5  # uniaxial anisotropy
    Keff = Ku - (0.5*0.9*mu_0*(p.Ms**2)) # effective anisotropy
    Delt = math.sqrt(Aex/Keff) # DW width param (Bloch)
    deltw = (p.w2-p.w1)/p.x_th
    Hslope = (6.000171568956471e4*deltw**2 + 1.084982104400131e+04*deltw + 1.228766907095338e+03)

    # compute DW position updates
    mu_Hfield = GammaLL*Delt/p.a  # DW mobility Neel
    mu_Hslope = GammaLL*Delt*p.a  # DW mobility Walker
    w = x*deltw + p.w1
    dx = dt * ((input_current * p.I / (w * p.d)) * 
        (2*mu_B*p.P)/(2*e*p.Ms*(1+p.a**2)) - mu_Hfield*p.H - mu_Hslope*Hslope)
    x_next = x + dx

    # compute new spikes
    z_new = threshold(x_next - p.x_th, p.method, p.alpha)

    # compute reset
    x_new = (1 - z_new) * x_next + z_new * p.x_reset
    x_new = torch.where(x_new < 0., torch.zeros_like(x_new), x_new)

    return z_new, x_new
