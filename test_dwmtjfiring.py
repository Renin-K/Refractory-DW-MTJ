import torch
import numpy as np
import matplotlib.pyplot as plt

import module_dwmtj_lif as dwmtj
from module_dwmtj_encoder import dwmtj_current_encoder

N = 1 # number of neurons to consider
T = 100 # number of timesteps to integrate

p = dwmtj.DWMTJParameters(w2=torch.as_tensor(100e-9))
x = torch.as_tensor(0e-9) # initial DW position
input_current_0 = torch.zeros(N)
input_current_1 = torch.ones(N)
input_current_n1 = -torch.ones(N)

dwpos = []

for ts in range(0,50):
  z, x = dwmtj_current_encoder(input_current_0, x, p)
  dwpos.append(x)

for ts in range(50,100):
  z, x = dwmtj_current_encoder(input_current_0, x, p)
  dwpos.append(x)

dwpos = torch.stack(dwpos)

plt.ylabel("x")
plt.xlabel("time [0.1ns]")
plt.plot(dwpos)
plt.show()