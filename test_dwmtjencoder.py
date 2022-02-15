import torchvision
import numpy as np
import matplotlib.pyplot as plt

import module_dwmtj_lif as dwmtj
from module_dwmtj_encoder import ConstantCurrentDWMTJEncoder
from norse.torch.module import encode

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_data = torchvision.datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform,
)

img, label = train_data[0]
# plt.imshow(img)
T = 32
example_encoder = encode.PoissonEncoder(seq_length=T,dt=1e-10,f_max=10e8)
# example_encoder = ConstantCurrentDWMTJEncoder(seq_length=T)
example_input = example_encoder(img)
# np.savetxt('spikes.txt',example_input.numpy())
example_spikes = example_input.reshape(T,28*28).to_sparse().coalesce()
one = example_input.reshape(T,28*28)
np.savetxt('spikes.txt',one.numpy())
np.savetxt('spikes.csv',one.numpy(),delimiter=',')
two = one.reshape(T,1,28,28)
t = example_spikes.indices()[0]
n = example_spikes.indices()[1]

plt.scatter(t, n, marker='|', color='black')
plt.ylabel('Input Unit')
plt.xlabel('Time [0.1ns]')
plt.savefig('spikes.png')