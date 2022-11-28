import torch
import numpy as np
import torchvision
import os

from noise_transform import AddGaussianNoise

from module_stochastic import StochParameters
from module_stochastic import StochCell

from norse.torch.module.leaky_integrator import LI,LILinearCell
from norse.torch import LIFParameters,LIParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module import encode
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

# folder to save results
target_dir = "1115_inf_lif"
if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
load_dir = "1111_s05_fashion_noise0.0_seqnet0500_lif_ep10_lr1e-3_T50_alpha100_beta25_pf1e3_noclip_nodecay_bn_initxunif"

BATCH_SIZE = 100
EPOCHS = 10
T = 50
LR = 1e-3
k = 15
alpha = 100
beta = 25
f_poisson = 1e3
seeds = 5
clipflag = False
clip = 0.2
decay = 0
weight_init_gain = 1
hidden = 500
noise_std = 3.0

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        AddGaussianNoise(mean=0,std=noise_std),
    ]
)

train_data = torchvision.datasets.FashionMNIST(
    root=".",
    train=True,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST(
        root=".",
        train=False,
        transform=transform,
    ),
    batch_size=BATCH_SIZE
)

dataiter = iter(test_loader)
images,labels = dataiter.next()
figim,axim = plt.subplots(1,1,figsize=(2,1.4))
print(images[0])
axim.imshow((images[0].reshape(28,28)),cmap='gray')
axim.get_xaxis().set_visible(False)
axim.get_yaxis().set_visible(False)
figim.savefig(f'noiseimage_{noise_std}.svg')

class SeqNet(torch.nn.Module):
    def __init__(
        self,  h1=1000, h2=1000, feature_size=28, beta=20, method="super", alpha=100
    ):
        super(SeqNet, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.fc1 = torch.nn.Linear(h1, h2)
        self.fc2 = torch.nn.Linear(h2, 10)
        self.bn0 = torch.nn.BatchNorm1d(h1)
        self.bn1 = torch.nn.BatchNorm1d(h2)
        self.bnout = torch.nn.BatchNorm1d(10)
        # self.stoch0 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch1 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch2 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.stoch1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.stoch2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        #self.stochfc = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        #self.out = LILinearCell(h2,10,p=LIFParameters(method=method, alpha=alpha))
        self.out = LI()
        # self.out = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        
        # specify the initial states
        s0 = s1 = s2 = so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.fc0(x[ts, :].view(-1,28*28))
            z = self.bn0(z)
            z, s0 = self.stoch0(z, s0)
            # print(z)
            z = self.fc1(z)
            z = self.bn1(z)
            z, s1 = self.stoch1(z, s1)
            z = self.fc2(z)        
            z = self.bnout(z)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages

class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

test_losses = []
accuracies = []
highest = 0

noise_std_range = np.linspace(0,3,16)

snn = SeqNet(h1=hidden,h2=hidden,beta=beta,alpha=alpha)
model = Model(
    # encoder=encode.ConstantCurrentLIFEncoder(T),
    encoder=encode.PoissonEncoder(seq_length=T,f_max=f_poisson),
    # snn=ConvNet(alpha=alpha,k=k,beta=beta),
    snn=snn,
    decoder=decode
).to(DEVICE)

model.load_state_dict(torch.load("./outputs/"+load_dir+"/fmnist_dwmtj.pt")["model_state_dict"])
pbar = tqdm(noise_std_range,ncols=80)
for noise in pbar:
    for s in range(seeds):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                AddGaussianNoise(mean=0,std=noise),
            ]
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(
                root=".",
                train=False,
                transform=transform,
            ),
            batch_size=BATCH_SIZE
        )
        test_loss, accuracy = test(model, DEVICE, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)  
    pbar.set_postfix(noise=noise,acc=np.mean(np.array(accuracies)[-5:]))
accs_np = np.array(accuracies).reshape(len(noise_std_range),seeds)
print(accs_np)

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/accs_np.npy",accs_np)
np.save("./outputs/" + target_dir + "/noise_std_range.npy",noise_std_range)
# np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
# np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
# np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,EPOCHS).T)
# np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,EPOCHS).T)