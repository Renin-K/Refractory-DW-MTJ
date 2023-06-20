import torch
import numpy as np
import torchvision
import os

from noise_transform import AddGaussianNoise

from module_stochastic import StochParameters
from module_stochastic import StochCell,StochMWCell

from norse.torch.module.leaky_integrator import LInoweight,LILinearCell
from norse.torch import LIFParameters,LIParameters
from norse.torch.module.lif import LIFCell
from norse.torch.module import encode

from tqdm import tqdm, trange

# folder to save results
target_dir = "230118_s05_fashion_noise0.0_seqnet0500_neurmw_ep20_lr0.001_T40_alpha100_beta1_pf1e3_bn_initxunif"
if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)

BATCH_SIZE = 100
EPOCHS = 20
T = 40
LR = 0.001
alpha = 100
beta = 5
f_poisson = 1e3
seeds = 5
clipflag = False
clip = 0.2
decay = 0
weight_init_gain = 1
hidden = 500
noise_std = 0.0

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

# train_data = torchvision.datasets.MNIST(
#     root=".",
#     train=True,
#     download=True,
#     transform=transform,
# )

# train_loader = torch.utils.data.DataLoader(
#     train_data,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST(
#         root=".",
#         train=False,
#         transform=transform,
#     ),
#     batch_size=BATCH_SIZE
# )

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

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
        # self.stochfc = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.out = LILinearCell(h2,10,p=LIFParameters(method=method, alpha=alpha))
        self.out = LInoweight(p=LIFParameters(method=method, alpha=alpha))
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
            z, s0 = self.stoch0(z, s0)
            z = self.bn0(z)
            # print(z)
            z = self.fc1(z)
            z, s1 = self.stoch1(z, s1)
            z = self.bn1(z)
            z = self.fc2(z)        
            # z = self.bnout(z)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages

class StochNet(torch.nn.Module):
    def __init__(
        self,  h1=1000, h2=1000, feature_size=28, beta=20, method="super", alpha=100
    ):
        super(StochNet, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.fc1 = torch.nn.Linear(h1, h2)
        self.fc2 = torch.nn.Linear(h2, 10)
        self.bn0 = torch.nn.BatchNorm1d(h1)
        self.bn1 = torch.nn.BatchNorm1d(h2)
        self.bnout = torch.nn.BatchNorm1d(10)
        self.stoch0 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch1 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch2 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stoch1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stoch2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stochfc = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.out = LILinearCell(h2,10,p=LIFParameters(method=method, alpha=alpha))
        self.out = LInoweight(p=LIFParameters(method=method, alpha=alpha))
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
            z, s0 = self.stoch0(z, s0)
            z = self.bn0(z)
            # print(z)
            z = self.fc1(z)
            z, s1 = self.stoch1(z, s1)
            z = self.bn1(z)
            z = self.fc2(z)        
            # z = self.bnout(z)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages


class SeqNetMW(torch.nn.Module):
    def __init__(
        self,  h1=1000, h2=1000, feature_size=28, beta=beta, method="super", alpha=100
    ):
        super(SeqNetMW, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.fc1 = torch.nn.Linear(h1, h2)
        self.fc2 = torch.nn.Linear(h2, 10)
        self.bn0 = torch.nn.BatchNorm1d(h1)
        self.bn1 = torch.nn.BatchNorm1d(h2)
        self.bnout = torch.nn.BatchNorm1d(10)
        self.stoch0 = StochMWCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch1 = StochMWCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch2 = StochMWCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stoch1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stoch2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.stochfc = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        # self.out = LILinearCell(h2,10,p=LIFParameters(method=method, alpha=alpha))
        self.out = LInoweight()
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
            z, s0 = self.stoch0(z, s0)
            z = self.bn0(z)
            # print(z)
            z = self.fc1(z)
            z, s1 = self.stoch1(z, s1)
            z = self.bn1(z)
            z = self.fc2(z)        
            # z = self.bnout(z)
            v, so = self.out(z, so)
            voltages[ts, :, :] = v
        return voltages

class SeqNetSmall(torch.nn.Module):
    def __init__(
        self,  h1=20, feature_size=28, beta=20, method="super", alpha=100
    ):
        super(SeqNetSmall, self).__init__()

        self.fc0 = torch.nn.Linear(feature_size*feature_size, h1)
        self.fc1 = torch.nn.Linear(h1, 10)
        self.bn0 = torch.nn.BatchNorm1d(h1)
        # self.stoch0 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch1 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        # self.stoch2 = StochCell(p=StochParameters(beta=beta,method=method, alpha=alpha))
        self.stoch0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.stoch1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.stoch2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        #self.stochfc = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.out = LILinearCell(h1, 10, p=LIFParameters(method=method, alpha=alpha))
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
            # z = self.bn0(z)
            z, s0 = self.stoch0(z, s0)
            # print(z)
            v = self.fc1(z)
            # z = self.bn1(z)
            # v, s1 = self.stoch1(z, s1)
            # v = self.fc2(z)        
            #v, s2 = self.stoch2(z, s2)
            # v, so = self.out(torch.nn.functional.relu(z), so)
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

def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        if clipflag:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip, norm_type=2)
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

def test(model, device, test_loader, epoch):
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

def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        # model,
        path,
    )

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data,gain=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
highest = 0
for i in range(seeds):
    # torch.manual_seed(i)
    # snn = SeqNet(h1=hidden,h2=hidden,beta=beta,alpha=alpha)
    snn = SeqNetMW(h1=hidden,h2=hidden,beta=beta,alpha=alpha)
    # snn = SeqNetSmall(h1=hidden,beta=beta,alpha=alpha)
    snn.apply(initialize_weights)
    model = Model(
        # encoder=encode.ConstantCurrentLIFEncoder(T),
        encoder=encode.PoissonEncoder(seq_length=T,f_max=f_poisson),
        # snn=ConvNet(alpha=alpha,k=k,beta=beta),
        snn=snn,
        decoder=decode
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=decay)

    pbar = trange(EPOCHS, ncols=100, unit="epoch")
    for epoch in pbar:
        training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
        test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        if accuracy > highest:
            model_path = "./outputs/" + target_dir + "/fmnist_dwmtj.pt"
            save(
                model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
            )
        accuracies.append(accuracy)       
        pbar.set_postfix(accuracy=accuracies[-4:])
    print(f"accuracies: {accuracies[-EPOCHS:]}")

if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(training_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses).reshape(seeds,EPOCHS).T)
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies).reshape(seeds,EPOCHS).T)