import os 
import torch
import numpy as np
import torchvision
import module_dwmtj_lif as dwmtj
import module_dwmtj_recurrent as dwmtjrec
from norse.torch.module import encode

from norse.torch.module.leaky_integrator import LILinearCell,LICell

import os
from tqdm import tqdm, trange

# set manual seed for PyTorch to reuse weights
torch.manual_seed(5)    # use this for the convolutional network

BATCH_SIZE = 100

# folder to save results
target_dir = "230620_temp"

# if folder does not exist, create it
if not os.path.isdir("./outputs/"):
    os.mkdir("./outputs/")
if not os.path.isdir("./outputs/" + target_dir):
    os.mkdir("./outputs/" + target_dir)

# normalize input images
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

# train data and loader
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

# test data and loader
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.FashionMNIST(
        root=".",
        train=False,
        transform=transform,
    ),
    batch_size=BATCH_SIZE
)

# define decoder to decode output spikes
def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

# define convolutional network class
class ConvNet(torch.nn.Module):
    def __init__( self,  num_channels=1, feature_size=28, method="super", 
                alpha=100, Hleak=0, w1 = 25e-9, w2=25e-9, I =  80e-6, x_th = 200e-9, x_reset = 0.0e-9):
        super(ConvNet, self).__init__()

        self.features = int(((feature_size - 4) / 2 - 4) / 2)               # features at the output of convolution
        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)                # convolutional filters
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500) # fully connected layer after convolution
        # dwmtj neuron layers
        self.dwmtj0 = dwmtj.DWMTJCell(p=dwmtj.DWMTJParameters(
            method=method,alpha=alpha,H=torch.as_tensor(Hleak),w2=torch.as_tensor(w2),I=torch.as_tensor(I),x_th=torch.as_tensor(x_th),x_reset =torch.as_tensor(x_reset))
            )
        self.dwmtj1 = dwmtj.DWMTJCell(p=dwmtj.DWMTJParameters(
            method=method,alpha=alpha,H=torch.as_tensor(Hleak),w2=torch.as_tensor(w2),I=torch.as_tensor(I),x_th=torch.as_tensor(x_th),x_reset =torch.as_tensor(x_reset))
            )
        self.dwmtj2 = dwmtj.DWMTJCell(p=dwmtj.DWMTJParameters(
            method=method,alpha=alpha,H=torch.as_tensor(Hleak),w2=torch.as_tensor(w2),I=torch.as_tensor(I),x_th=torch.as_tensor(x_th),x_reset =torch.as_tensor(x_reset))
            )
        # output layer
        self.out = LILinearCell(500, 10)

    def forward(self, x):
        # extract sequence length and batch_size from input data
        seq_length = x.shape[0]
        batch_size = x.shape[1]
        
        # specify the initial neuron states
        s0 = s1 = s2 = so = None

        # initialize output voltages
        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )
        
        # feed forward steps of the neural network
        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.dwmtj0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.dwmtj1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, 4 ** 2 * 50)
            z = self.fc1(z)        
            z, s2 = self.dwmtj2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages

# define recurrent network class
class RecNet(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, 
                method="super", alpha=100, Hleak=0, w2=25e-9, I=100e-6, dt=1e-10):
        super(RecNet, self).__init__()
        self.l1 = dwmtjrec.DWMTJRecurrentCell(
            input_size=input_features,
            hidden_size=hidden_features,
            p=dwmtjrec.DWMTJParameters(method=method,alpha=alpha,H=Hleak,w2=w2,I=I),
            dt=dt                     
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        
    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            voltages += [vo]

        return torch.stack(voltages)

# define model container, putting together the model, encoder, and decoder
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

# training function: feeds data forward, gets loss function, then applies optimization
def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []
    for (data, target) in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

# test function: feeds data forward, then compares to correct class
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

# save model weights
def save(path, epoch, model, optimizer, is_best=False):
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

# model parameters
EPOCHS = 4         # number of iterations
T = [40]            # list of number of timesteps
LR = 5e-3           # learning rate
SEED = 1            # number of seeds to run the network (kept as 1 if manual seed is applied)
MTYPE = 'conv'      # snn type

# cpu is broken, but kept here in case I ever fix it
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# sweep parameters (define as needed)
f_poisson = np.linspace(10e9,10e9,1)

w1 = np.linspace(50e-9,25e-9,7)
w2 = np.linspace(50e-9,75e-9,7)



fin_acc = []    # empty array to hold final accuracies
# sweep variables of interest
for f in range(0,len(w1)):
    #for h in range(0,len(T)):
    # seed counter
    s = 0   
    while s < SEED:
        torch.manual_seed(s)    # set torch manual seed (convolutional)

        if MTYPE == 'rec':
            snn = RecNet(28*28, 100, 10)
        elif MTYPE == 'conv':
            snn = ConvNet(alpha=100, w1 = w1[f], w2 = w2[f])

        model = Model( # instantiate a model
            encoder=encode.PoissonEncoder(seq_length=int(T[0]),dt=1e-10,f_max = f_poisson[0]),
            snn=snn,
            decoder=decode
        ).to(DEVICE)

        # set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        accuracies = [] # empty containger for the accuracy
        pbar = trange(EPOCHS, ncols=160, unit="epoch")
        for epoch in pbar:
            training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
            test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
            # training_losses += training_loss
            # mean_losses.append(mean_loss)
            # test_losses.append(test_loss)
            accuracies.append(accuracy)       
            pbar.set_postfix(accuracy=accuracies)
        fin_acc.append(accuracies)
        s += 1
        np.save("./outputs/" + target_dir + "/fin_acc.npy", np.concatenate(fin_acc))
# reshape accuracies to a format that makes sense, then save
fin_acc = np.concatenate(fin_acc).reshape(SEED, EPOCHS * (len(w1)))
np.save("./outputs/" + target_dir + "/fin_acc.npy", np.array(fin_acc))