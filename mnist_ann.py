import torch
from torch._C import device
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm, trange
import os

# folder to save results
target_dir = "0712_mnist_ann"

BATCH_SIZE = 100
FEATURE_SIZE = 28
INPUT_SIZE = FEATURE_SIZE ** 2

mnist_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_data_mnist = torchvision.datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=mnist_transform,
)
test_data_mnist = torchvision.datasets.MNIST(
    root=".",
    train=False,
    transform=mnist_transform,
)
train_loader_mnist = torch.utils.data.DataLoader(
    train_data_mnist,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader_mnist = torch.utils.data.DataLoader(
    test_data_mnist,
    batch_size=BATCH_SIZE
)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.features = int(((FEATURE_SIZE - 4) / 2 - 4) / 2)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(self.features**2 * 50, 500)
        self.out = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = x.view(-1, 4 ** 2 * 50)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer):
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

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
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
        path,
    )

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

train_loader = train_loader_mnist
test_loader = test_loader_mnist

rseed = 0
torch.manual_seed(rseed)
lr = 0.001
model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 5

train_losses = []
test_losses = []
mean_losses = []
accuracies = []
pbar = trange(epochs, ncols=80, unit="epoch")
for epoch in pbar:
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer)
    test_loss, accuracy = test(model, DEVICE, test_loader)
    train_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)       
    pbar.set_postfix(accuracy=accuracies[-1])

os.mkdir("./outputs/" + target_dir)
np.save("./outputs/" + target_dir + "/training_losses.npy", np.array(train_losses))
np.save("./outputs/" + target_dir + "/mean_losses.npy", np.array(mean_losses))
np.save("./outputs/" + target_dir + "/test_losses.npy", np.array(test_losses))
np.save("./outputs/" + target_dir + "/accuracies.npy", np.array(accuracies))
model_path = "./outputs/" + target_dir + "/model.pt"
save(
    model_path,
    epoch=epoch,
    model=model,
    optimizer=optimizer,
)
