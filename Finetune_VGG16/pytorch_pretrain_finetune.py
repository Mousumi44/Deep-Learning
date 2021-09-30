import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import sys

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5
load_model = True
ckpt_file = 'my_checkpoint_vgg16.pth.tar'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#Model
model = torchvision.models.vgg16(pretrained=True)

#Finetuning so freezing the last layers
for param in model.parameters():
    param.requires_grad = False

model.avgpool = Identity() #Erase  average pooling layer
model.classifier = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Linear(100, num_classes)
)

model.to(device)
# print(model)
# sys.exit()

#Load Data
train_data = datasets.CIFAR10(
    root='./dataset/', train=True, transform=transforms.ToTensor(), download=True
)
train_loaded = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = datasets.CIFAR10(
    root='./dataset/', train=False, transform=transforms.ToTensor(), download=True
)
test_loaded = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

#Train Network
for epoch in range(num_epochs):

    #Save Model
    checkpoint ={"model_state_dict": model.state_dict(), "optim_state_dict":optimizer.state_dict()}
    torch.save(checkpoint, ckpt_file)


    for batch_idx, (data,targets) in enumerate(train_loaded):
        data = data.to(device)
        targets = targets.to(device)

        #forward
        scores = model(data)
        loss = loss_func(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

def check_accuracy(loader, model):
    num_correct=0
    num_samples=0

    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(dim=1)

            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

print(f'Accuracy on Training Data: {check_accuracy(train_loaded, model)*100:.2f} %')
print(f'Accuracy on Testing Data: {check_accuracy(test_loaded, model)*100:.2f} %')