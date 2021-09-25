import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm #For Nice Progress bar!

#Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels =1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(
            in_channels = in_channels,
            out_channels = 8,
            kernel_size = (3,3),
            stride = (1,1),
            padding= (1,1)
        ) #this k, s, p keeps the same dimension for input
        self.pool = nn.MaxPool2d(
            kernel_size = (2,2),
            stride = (2,2)  
        ) #this k, p for max pool half the input dimension

        self.conv2=nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = (3,3),
            stride = (1,1),
            padding= (1,1)
        ) #this k, s, p keeps the same dimension for input

        self.fc1 = nn.Linear(
            16*7*7, num_classes
        ) #16 is for conv2 output channel, I'll do maxpool twice in forward, so 28/2, 14/2 = 7 will hold 28: input_feature dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
in_channels = 1
num_classes =10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
load_model = True
ckpt_file = "my_checkpoint_cnn.pth.tar"

#Load Data
train_dataset = datasets.FashionMNIST(root="./datasets/", train = True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(root="./datasets/", train = False, transform=transforms.ToTensor(), download=True)

train_loaded = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Initialize Network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

#Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Load Model
if load_model:
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

#Train Network
for epoch in range(num_epochs):

    #Save Model
    checkpoint = {"model_state_dict":model.state_dict(), "optim_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, ckpt_file)

    for batch_idx, (data, targets) in enumerate(tqdm(train_loaded)):
        data = data.to(device)
        targets = targets.to(device)

        #forward
        scores = model(data)
        loss = loss_func(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #optimize gradient descent or adam step
        optimizer.step()

#Check accuracy on train and test to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(dim=1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(dim=0)
            
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Training Set: {check_accuracy(train_loaded, model)*100:.2f} %')
print(f'Accuracy on Testing Set: {check_accuracy(test_loaded, model)*100:.2f} %')