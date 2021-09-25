import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

#Create Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_checkpoint(checkpoint, filename="my_checkpoint_ff.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])



#Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1
load_model = True

#Load Data
train_dataset = datasets.FashionMNIST(root='./datasets/',train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

test_dataset = datasets.FashionMNIST(root='./datasets/',train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True)

#initialize Network
model = NN(input_size, num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

#At first epoch assign load_model=False
if load_model:
    load_checkpoint(torch.load("my_checkpoint_ff.pth.tar"))

#Check accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")

    num_correct=0
    num_samples=0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)

            predictions = scores.argmax(dim=1)

            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)

        model.train()
        return num_correct/num_samples

#Train Network
for epoch in range(num_epochs):

    #save checkpoint
    checkpoint ={'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        data = data.reshape(data.shape[0], -1)

        #forward
        scores = model(data)
        loss = criterion(scores, targets)

        #backward
        optimizer.zero_grad() #grad set to zero for each bacth
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

print(f'Accuracy at eopch :{epoch} is {check_accuracy(train_loader, model)*100:.2f} %')
print(f'Accuracy at eopch :{epoch} is {check_accuracy(test_loader, model)*100:.2f} %')