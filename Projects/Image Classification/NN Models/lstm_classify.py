import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


#set device
device = ("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
input_size = 28
sequence_length =28
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.0001
batch_size = 64
num_epochs = 3
load_model = True
ckpt_file = "my_checkpoint_lstm.pth.tar"

#LSTM network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

        #If we want to consider only last hidden state
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))

        #out = out.reshape(out.shape[0], -1)

        #If we want to consider only last hidden state
        out = out[:, -1, :] #All batch, last hidden state, all features

        out = self.fc(out)
        return out

#Dataset
train_dataset =  datasets.FashionMNIST(root='./datasets/', train = True, transform=transforms.ToTensor(), download = True)
test_dataset =  datasets.FashionMNIST(root='./datasets/', train = False, transform=transforms.ToTensor(), download = True)
train_loaded = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Initialize Network
model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

#Loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

#Train Network
for epoch in range(num_epochs):

    #Save model
    checkpoint = {"model_state_dict": model.state_dict(), "optim_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, ckpt_file)

    for batch_idx, (data, targets) in enumerate(tqdm(train_loaded)):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        #forward
        scores = model(data)

        #loss 
        loss = loss_func(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #optimization
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    #set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y=y.to(device)

            scores = model(x)

            _, predictions = torch.max(scores, dim=1)
            num_correct += (predictions==y).sum()
            num_samples += x.shape[0]
            
    #Toggle model back to training
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Training set: {check_accuracy(train_loaded, model)*100:.2f} %')
print(f'Accuracy on Testing set: {check_accuracy(test_loaded, model)*100:.2f} %')