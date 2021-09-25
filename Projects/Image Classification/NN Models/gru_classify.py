import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
num_epochs = 3
batch_size = 64
load_model = True
ckpt_file = "my_checkpoint_gru.pth.tar"

class GRU(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(sequence_length*hidden_size, num_classes)

    def forward(self, x):
        #num_layer*batch_size*hidden_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out.reshape( out.shape[0] ,-1)
        out = self.fc(out)
        return out
    
#Load Dataset
train_dataset =  datasets.FashionMNIST(root='./datasets/', train = True, transform=transforms.ToTensor(), download = True)
test_dataset =  datasets.FashionMNIST(root='./datasets/', train = False, transform=transforms.ToTensor(), download = True)
train_loaded = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Initialize Network
model = GRU(input_size, num_layers, hidden_size, num_classes).to(device)

#Loss Function and Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Load Model
if load_model:
    checkpoint=torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

#Training
for epoch in range(num_epochs):

    #Save Model
    checkpoint ={"model_state_dict": model.state_dict(), "optim_state_dict":optimizer.state_dict()}
    torch.save(checkpoint, ckpt_file)

    for batch_idx, (data, targets) in enumerate(tqdm(train_loaded)):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        #forward
        scores = model(data)

        #loss function
        loss = loss_func(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #optimize or adam step
        optimizer.step()

#Check Accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    #set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)
            scores = model(x)
            _, predictions = scores.max(dim=1)

            num_correct += (predictions==y).sum()
            num_samples += x.shape[0]
            
    #Toggle model back to training
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Training Data: {check_accuracy(train_loaded, model)*100:.2f} %')
print(f'Accuracy on Testing Data: {check_accuracy(test_loaded, model)*100:.2f} %')