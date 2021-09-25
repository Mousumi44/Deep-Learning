import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from tqdm import tqdm #for nice progress bar

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
num_epochs = 2
batch_size = 64
learning_rate = 0.0001
load_model = True
ckpt_file = "my_checkpoint_bilstm.pth.tar"

#Dataset
train_dataset = datasets.FashionMNIST(root='./datasets/', train = True, transform = transforms.ToTensor(),  download = True)
test_dataset = datasets.FashionMNIST(root='./datasets/', train = False, transform = transforms.ToTensor(),  download = True)

train_loaded = DataLoader(train_dataset, batch_size, shuffle=True)
test_loaded = DataLoader(test_dataset, batch_size, shuffle=True)

#Bidirectional LSTM Network
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm =  nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


#Initialize Network
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

#Loss and Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

#Load Model
if load_model:
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])

#Training
for num in range(num_epochs):

    #Save Model
    checkpoint={"model_state_dict":model.state_dict(), "optim_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, ckpt_file)

    for batch_idx, (data, target) in enumerate(tqdm(train_loaded)):
        data = data.to(device).squeeze(1)
        target = target.to(device)

        #forward
        scores = model(data)

        #loss
        loss = loss_func(scores, target)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #Adam step
        optimizer.step()

#check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    #set model to eval
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)

            _, predictions = scores.max(dim=1)

            num_correct += (predictions==y).sum()
            num_samples += x.size(0)

    #retrun to train
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Training Set: {check_accuracy(train_loaded, model)*100:.2f} %')
print(f'Accuracy on Testing Set: {check_accuracy(test_loaded, model)*100:.2f} %')