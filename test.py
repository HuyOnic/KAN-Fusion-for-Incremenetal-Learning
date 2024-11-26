import pandas as pd
import matplotlib.pyplot as plt
import torch 
from torch import nn
from collections import Counter
import numpy as np
from model.KAN import KAN
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import Capture_128
torch.manual_seed(2024)
kan_width = [128,32,13]
epochs = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = KAN(layers_hidden=kan_width)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
dataset = Capture_128(root='dataset/Capture_train_128.feather', isTrain=True, transform=transforms.ToTensor())
print(dataset.samples[:10])
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
print(model)
criterion = torch.nn.CrossEntropyLoss()
loss_hist = []
for epoch in tqdm(range(epochs)):
    running_loss = []
    print(f'Start Epoch {epoch}/{epochs}', end=" ")
    for batch_idx, (samples, labels) in enumerate(train_loader):
        samples.to(device)
        labels.unsqueeze(1).to(device)
        optimizer.zero_grad()

        outputs = model(samples)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    avg_loss = sum(running_loss)/len(running_loss)
    print(f'Epoch {epoch}/{epochs}, Loss: {avg_loss}')
    loss_hist.append(avg_loss)
print("Saving model...")
torch.save(model,"model/kan.pt")
plt.plot(range(epochs),loss_hist)
plt.title("Loss per epochs")
plt.savefig('output.png')

