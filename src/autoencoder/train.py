from genericpath import exists
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from autoencoder.model import AutoEncoder
from torch import nn
import torch
import utils
import os
from pathlib import Path
import dvclive


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hypers
latent_size = 8
epochs=10
log_dir = Path('reports/autoencoder')
gen_dir = log_dir/'gen'
dvclive_dir = log_dir/'logs'

gen_dir.mkdir(exist_ok=True, parents=True)

dvclive.init(str(dvclive_dir), summary=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5)),
])
train = torchvision.datasets.FashionMNIST('/tmp', download=True, train=True, transform=transform)
trainloader = DataLoader(
    train, 
    batch_size=128, 
    shuffle=True, 
    drop_last=True, 
    num_workers=8,
    persistent_workers=True, # makes short epochs start faster
    pin_memory=True
)

ae = AutoEncoder((1, 28, 28), latent_size)
ae = ae.to(device)

# Reference random tensor
rt = torch.rand(latent_size).to(device) # fix values

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
for epoch in range(epochs):
    print(f"{epoch}/{epochs}")
    running_loss = 0
    total = 0 # use total as drop_last=True
    ae.train()
    for image, label in tqdm(trainloader):
        optimizer.zero_grad()
        #print(data[0])
        image = image.to(device)
        y_pred = ae(image)

        loss = criterion(y_pred, image)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += image.size(0)
    print(f"loss: {running_loss/total}")
    dvclive.log("loss", running_loss, epoch)
    ae.eval()
    with torch.no_grad():
        #for idx in [0, 100, 1000]:
            #im = transforms.ToPILImage()(ae(train[idx][0].to(device))[0].cpu().data)
            #im.show()
        
        utils.save(
            ae.decoder(rt.to(device))[0].cpu().data, 
            str(gen_dir/f"{epoch}.jpg"))
    #dvclive.next_step()