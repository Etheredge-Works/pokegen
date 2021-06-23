from genericpath import exists
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import RandomHorizontalFlip, Resize
from tqdm import tqdm
from autoencoder.model import AutoEncoder
from torch import nn
import torch
import utils
import os
from pathlib import Path
import dvclive
import yaml


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('params.yaml') as f:
    raw_config =yaml.safe_load(f)
    config = raw_config['autoencoder']
    poke_config = raw_config['pokemon']

# Hypers
latent_size = config['latent_size']
epochs = config['epochs']
batch_size = config['batch_size']
log_dir = Path(config['log_dir'])
gen_dir = log_dir/'gen'
dvclive_dir = log_dir/'logs'

gen_dir.mkdir(exist_ok=True, parents=True)

dvclive.init(str(dvclive_dir), summary=True)

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #transforms.RandomErasing(),
])

from data.sprites import data

#train = torchvision.datasets.FashionMNIST('/tmp', download=True, train=True, transform=transform)
train = data.PokemonDataset(
    config['data_dir'],
    transform=transform
)

trainloader = DataLoader(
    train, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=8,
    persistent_workers=True, # 'True' makes short epochs start faster
    pin_memory=True
)
assert len(trainloader) > 0

ae = AutoEncoder((3, 96, 96), latent_size)
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
    for image in tqdm(trainloader):
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
    dvclive.log("loss", running_loss/total, epoch)
    ae.eval()
    with torch.no_grad():
        #for idx in [0, len(trainloader)//4, len(trainloader)//2]:
            #im.save(str(log_dir/'val'/epoch/f"{idx}.jpg"))
            # TODO don't loop, just do all
            #im = transforms.ToPILImage()(ae(train[idx].to(device))[0].cpu().data)
            #im.save(str(log_dir/'val'/epoch/f"gen_{idx}.jpg"))
        
        utils.save(
            ae.decoder(rt.to(device))[0].cpu().data, 
            str(gen_dir/f"{epoch}.jpg"))
    dvclive.next_step()
utils.make_gif(str(gen_dir), str(log_dir/'gen.gif'))