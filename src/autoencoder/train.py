from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import utils
from pathlib import Path
import dvclive
import yaml


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# TODO why does 0 go to gpu1, how does torch order gpus?
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_ae(
    log_dir: str,
    epochs: int,
    trainloader: DataLoader,
    ae,
):
    log_dir = Path(log_dir)
    gen_dir = log_dir/'gen'
    dvclive_dir = log_dir/'logs'
    gen_dir.mkdir(exist_ok=True, parents=True)

    dvclive.init(str(dvclive_dir), summary=True)

    assert len(trainloader) > 0

    ae = ae.to(device)

    # Reference random tensor
    # TODO repeat in shape
    random_tensors = torch.stack([
        torch.rand(ae.latent_size), # fix values
        torch.rand(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
    ]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        running_loss = 0
        total = 0 # use total as drop_last=True
        ae.train()
        for image in tqdm(trainloader):
            #print(data[0])
            image = image.to(device)
            y_pred = ae(image)

            loss = criterion(y_pred, image)

            optimizer.zero_grad()
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
            
            generations = ae.decoder(random_tensors)
            utils.save(
                generations.cpu(),
                str(gen_dir),
                epoch)
        dvclive.next_step()
    utils.make_gifs(str(gen_dir))