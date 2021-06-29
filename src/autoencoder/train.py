from data import sprites
from autoencoder.models import AutoEncoder
from autoencoder.models import VAE
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import utils
from pathlib import Path
import dvclive
import yaml
import click
from autoencoder.encoders import DenseEncoder, ConvEncoder
from autoencoder.decoders import DenseDecoder, ConvDecoder
from torchsummary import summary
from contextlib import redirect_stdout
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO why does 0 go to gpu1, how does torch order gpus?
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_ae(
    log_dir: str,
    epochs: int,
    trainloader: DataLoader,
    ae: torch.nn.Module,
    lr=0.001,
    should_tqdm=os.getenv('SHOULD_TQDM', 1)  # using env for github action running 
):
    log_dir = Path(log_dir)
    gen_dir = log_dir/'gen'
    dvclive_dir = log_dir/'logs'
    gen_dir.mkdir(exist_ok=True, parents=True)

    results_dir = log_dir/'results'
    results_dir.mkdir(exist_ok=True, parents=True)

    ae = ae.to(device)

    with open(log_dir/"summary.txt", 'w') as f:
        with redirect_stdout(f):
            summary(ae, input_size=(3, 96, 96))

    dvclive.init(str(dvclive_dir), summary=True)

    assert len(trainloader) > 0

    # Reference random tensor
    # TODO repeat in shape
    random_tensors = torch.stack([
        # NOTE by doing two of each, two are used at once for VAE
        torch.rand(ae.latent_size), # fix values
        torch.rand(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
    ]).to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs)

    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        running_loss = 0
        total = 0 # use total as drop_last=True
        ae.train()
        if int(should_tqdm) != 0:
            iter_trainloader = tqdm(trainloader)
        else:
            iter_trainloader = trainloader
        for image_b in iter_trainloader:
            #print(data[0])
            image_b = image_b.to(device)
            y_pred = ae(image_b)

            loss = ae.criterion(y_pred, image_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += image_b.size(0)

        print(f"loss: {running_loss/total}")
        dvclive.log("loss", running_loss/total, epoch)
        ae.eval()
        with torch.no_grad():
            #for idx in [0, len(trainloader)//4, len(trainloader)//2]:
                #im.save(str(log_dir/'val'/epoch/f"{idx}.jpg"))
                # TODO don't loop, just do all
                #im = transforms.ToPILImage()(ae(train[idx].to(device))[0].cpu().data)
                #im.save(str(log_dir/'val'/epoch/f"gen_{idx}.jpg"))
            
            if epoch % (epochs//30) == 0 or epoch == (epochs-1):
                generations = ae.generate(random_tensors)
                utils.save(
                    generations.cpu(),
                    str(gen_dir),
                    epoch)
        dvclive.log("lr", scheduler.get_last_lr()[0])
        dvclive.next_step()
        scheduler.step()
    utils.make_gifs(str(gen_dir))

    # save off some final results
    batch = next(iter(trainloader))
    ae.eval()
    with torch.no_grad():
        utils.save(
            batch[:8].cpu(), # slice after incase of batch norm or something
            str(results_dir),
            'raw'
        )
        results = ae.predict(batch.to(device))
        utils.save(
            results[:8].cpu(), # slice after incase of batch norm or something
            str(results_dir),
            'encdec'
        )



@click.command()
@click.option("--encoder-type", type=click.STRING)
@click.option("--decoder-type", type=click.STRING)
@click.option("--ae-type", type=click.STRING)
@click.option("--log-dir", type=click.Path())
@click.option("--latent-size", type=click.INT)
@click.option("--epochs", type=click.INT)
@click.option("--lr", type=click.FLOAT)
@click.option("--batch-size", type=click.INT)
def main(
    encoder_type,
    decoder_type,
    ae_type,
    log_dir,
    latent_size,
    epochs,
    lr,
    batch_size,
):
    encoder_const = DenseEncoder if encoder_type == 'dense' else ConvEncoder
    decoder_const = DenseDecoder if decoder_type == 'dense' else ConvDecoder

    # TODO pull out so train file doesn't need these imported
    model_const = VAE if ae_type == 'vae' else AutoEncoder

    # TODO pull out shape
    ae = model_const(
        (3, 96, 96), 
        latent_size, encoder_const, 
        decoder_const)

    loader = sprites.get_loader(batch_size=batch_size)
    print(lr)

    train_ae(
        log_dir=log_dir, 
        epochs=epochs, 
        trainloader=loader, 
        ae=ae,
        lr=lr)


if __name__ == "__main__":
    main()