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
torch.autograd.set_detect_anomaly(True) 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO why does 0 go to gpu1, how does torch order gpus?
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
from tensorboardX import SummaryWriter
#logger = SummaryWriter('tensorboard/')
logger = SummaryWriter()
    
def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)
        # TODO add weights
        

def train_ae(
    model_path: str,
    log_dir: str,
    epochs: int,
    trainloader: DataLoader,
    valloader: DataLoader,
    ae: torch.nn.Module,
    lr=0.001,
    should_tqdm=os.getenv('SHOULD_TQDM', 1),  # using env for github action running 
    gen_gifs=False
):
    log_dir = Path(log_dir)
    gen_dir = log_dir/'gen'
    latent_dir = log_dir/'latent'
    dvclive_dir = log_dir/'logs'
    gen_dir.mkdir(exist_ok=True, parents=True)
    latent_dir.mkdir(exist_ok=True, parents=True)

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
        torch.rand(ae.latent_size)*3, # fix values
        torch.rand(ae.latent_size)*3, # fix values
        torch.rand(ae.latent_size), # fix values
        torch.rand(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
        torch.randn(ae.latent_size), # fix values
        *torch.unbind(torch.distributions.Normal(
            torch.zeros(ae.latent_size), torch.ones(ae.latent_size)
            ).rsample((4,)))
    ]).to(device)

    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( NOTE cosine decay seems not as good
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=epochs)

    for epoch in range(epochs):
        #print(f"{epoch}/{epochs}")
        running_loss = 0
        total = 0 # use total as drop_last=True
        ae.train()
        if int(should_tqdm) != 0:
            iter_trainloader = tqdm(trainloader)
        else:
            iter_trainloader = trainloader

        latent_spaces = []
        for step, data in enumerate(iter_trainloader):

            transformed_image_b, label_b = data['transformed_image'], data['label']
            transformed_image_b = transformed_image_b.to(device)
            label_b = label_b.to(device)

            y_pred = ae(transformed_image_b)

            loss = ae.criterion(y_pred, transformed_image_b)

            loss.backward()
            #if int(should_tqdm) != 0:
                #log_gradients_in_model(ae, logger, step + epoch * len(trainloader))
            #torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1) #TODO param
            optimizer.step()

            running_loss += loss.item()
            total += transformed_image_b.size(0)

            latent_encoding = ae.latent
            #print(latent_encoding.shape)
            latent_spaces += latent_encoding

            optimizer.zero_grad()
            ae.reset()

        #assert len(latent_spaces) == len(trainloader), f"{len(latent_spaces)} != {len(trainloader.dataset)}"
        utils.save_latents(
            latent_spaces,
            ae.latent_size,
            str(latent_dir/"train"),
        )

        #print(f"loss: {running_loss/total}")
        dvclive.log("loss", running_loss/total, epoch)
        dvclive.log("lr", scheduler.get_last_lr()[0])
        ae.eval()
        val_latent_spaces = []
        with torch.no_grad():
            running_loss = 0
            total = 0
            for data in valloader:
                label_b = data['label']

                label_b = label_b.to(device)
                y_pred = ae(label_b)

                loss = ae.criterion(y_pred, label_b)

                running_loss += loss.item()
                total += label_b.size(0)
            
            if epoch % (epochs//10) == 0 or epoch == (epochs-1):
                generations = ae.generate(random_tensors)
                utils.save_image(
                    generations.cpu(),
                    str(gen_dir),
                    epoch)

            val_latent_encoding = ae.encoder(label_b).detach().cpu()
            val_latent_spaces += val_latent_encoding.tolist()
            utils.save_latents(
                val_latent_spaces,
                ae.latent_size,
                str(latent_dir/"val"),
            )
        
        dvclive.log("val_loss", running_loss/total, epoch)
        #print(f"val_loss: {running_loss/total}")

        dvclive.next_step()
        scheduler.step()

        ae.epoch_reset()
    
        # save off some results
        data = next(iter(trainloader))
        batch = data['label']
        ae.eval()
        # TODO torchvision.make_grid
        with torch.no_grad():
            utils.save_image(
                # TODO unhardcode
                batch[:8].cpu(), # slice after incase of batch norm or something
                str(results_dir),
                'raw'
            )
            results = ae.predict(batch.to(device))
            utils.save_image(
                results[:8].cpu(), # slice after incase of batch norm or something
                str(results_dir),
                'encoded'
            )
    if gen_gifs:
        utils.make_gifs(str(gen_dir))



@click.command()
@click.option("--encoder-type", type=click.STRING)
@click.option("--decoder-type", type=click.STRING)
@click.option("--ae-type", type=click.STRING)
@click.option("--model-path", type=click.STRING)
@click.option("--log-dir", type=click.Path())
@click.option("--latent-size", type=click.INT)
@click.option("--epochs", type=click.INT)
@click.option("--lr", type=click.FLOAT)
@click.option("--batch-size", type=click.INT)
@click.option("--val-ratio", type=click.FLOAT)
@click.option("--reg-type", type=click.STRING)
@click.option("--reg-rate", type=click.FLOAT)
@click.option("--gen-gifs", type=click.BOOL)
def main(
    encoder_type,
    decoder_type,
    ae_type,
    model_path,
    log_dir,
    latent_size,
    epochs,
    lr,
    batch_size,
    val_ratio,
    reg_type,
    reg_rate,
    gen_gifs
):

    # TODO pull out so train file doesn't need these imported
    model_const = VAE if ae_type == 'vae' else AutoEncoder

    # TODO pull out shape
    ae = model_const(
        (3, 96, 96), 
        latent_size, 
        reg_type,
        reg_rate,
        encoder_type, 
        decoder_type)

    trainloader, valloader = sprites.get_loader(
        batch_size=batch_size,
        workers=4,
        val_ratio=val_ratio)

    train_ae(
        model_path=model_path,
        log_dir=log_dir, 
        epochs=epochs, 
        trainloader=trainloader, 
        valloader=valloader, 
        ae=ae,
        lr=lr,
        gen_gifs=gen_gifs)

    # Save model
    torch.save(ae.state_dict(), str(model_path)+'.pt')
    with open(str(model_path)+"_kwargs.yaml", 'w') as f:
        #yaml.dump(locals(), f)  NOTE cool locals() thing
        kwargs = {
            "input_shape": (3, 96, 96),
            "latent_size": latent_size,
            "reg_type": reg_type,
            "reg_rate": reg_rate,
            "encoder_type": encoder_type,
            "decoder_type": decoder_type
        }
        yaml.dump(kwargs, f)


if __name__ == "__main__":
    main()