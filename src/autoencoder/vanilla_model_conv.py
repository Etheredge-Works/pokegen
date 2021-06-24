from autoencoder.train import train_ae
from torch.nn import functional as F
import click
import yaml
from data import sprites
from autoencoder.models import AutoEncoder
from autoencoder.encoders import ConvEncoder
from autoencoder.decoders import ConvDecoder
# TODO why can't I use relative imports here?


@click.command()
def main():
    with open('params.yaml') as f:
        raw_config = yaml.safe_load(f)
        config = raw_config['autoencoder_conv']
    ae = AutoEncoder(
        (3, 96, 96), 
        config['latent_size'],
        ConvEncoder,
        ConvDecoder)
    loader = sprites.get_loader(
        batch_size=config['batch_size']
    )

    train_ae(
        ae=ae,
        trainloader=loader,
        **config['train_kwargs']
    )


if __name__ == "__main__":
    main()