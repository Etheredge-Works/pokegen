from autoencoder.encoders.conv import ConvEncoder
from autoencoder.train import train_ae
import torch
from torch import nn
from torch.nn import functional as F
import click
from pathlib import Path
import yaml
from data import sprites
from autoencoder.models import VAE
from autoencoder.decoders import ConvDecoder
from autoencoder.encoders import DenseEncoder


# TODO pull out to one function that takes args for vae, dense, etc
@click.command()
def main():
    with open('params.yaml') as f:
        raw_config = yaml.safe_load(f)
        config = raw_config['vae_conv']  # <----------- here is change
    ae = VAE(
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