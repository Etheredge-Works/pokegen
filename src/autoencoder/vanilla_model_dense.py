from autoencoder.decoders.dense import DenseDecoder
from autoencoder.train import train_ae
import torch
from torch import nn
from torch.nn import functional as F
import click
from pathlib import Path
import yaml
from data import sprites
from autoencoder.models import AutoEncoder
from autoencoder.decoders import DenseDecoder
from autoencoder.encoders import DenseEncoder


# TODO pull out to one function that takes args for vae, dense, etc
@click.command()
def main():
    with open('params.yaml') as f:
        raw_config = yaml.safe_load(f)
        config = raw_config['autoencoder_dense']  # <----------- here is change
    ae = AutoEncoder(
        (3, 96, 96), 
        config['latent_size'],
        DenseEncoder,
        DenseDecoder)
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