from autoencoder.train import train_ae
import torch
from torch import nn
from torch.nn import functional as F
import click
from pathlib import Path
import yaml
from data import sprites


class Encoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape):
        super(Encoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.f = nn.Flatten()
        self.dense1 = nn.Linear(flattened_size, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.fc = nn.Linear(64, latent_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = self.f(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.relu(self.fc(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.flattened_size = torch.prod(torch.tensor(output_shape), 0)

        self.dense1 = nn.Linear(latent_shape, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 256)
        self.dense4 = nn.Linear(256, 512)
        self.fc = nn.Linear(512, self.flattened_size)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = F.relu(self.dense4(x))
        x = F.relu(self.fc(x))
        x = x.view(-1, *self.output_shape)
        return x


# TODO pull out
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_shape, latent_size):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# TODO pull out
@click.command()
def main():
    with open('params.yaml') as f:
        raw_config = yaml.safe_load(f)
        config = raw_config['autoencoder_dense']  # <----------- here is change
    ae = AutoEncoder((3, 96, 96), config['latent_size'])
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