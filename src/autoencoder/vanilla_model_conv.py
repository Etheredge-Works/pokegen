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
        self.conv1 = nn.Conv2d(input_shape[0], 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2)
        self.f = nn.Flatten()
        # Pooling here
        self.fc = nn.Linear(64, latent_shape)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # TODO was pooling used just to make it easier to construct model? no mathing dims
        # Pooling
        x = x.mean([2, 3])

        x = self.f(x)
        x = F.relu(self.fc(x))
        #x = F.sigmoid(self.fc(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(Decoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        # TODO math latent_to_shape form input shape and layers
        
        self.latent_to_shape = (64, 5, 5)

        self.fc = nn.Linear(latent_shape, 64) # TODO math out better
        self.up = nn.Upsample(scale_factor=5)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.conv3 = nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.conv4 = nn.ConvTranspose2d(8, 3, 3, stride=2, output_padding=1)
        #self.conv1 = nn.ConvTranspose2d(64, 32)
        #self.dense = torch.nn.Linear()

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), x.size(1), 1, 1) 
        x = self.up(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.sigmoid(self.conv4(x))

        #x = torch.view(x, [-1, *self.output_shape])
        #x = x.view(-1, *self.output_shape)
        # TODO test speed view
        return x


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


@click.command()
def main():
    with open('params.yaml') as f:
        raw_config = yaml.safe_load(f)
        config = raw_config['autoencoder_conv']
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