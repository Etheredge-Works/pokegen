from typing_extensions import ParamSpec
import torch
from torch import nn
from torch.nn import functional as F


class ConvEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_shape):
        super(ConvEncoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        # TODO enlarge kernel
        conv_layers = [
            nn.Conv2d(input_shape[0], 16, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        ]
        self.convs = nn.ModuleList(conv_layers)
        self.flatten = nn.Flatten()
        # Pooling here
        # TODO test laten features from conv layer
        self.fc = nn.Linear(128, latent_shape)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)

        # TODO was pooling used just to make it easier to construct model? no mathing dims
        # Pooling
        x = x.mean([2, 3])

        x = self.flatten(x)
        x = self.fc(x)

        return x
