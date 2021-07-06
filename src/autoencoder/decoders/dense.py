import torch
from torch import nn
import torch.nn.functional as F


class DenseDecoder(nn.Module):
    def __init__(
        self, 
        latent_shape, 
        output_shape, 
        node_count=512,
    ):
        super(DenseDecoder, self).__init__()
        self.output_shape = output_shape
        self.flattened_size = torch.prod(torch.tensor(output_shape), 0)

        input_count = latent_shape
        layers = []
        while(input_count < node_count):
            layers.append(nn.Linear(input_count, input_count*2))
            layers.append(nn.LeakyReLU())

            input_count *= 2

        self.dense = nn.ModuleList(layers) # registers modules

        self.fc = nn.Linear(node_count, self.flattened_size)

    def forward(self, x):
        for layer in self.dense:
            x = layer(x)

        x = torch.sigmoid(self.fc(x))
        x = x.view(-1, *self.output_shape)
        return x

