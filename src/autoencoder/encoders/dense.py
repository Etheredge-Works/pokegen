import torch
from torch import nn
from torch.nn import functional as F


class DenseEncoder(torch.nn.Module):
    def __init__(self, input_shape, latent_shape, node_count=1024):
        super(DenseEncoder, self).__init__()
        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.f = nn.Flatten()

        input_count = flattened_size
        layers = []
        while(node_count > latent_shape):
            layers.append(nn.Linear(input_count, node_count))
            layers.append(nn.ReLU())

            input_count = node_count
            node_count = node_count // 2

        self.dense = nn.ModuleList(layers) # registers modules

        self.fc = nn.Linear(input_count, latent_shape)

    def forward(self, x):
        x = self.f(x)
        for dense in self.dense:
            x = dense(x)
        x = self.fc(x)
        return x

