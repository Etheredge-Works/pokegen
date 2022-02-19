import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import activation


class DenseEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_shape, 
        activation_regularization_func: lambda _: 0,
        node_count=256,
    ):
        super(DenseEncoder, self).__init__()

        self.activation_regularization_func = activation_regularization_func

        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        self.f = nn.Flatten()

        input_count = flattened_size
        layers = []
        while(node_count > latent_shape):
            layers.append(nn.Linear(input_count, node_count))

            input_count = node_count
            node_count = node_count // 2

        self.dense = nn.ModuleList(layers) # registers modules

        self.fc = nn.Linear(input_count, latent_shape)

        self.activations_total = None

    def forward(self, x):

        x = self.f(x)
        for layer in self.dense:
            x = layer(x)
            F.leaky_relu_(x)
            self.activations_total += self.activation_regularization_func(x)

        x = self.fc(x)
        self.activations_total += self.activation_regularization_func(x)
        return x

