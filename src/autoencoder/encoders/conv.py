from typing_extensions import ParamSpec
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList


class ConvEncoder(torch.nn.Module):
    def __init__(
        self, 
        input_shape, 
        latent_shape,
        activation_regularization_func=lambda _: 0
    ):
        super(ConvEncoder, self).__init__()

        self.act_reg_func = activation_regularization_func

        flattened_size = torch.prod(torch.tensor(input_shape), 0)
        # TODO enlarge kernel
        conv_layers = [
            (
                nn.Conv2d(input_shape[0], 16, 3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(16)
            ),
            # (
            #     nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=True),
            #     nn.BatchNorm2d(16)
            # ),
            (
                nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(32)
            ),
            # (
            #     nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True),
            #     nn.BatchNorm2d(32)
            # ),
            (
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(64)
            ),
            # (
            #     nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True),
            #     nn.BatchNorm2d(64)
            # ),
            (
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(128)
            ),
            # (
            #     nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True),
            #     nn.BatchNorm2d(128)
            # ),
            (
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(256)
            ),
            # (
            #     nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            #     nn.BatchNorm2d(256)
            # ),
            #(
                #nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
                #nn.BatchNorm2d(512)
            #),
        ]
        self.convs = nn.ModuleList([nn.ModuleList(layer_group) for layer_group in conv_layers])
        self.flatten = nn.Flatten()

        # TODO test laten features from conv layer
        self.fc = nn.Linear(256*6*6, latent_shape)
        self.activations_total = None

    def forward(self, x):

        #for layer, batch_norm in self.convs:
        for layer, batch_norm in self.convs:
            x = layer(x)
            #x = batch_norm(x)
            F.leaky_relu_(x)
            self.activations_total += self.act_reg_func(x)

        # TODO was pooling used just to make it easier to construct model? no mathing dims
        # Pooling
        #x = x.mean([2, 3])

        x = self.flatten(x)
        x = self.fc(x)
        #total += self.act_reg_func(x) TODO here?

        return x
