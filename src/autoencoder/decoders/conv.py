import torch
from torch import nn
from torch.nn import functional as F


class ConvDecoder(nn.Module):
    def __init__(
        self, 
        latent_shape, 
        output_shape,
        activation_regularization_func=lambda _: 0
    ):
        super(ConvDecoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        self.act_reg_func = activation_regularization_func
        # TODO math latent_to_shape form input shape and layers
        
        self.fc = nn.Linear(latent_shape, 128) # TODO math out better
        conv_layers = [
            # Upscale
            nn.ConvTranspose2d(128, 128, 3, stride=1),

            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(32, 32, 3, stride=1,padding=1),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),

        ]
        self.convs = nn.ModuleList(conv_layers)
        self.final_conv = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)
        self.activations_total = None

    def forward(self, x):
        if self.activations_total is None:
            self.activations_total = torch.tensor([0.]).to(x.device)

        x = F.leaky_relu(self.fc(x))
        self.activations_total += self.act_reg_func(x)
        x = x.view(x.size(0), x.size(1), 1, 1) 

        for layer in self.convs:
            x = layer(x)
            F.leaky_relu_(x)
            self.activations_total += self.act_reg_func(x)


        x = self.final_conv(x)
        x = torch.sigmoid(x)
        #activations.append(x) TODO here?

        return x
