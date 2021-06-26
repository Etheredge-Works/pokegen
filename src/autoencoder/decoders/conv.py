import torch
from torch import nn
from torch.nn import functional as F


class ConvDecoder(nn.Module):
    def __init__(self, latent_shape, output_shape):
        super(ConvDecoder, self).__init__()
        self.output_flatten_shape = torch.prod(torch.tensor(output_shape), 0).item()
        self.output_shape = output_shape
        # TODO math latent_to_shape form input shape and layers
        
        self.fc = nn.Linear(latent_shape, 128) # TODO math out better
        self.up = nn.Upsample(scale_factor=3)  # TODO make not manually determined
        self.conv1 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(32, 16, 5, stride=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 3, 5, stride=2, output_padding=1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), x.size(1), 1, 1) 
        x = self.up(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        return x
